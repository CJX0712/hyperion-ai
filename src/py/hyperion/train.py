"""Hyperion AI - Python 训练内核主入口

作者：晨星

分四个阶段训练，全部在 numpy 上完成（无 torch、无 GPU、无编译）：
  1. LM 阶段：字节级 BPE + 自回归语言建模（交叉熵）
  2. 双塔阶段：共享编码器 + 均值池化 + 投影头，InfoNCE 对比学习
  3. 交叉编码器阶段：共享编码器 + 打分头，组内 softmax 排序损失
  4. 联合多任务阶段（关键修复）：每步随机选 LM / 双塔 / 交叉编码器 之一，
     共用一个优化器持续训练。BE/CE 的对比损失会把共享 Transformer 的主干
     改写成「所有 token 向量趋同」的空间，从而灾难性遗忘 LM 能力——这正是
     此前生成退化为重复符号的根因。联合阶段用高频 LM 回放阻止遗忘，
     让同一套权重同时具备「会续写」和「会检索」两种能力。

产物：assets/tokenizer.json、assets/model.hmf、assets/corpus.jsonl、assets/qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .arch import ModelConfig, Transformer
from . import engine as E
from .corpus import build_corpus, export_corpus
from .engine import Tensor
from .export import export_hwm
from .optim import AdamW, cosine_schedule
from .tokenizer import SPECIALS, BPETokenizer, save_tokenizer, train_bpe

PAD_ID = SPECIALS.index("<pad>")
SEP_ID = SPECIALS.index("<sep>")


def l2_normalize(x: Tensor) -> Tensor:
    n = np.linalg.norm(x.data, axis=-1, keepdims=True)
    out = Tensor(x.data / (n + 1e-8), requires_grad=x.requires_grad, _prev=(x,))

    def _bw() -> None:
        g, y = out.grad, out.data
        dot = np.sum(g * y, axis=-1, keepdims=True)
        x.grad = (x.grad + (g - y * dot) / (n + 1e-8)).astype(np.float32)

    out._backward = _bw
    return out


def mean_pool(hidden: Tensor, mask: np.ndarray) -> Tensor:
    """mask: (B, T, 1) float32，1 表示有效 token。"""
    m = Tensor(mask.astype(np.float32))
    summed = E.sum_to(E.mul(hidden, m), axis=1)
    counts = np.maximum(mask.sum(axis=1), 1.0).astype(np.float32)  # (B, 1)
    return E.mul(summed, Tensor(1.0 / counts))


def pad_batch(seqs: Sequence[Sequence[int]], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    B = len(seqs)
    ids = np.full((B, max_len), PAD_ID, dtype=np.int64)
    mask = np.zeros((B, max_len, 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        ids[i, :n] = s[:n]
        mask[i, :n, 0] = 1.0
    return ids, mask


# --------------------------------------------------------------------------- 阶段 1
def train_lm(
    model: Transformer, tok: BPETokenizer, token_ids: np.ndarray, args
) -> Dict:
    rng = np.random.default_rng(args.seed + 1)
    params = model.param_list()
    opt = AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, grad_clip=args.grad_clip)
    B, T = args.batch, args.seq
    n = len(token_ids)
    log: List[Dict] = []
    t0 = time.time()
    best = float("inf")
    for step in range(args.lm_steps):
        starts = rng.integers(0, max(n - T - 1, 1), size=B)
        x = np.stack([token_ids[s : s + T] for s in starts])
        y = np.stack([token_ids[s + 1 : s + T + 1] for s in starts])
        opt.zero_grad()
        loss, _, _ = model.forward(x, targets=y)
        loss.backward()
        lr_scale = cosine_schedule(step, args.lm_steps, args.warmup)
        gnorm = opt.step(lr_scale)
        lv = float(loss.data)
        best = min(best, lv)
        if step % args.log_every == 0 or step == args.lm_steps - 1:
            el = time.time() - t0
            rec = {
                "step": step,
                "loss": round(lv, 5),
                "ppl": round(float(np.exp(min(lv, 20))), 3),
                "gnorm": round(gnorm, 4),
                "lr_scale": round(lr_scale, 4),
                "elapsed_s": round(el, 1),
            }
            log.append(rec)
            print(f"[LM] {rec}", flush=True)
        if args.time_budget and (time.time() - t0) > args.time_budget:
            print(f"[LM] 触发时间预算 {args.time_budget}s，提前停止于 step {step}", flush=True)
            break
    return {"log": log, "final_loss": log[-1]["loss"] if log else None, "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- 阶段 2
def train_biencoder(model: Transformer, tok: BPETokenizer, docs, pairs, args) -> Dict:
    rng = np.random.default_rng(args.seed + 2)
    doc_text = {d.doc_id: d.text for d in docs}
    D = model.cfg.d_model
    emb_proj = Tensor(rng.normal(0, 0.05, (D, args.embed_dim)).astype(np.float32), True, name="emb_proj")
    params = model.param_list() + [emb_proj]
    opt = AdamW(params, lr=args.lr * 0.5, betas=(0.9, 0.95), weight_decay=args.weight_decay, grad_clip=args.grad_clip)

    enc_q = [tok.encode(p.query)[: args.q_len] for p in pairs]
    enc_d = [tok.encode(doc_text[p.gold_doc_id])[: args.doc_len] for p in pairs]
    idx = np.arange(len(pairs))
    tau = np.float32(1.0 / args.temperature)
    log: List[Dict] = []
    t0 = time.time()
    steps = args.emb_steps
    for step in range(steps):
        sel = rng.choice(idx, size=args.be_batch, replace=False)
        q_ids, q_mask = pad_batch([enc_q[i] for i in sel], args.q_len)
        d_ids, d_mask = pad_batch([enc_d[i] for i in sel], args.doc_len)
        opt.zero_grad()
        _, _, hq = model.forward(q_ids)
        _, _, hd = model.forward(d_ids)
        pq = l2_normalize(E.matmul(mean_pool(hq, q_mask), emb_proj))
        pd = l2_normalize(E.matmul(mean_pool(hd, d_mask), emb_proj))
        logits = E.scale(E.matmul(pq, E.permute(pd, (1, 0))), tau)
        loss = E.cross_entropy(logits, np.arange(args.be_batch))
        loss.backward()
        lr_scale = cosine_schedule(step, steps, max(args.warmup // 2, 5))
        opt.step(lr_scale)
        if step % args.log_every == 0 or step == steps - 1:
            lv = float(loss.data)
            # 训练中即时看准确率：对角线是否为最大
            acc = float(np.mean(np.argmax(logits.data, axis=-1) == np.arange(args.be_batch)))
            rec = {"step": step, "loss": round(lv, 5), "batch_acc": round(acc, 4)}
            log.append(rec)
            print(f"[BE] {rec}", flush=True)
        if args.time_budget and (time.time() - t0) > args.time_budget * 0.6:
            print(f"[BE] 触发时间预算，提前停止于 step {step}", flush=True)
            break
    return {"log": log, "seconds": round(time.time() - t0, 1), "proj": emb_proj}


# --------------------------------------------------------------------------- 阶段 3
def train_crossencoder(model: Transformer, tok: BPETokenizer, docs, pairs, args) -> Dict:
    rng = np.random.default_rng(args.seed + 3)
    doc_text = {d.doc_id: d.text for d in docs}
    D = model.cfg.d_model
    ce_score = Tensor(rng.normal(0, 0.02, (D, 1)).astype(np.float32), True, name="ce_score")
    params = model.param_list() + [ce_score]
    opt = AdamW(params, lr=args.lr * 0.3, betas=(0.9, 0.95), weight_decay=args.weight_decay, grad_clip=args.grad_clip)

    seq_len = args.q_len + 1 + args.doc_len
    # 每个训练样本 = 1 正例 + K 负例
    groups = []
    for i, p in enumerate(pairs):
        negs = list(p.hard_negatives)[: args.ce_negatives]
        while len(negs) < args.ce_negatives:
            j = int(rng.integers(0, len(pairs)))
            if pairs[j].gold_doc_id != p.gold_doc_id:
                negs.append(pairs[j].gold_doc_id)
        groups.append(
            [
                tok.encode(p.query)[: args.q_len] + [SEP_ID] + tok.encode(doc_text[p.gold_doc_id])[: args.doc_len]
            ]
            + [tok.encode(p.query)[: args.q_len] + [SEP_ID] + tok.encode(doc_text[n])[: args.doc_len] for n in negs]
        )

    G = args.ce_groups
    K = args.ce_negatives + 1
    log: List[Dict] = []
    t0 = time.time()
    steps = args.ce_steps
    for step in range(steps):
        sel = rng.choice(len(groups), size=G, replace=False)
        flat: List[List[int]] = []
        for gi in sel:
            flat.extend(groups[gi])
        ids, mask = pad_batch(flat, min(seq_len, model.cfg.max_seq))
        opt.zero_grad()
        _, _, hidden = model.forward(ids)
        pooled = mean_pool(hidden, mask)
        scores = E.matmul(pooled, ce_score)  # (G*K, 1)
        scores2 = E.reshape(scores, (G, K))
        loss = E.cross_entropy(scores2, np.zeros(G, dtype=np.int64))
        loss.backward()
        lr_scale = cosine_schedule(step, steps, max(args.warmup // 2, 5))
        opt.step(lr_scale)
        if step % args.log_every == 0 or step == steps - 1:
            acc = float(np.mean(np.argmax(scores2.data, axis=-1) == 0))
            rec = {"step": step, "loss": round(float(loss.data), 5), "top1_acc": round(acc, 4)}
            log.append(rec)
            print(f"[CE] {rec}", flush=True)
        if args.time_budget and (time.time() - t0) > args.time_budget * 0.5:
            print(f"[CE] 触发时间预算，提前停止于 step {step}", flush=True)
            break
    return {"log": log, "seconds": round(time.time() - t0, 1), "score": ce_score}


# --------------------------------------------------------------------------- 阶段 4：联合多任务（修复灾难性遗忘）
def train_joint(model, tok, token_ids, kb, pairs, emb_proj, ce_score, args) -> Dict:
    """联合多任务阶段。

    根因：LM→BE→CE 三阶段共用同一个 Transformer，BE/CE 的对比损失（均值池化 +
    L2 归一）会把共享主干改写成「所有 token 向量近似相等」的空间，从而把已学好的
    LM 动态彻底覆盖掉（实测最终 LM loss 从 0.08 退化到 3.15，生成退化为重复符号）。

    修复：每步随机选一个任务（LM / 双塔 / 交叉编码器），三者共用一个 AdamW 优化
    器。LM 以高比例（默认 0.6）持续回放，使共享主干在学会检索的同时不被遗忘；
    检索/重排头（emb_proj / ce_score）也在这个阶段继续微调，保证端到端指标不回退。
    """
    rng = np.random.default_rng(args.seed + 7)
    doc_text = {d.doc_id: d.text for d in kb}
    D = model.cfg.d_model
    params = model.param_list() + [emb_proj, ce_score]
    opt = AdamW(params, lr=args.lr * 0.5, betas=(0.9, 0.95), weight_decay=args.weight_decay, grad_clip=args.grad_clip)
    B, T = args.batch, args.seq
    n = len(token_ids)
    # 双塔编码缓存（query / doc 固定，只算一次）
    enc_q = [tok.encode(p.query)[: args.q_len] for p in pairs]
    enc_d = [tok.encode(doc_text[p.gold_doc_id])[: args.doc_len] for p in pairs]
    idx = np.arange(len(pairs))
    tau = np.float32(1.0 / args.temperature)
    # 交叉编码器组：每组 1 正例 + K 负例
    seq_len = args.q_len + 1 + args.doc_len
    groups: List[List[List[int]]] = []
    for p in pairs:
        negs = list(p.hard_negatives)[: args.ce_negatives]
        while len(negs) < args.ce_negatives:
            j = int(rng.integers(0, len(pairs)))
            if pairs[j].gold_doc_id != p.gold_doc_id:
                negs.append(pairs[j].gold_doc_id)
        groups.append(
            [tok.encode(p.query)[: args.q_len] + [SEP_ID] + tok.encode(doc_text[p.gold_doc_id])[: args.doc_len]]
            + [tok.encode(p.query)[: args.q_len] + [SEP_ID] + tok.encode(doc_text[n])[: args.doc_len] for n in negs]
        )
    G = args.ce_groups
    K = args.ce_negatives + 1
    steps = args.joint_steps
    lm_ratio = float(args.joint_lm_ratio)
    be_ratio = (1.0 - lm_ratio) / 2.0
    log: List[Dict] = []
    t0 = time.time()
    last_lm_loss = None
    for step in range(steps):
        task = rng.random()
        opt.zero_grad()
        if task < lm_ratio:
            starts = rng.integers(0, max(n - T - 1, 1), size=B)
            x = np.stack([token_ids[s : s + T] for s in starts])
            y = np.stack([token_ids[s + 1 : s + T + 1] for s in starts])
            loss, _, _ = model.forward(x, targets=y)
            loss.backward()
            last_lm_loss = float(loss.data)
            tname = "lm"
        elif task < lm_ratio + be_ratio:
            sel = rng.choice(idx, size=args.be_batch, replace=False)
            q_ids, q_mask = pad_batch([enc_q[i] for i in sel], args.q_len)
            d_ids, d_mask = pad_batch([enc_d[i] for i in sel], args.doc_len)
            _, _, hq = model.forward(q_ids)
            _, _, hd = model.forward(d_ids)
            pq = l2_normalize(E.matmul(mean_pool(hq, q_mask), emb_proj))
            pd = l2_normalize(E.matmul(mean_pool(hd, d_mask), emb_proj))
            logits = E.scale(E.matmul(pq, E.permute(pd, (1, 0))), tau)
            loss = E.cross_entropy(logits, np.arange(args.be_batch))
            loss.backward()
            tname = "be"
        else:
            sel = rng.choice(len(groups), size=G, replace=False)
            flat: List[List[int]] = []
            for gi in sel:
                flat.extend(groups[gi])
            ids, mask = pad_batch(flat, min(seq_len, model.cfg.max_seq))
            _, _, hidden = model.forward(ids)
            pooled = mean_pool(hidden, mask)
            scores = E.matmul(pooled, ce_score)
            scores2 = E.reshape(scores, (G, K))
            loss = E.cross_entropy(scores2, np.zeros(G, dtype=np.int64))
            loss.backward()
            tname = "ce"
        lr_scale = cosine_schedule(step, steps, max(args.warmup // 2, 10))
        gn = opt.step(lr_scale)
        if step % args.log_every == 0 or step == steps - 1:
            rec = {
                "step": step,
                "task": tname,
                "loss": round(float(loss.data), 5),
                "lm_loss": None if last_lm_loss is None else round(last_lm_loss, 5),
                "gnorm": round(gn, 4),
            }
            log.append(rec)
            print(f"[JOINT] {rec}", flush=True)
    return {
        "log": log,
        "seconds": round(time.time() - t0, 1),
        "proj": emb_proj,
        "score": ce_score,
        "final_lm_loss": last_lm_loss,
    }


# --------------------------------------------------------------------------- 主流程
def parse_args(argv=None):
    p = argparse.ArgumentParser("hyperion-train")
    p.add_argument("--out", default="assets")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--variants", type=int, default=8)
    # 模型（默认档：约 1.05M 参数，int8 后约 1.1MB，CPU 解码 ~200 tok/s）
    # 规模受内存约束：本沙箱可用内存约 4GB，自研 autograd 会保留全部中间激活，
    # 因此 batch*seq*d_ff 必须控制在 ~8e5 个元素以内。
    p.add_argument("--vocab", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--max-seq", type=int, default=256)
    # 优化。batch/be-batch/ce-groups 默认值按「自研 autograd 保留全部中间激活、
    # 目标机器为 16GB 内存无 GPU」标定：batch*seq*d_ff 控制在 ~4e5 元素以内，
    # 否则反向传播阶段会因内存 commit 耗尽而 OOM（详见 docs/ARCHITECTURE.md）。
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=96)
    p.add_argument("--lm-steps", type=int, default=900)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--time-budget", type=float, default=0.0, help="秒；0 表示不限")
    # 嵌入/重排头
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--emb-steps", type=int, default=150)
    p.add_argument("--ce-steps", type=int, default=150)
    p.add_argument("--be-batch", type=int, default=8)
    p.add_argument("--ce-groups", type=int, default=2)
    p.add_argument("--ce-negatives", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.07)
    # 阶段 4：联合多任务（修复灾难性遗忘）。lm_ratio = LM 任务占比，其余均分给 BE/CE
    p.add_argument("--joint-steps", type=int, default=600, help="联合阶段步数（LM 回放强度）")
    p.add_argument("--joint-lm-ratio", type=float, default=0.6, help="LM 任务在联合阶段中的随机占比")
    # CE 序列长同样受内存标定：注意力分数 (G*K, H, L, L) 全部进激活图，
    # L=24+1+96 时峰值约 1.9MB；L=32+1+160（约 4.8MB）实测在 16GB 共存机器上 commit 耗尽。
    p.add_argument("--q-len", type=int, default=24)
    p.add_argument("--doc-len", type=int, default=96)
    # 其它
    p.add_argument("--smoke", action="store_true", help="极小规模冒烟测试")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.lm_steps, args.emb_steps, args.ce_steps = 20, 8, 8
        args.joint_steps = 16
        args.log_every = 4
        args.vocab = 512
        args.batch = 8
    os.makedirs(args.out, exist_ok=True)

    print("== Hyperion AI 训练内核 ==", flush=True)
    kb, lm_texts, pairs = build_corpus(seed=args.seed, variants=args.variants)
    info = export_corpus(kb, lm_texts, pairs, args.out)
    print(f"语料：知识库 {info['docs']} 篇 / 查询 {info['pairs']} 条 / 训练文本 {info['lm_texts']} 段", flush=True)

    lm_corpus = "\n".join(lm_texts)
    print(f"LM 语料字符数：{len(lm_corpus)}", flush=True)
    t0 = time.time()
    tok = train_bpe([lm_corpus], vocab_size=args.vocab, verbose=True)
    save_tokenizer(tok, os.path.join(args.out, "tokenizer.json"))
    print(f"BPE 训练完成：{len(tok.merges)} 条合并，用时 {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    # 每段末尾追加 <eos>：教模型在完整表述后停止生成，避免退化式无限续写
    seg_ids = [tok.encode(t, add_eos=True) for t in lm_texts]
    token_ids = np.asarray([i for s in seg_ids for i in s], dtype=np.int64)
    print(f"编码完成：{len(token_ids)} tokens（含 EOS 分隔），用时 {time.time() - t0:.1f}s", flush=True)

    cfg = ModelConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_ff=args.d_ff,
        max_seq=args.max_seq,
    )
    model = Transformer(cfg, seed=args.seed)
    print(f"模型参数量：{model.num_params():,}", flush=True)

    r_lm = train_lm(model, tok, token_ids, args)
    r_be = train_biencoder(model, tok, kb, pairs, args)
    r_ce = train_crossencoder(model, tok, kb, pairs, args)
    # 阶段 4：联合多任务，修复 BE/CE 对 LM 的灾难性遗忘（关键修复）
    r_joint = train_joint(model, tok, token_ids, kb, pairs, r_be["proj"], r_ce["score"], args)

    tensors: Dict[str, np.ndarray] = {k: v.data for k, v in model.params.items()}
    tensors["emb_proj"] = r_joint["proj"].data
    tensors["ce_score"] = r_joint["score"].data

    meta = {
        "author": "晨星",
        "embed_dim": args.embed_dim,
        "seed": args.seed,
        "trained_tokens": int(len(token_ids)),
        "lm_steps": args.lm_steps,
        "joint_steps": args.joint_steps,
        "joint_lm_ratio": args.joint_lm_ratio,
        "final_lm_loss": r_joint["final_lm_loss"] if r_joint["final_lm_loss"] is not None else r_lm["final_loss"],
        "biencoder_batch_acc": r_be["log"][-1]["batch_acc"] if r_be["log"] else None,
        "crossencoder_top1_acc": r_ce["log"][-1]["top1_acc"] if r_ce["log"] else None,
    }
    stat_i8 = export_hwm(
        os.path.join(args.out, "model.i8.hwm"),
        arch=cfg.to_dict(),
        tensors=tensors,
        dtype="i8",
        tokenizer_ref="tokenizer.json",
        extra_meta=meta,
    )
    stat_f32 = export_hwm(
        os.path.join(args.out, "model.f32.hwm"),
        arch=cfg.to_dict(),
        tensors=tensors,
        dtype="f32",
        tokenizer_ref="tokenizer.json",
        extra_meta=meta,
    )
    stat = {"int8": stat_i8, "fp32": stat_f32}
    print(f"导出 int8：{stat_i8}", flush=True)
    print(f"导出 fp32：{stat_f32}", flush=True)

    report = {
        "corpus": info,
        "tokens": int(len(token_ids)),
        "model": cfg.to_dict(),
        "num_params": model.num_params(),
        "lm": {"seconds": r_lm["seconds"], "final_loss": r_lm["final_loss"], "log": r_lm["log"]},
        "biencoder": {"seconds": r_be["seconds"], "log": r_be["log"]},
        "crossencoder": {"seconds": r_ce["seconds"], "log": r_ce["log"]},
        "joint": {"seconds": r_joint["seconds"], "final_lm_loss": r_joint["final_lm_loss"], "log": r_joint["log"]},
        "export": stat,
    }
    with open(os.path.join(args.out, "train-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("训练完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
