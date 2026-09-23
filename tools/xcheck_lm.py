"""双引擎交叉验证：用 Python 侧加载导出权重，复算 LM loss 与 greedy 生成。

作者：晨星

用途：区分「训练侧缺陷」与「推理侧缺陷」。
同一份 .hwm 权重在 Python autograd 前向里重跑一遍：
  * LM loss 若远高于 LM 阶段结束时的 0.08x  -> 说明后续对比学习阶段把 LM 能力覆盖了
  * LM loss 若仍然很低而生成仍退化          -> 说明问题在推理侧或分词契约
这是把「模型坏了」压缩成可机械定位结论的判据。
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np


def _ensure_path(root: str) -> None:
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> int:
    ap = argparse.ArgumentParser("xcheck-lm")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap.add_argument("--model", default=os.path.join(root, "assets", "model.f32.hwm"))
    ap.add_argument("--tokenizer", default=os.path.join(root, "assets", "tokenizer.json"))
    ap.add_argument("--lm", default=os.path.join(root, "assets", "lm.txt"))
    ap.add_argument("--windows", type=int, default=8, help="评估 LM loss 的随机窗口数")
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--gen", type=int, default=24)
    args = ap.parse_args()

    _ensure_path(os.path.join(root, "src", "py"))
    from hyperion.load import load_hwm
    from hyperion.arch import ModelConfig, Transformer
    from hyperion.tokenizer import load_tokenizer

    meta, tensors = load_hwm(args.model)
    print(f"已加载：{os.path.basename(args.model)}  dtype={meta['dtype']}  张量 {meta['n_tensors']}  CRC 校验通过")

    # 只保留 ModelConfig 认识的字段，避免 arch 里多出字段导致构造失败
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    arch_kwargs = {k: v for k, v in meta["arch"].items() if k in field_names}
    cfg = ModelConfig(**arch_kwargs)
    model = Transformer(cfg, seed=0)

    missing, loaded = [], []
    for k, v in tensors.items():
        if k in model.params:
            if model.params[k].data.shape != v.shape:
                raise SystemExit(f"形状不符 {k}: 期望 {model.params[k].data.shape}，加载 {v.shape}")
            model.params[k].data[:] = v
            loaded.append(k)
        else:
            missing.append(k)
    print(f"灌入 {len(loaded)} 个张量；非 Transformer 参数（投影头）{len(missing)} 个：{sorted(missing)}")

    tok = load_tokenizer(args.tokenizer)

    # ---------- 1) 最终权重下的 LM loss ----------
    with open(args.lm, encoding="utf-8") as f:
        text = f.read()
    ids = np.asarray(tok.encode(text), dtype=np.int64)
    print(f"LM 语料 tokens：{len(ids)}")

    rng = np.random.default_rng(20260923)
    T = args.seq
    losses = []
    for _ in range(args.windows):
        s = int(rng.integers(0, max(len(ids) - T - 1, 1)))
        x = ids[s : s + T][None, :]
        y = ids[s + 1 : s + T + 1][None, :]
        loss, _, _ = model.forward(x, targets=y)
        losses.append(float(loss.data))
    avg = float(np.mean(losses))
    print(f"\n[最终权重 LM loss] {avg:.5f}  ppl {np.exp(min(avg, 20)):.3f}")
    if avg < 1.0:
        print("[诊断] ✅ LM 能力已保留：联合多任务阶段的 LM 回放生效，生成退化应已修复。")
    elif avg < 2.0:
        print("[诊断] ⚠️ LM loss 偏高：联合阶段 LM 回放不足，建议上调 --joint-steps 或 --joint-lm-ratio。")
    else:
        print("[诊断] ❌ 灾难性遗忘仍在：LM loss 远超阶段末 (~0.08)，需大幅加强联合阶段 LM 占比。")

    # ---------- 2) greedy 生成 ----------
    prompts = [
        "问：inference-runtime 出现 ECONNREFUSED 的排查步骤是什么\n答：",
        "问：推理运行时需要重点观测哪些指标\n答：",
    ]
    for p in prompts:
        seq = list(tok.encode(p))
        gen_ids: list[int] = []
        for _ in range(args.gen):
            x = np.asarray([seq], dtype=np.int64)
            _, logits, _ = model.forward(x)
            nxt = int(np.argmax(logits.data[0, -1]))
            gen_ids.append(nxt)
            seq.append(nxt)
        print(f"\n[greedy] {p!r}")
        print(f"  -> {tok.decode(gen_ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
