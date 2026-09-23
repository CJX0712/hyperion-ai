"""加载器自检：验证 load_hwm 还原的权重与 TS 推理侧语义一致。

作者：晨星

判据很直接：TS 侧已知 `${first query}` 能检索到 gold 文档；
如果 Python 用 load_hwm 读出来的权重跑同一套双塔检索也命中同一批 gold，
就证明"反量化 + 转置还原"这条链路是对的，后续基于 Python 侧测出的
LM loss 结论才可信。

用法：PYTHONPATH=src/py python tools/xcheck_embed.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser("xcheck-embed")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    assets = os.path.join(root, "assets")
    ap.add_argument("--model", default=os.path.join(assets, "model.f32.hwm"))
    ap.add_argument("--tokenizer", default=os.path.join(assets, "tokenizer.json"))
    ap.add_argument("--corpus", default=os.path.join(assets, "corpus.jsonl"))
    ap.add_argument("--qa", default=os.path.join(assets, "qa.jsonl"))
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=160)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(root, "src", "py"))
    from hyperion.load import load_hwm
    from hyperion.arch import ModelConfig, Transformer
    from hyperion.tokenizer import load_tokenizer

    meta, tensors = load_hwm(args.model)
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in meta["arch"].items() if k in field_names})
    model = Transformer(cfg, seed=0)
    for k, v in tensors.items():
        if k in model.params:
            model.params[k].data[:] = v
    proj = tensors["emb_proj"]  # (d_model, embed_dim)
    tok = load_tokenizer(args.tokenizer)

    def embed(text: str) -> np.ndarray:
        ids = tok.encode(text)[: args.max_len]
        if not ids:
            return np.zeros(proj.shape[1], dtype=np.float32)
        x = np.asarray([ids], dtype=np.int64)
        hidden = model.encode(x).data[0]  # (T, d_model)
        pooled = hidden.mean(axis=0)
        e = pooled @ proj
        n = float(np.linalg.norm(e))
        return (e / n) if n > 0 else e

    docs = [json.loads(l) for l in open(args.corpus, encoding="utf-8") if l.strip()]
    qa = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    print(f"语料 {len(docs)} 篇 / 查询 {len(qa)} 条；emb_proj 形状 {proj.shape}")

    doc_vecs = np.stack([embed(d["text"]) for d in docs])
    hit1 = hit3 = 0
    n = min(args.cases, len(qa))
    for i in range(n):
        row = qa[i]
        qv = embed(row["query"])
        sims = doc_vecs @ qv
        order = np.argsort(-sims)
        top1 = docs[int(order[0])]["doc_id"]
        gold = row["gold_doc_id"]
        ok1 = top1 == gold
        hit1 += int(ok1)
        hit3 += int(gold in [docs[int(j)]["doc_id"] for j in order[:3]])
        rank = int(np.where([docs[int(j)]["doc_id"] == gold for j in order])[0][0]) + 1
        print(f"  [{i}] q={row['query'][:28]!r} gold={gold} top1={top1} rank={rank} {'OK' if ok1 else 'MISS'}")
    print(f"\nPython 侧 top-1 命中 {hit1}/{n}，top-3 命中 {hit3}/{n}")
    print("（TS 侧黄金集 recall@5 达标 >=0.80；若此处同样高命中，则加载器与两侧权重语义一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
