"""定位训练热点：对 LM 单步 forward+backward 做 cProfile 统计。"""
import cProfile
import pstats
import io
import sys
import time

import numpy as np

from hyperion.arch import ModelConfig, Transformer
from hyperion.corpus import build_corpus, corpus_text
from hyperion.tokenizer import train_bpe
from hyperion.optim import AdamW
from hyperion import engine as E

cfg = ModelConfig(vocab_size=2048, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4, d_ff=512, max_seq=256)
kb, lm_texts, pairs = build_corpus(seed=20260923, variants=8)
tok = train_bpe([corpus_text(lm_texts)], vocab_size=2048)
ids = np.asarray(tok.encode(corpus_text(lm_texts)), dtype=np.int64)
print("tokens:", len(ids), flush=True)

model = Transformer(cfg, seed=1)
opt = AdamW(model.param_list(), lr=1e-3)
B, T = 16, 96
rng = np.random.default_rng(0)


def one_step():
    starts = rng.integers(0, max(len(ids) - T - 1, 1), size=B)
    x = np.stack([ids[s:s + T] for s in starts])
    y = np.stack([ids[s + 1:s + T + 1] for s in starts])
    opt.zero_grad()
    loss, _, _ = model.forward(x, targets=y)
    loss.backward()
    opt.step(1.0)
    return float(loss.data)


# 预热
one_step()
t0 = time.time()
one_step()
t1 = time.time()
print(f"单步（含 opt.step）耗时: {(t1 - t0) * 1000:.1f} ms", flush=True)

# 分段计时
starts = rng.integers(0, max(len(ids) - T - 1, 1), size=B)
x = np.stack([ids[s:s + T] for s in starts])
y = np.stack([ids[s + 1:s + T + 1] for s in starts])
opt.zero_grad()
t = time.time(); loss, logits, _ = model.forward(x, targets=y); t_fwd = time.time() - t
t = time.time(); loss.backward(); t_bwd = time.time() - t
t = time.time(); opt.step(1.0); t_opt = time.time() - t
print(f"forward {t_fwd*1000:.1f} ms | backward {t_bwd*1000:.1f} ms | opt.step {t_opt*1000:.1f} ms", flush=True)

pr = cProfile.Profile()
pr.enable()
for _ in range(3):
    one_step()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(25)
print(s.getvalue(), flush=True)
