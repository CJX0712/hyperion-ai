"""Hyperion AI - Transformer 架构

作者：晨星

复用的业界成果（架构层，非自研）：
  * RMSNorm（LLaMA / GPT-NeoX 采用的预归一化）
  * RoPE 旋转位置编码（Su et al.）
  * SwiGLU 前馈（Shazeer / LLaMA）
  * GQA 分组查询注意力（Ainslie et al.，n_kv_heads < n_heads 时生效）
  * 输入/输出嵌入权重绑定（Press & Wolf / GPT-2）

自研部分：numpy 上的前向与反向实现、参数命名与导出布局。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import engine as E
from .engine import Tensor


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    d_model: int = 160
    n_layers: int = 4
    n_heads: int = 5
    n_kv_heads: int = 5
    d_ff: int = 384
    max_seq: int = 192
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "max_seq": self.max_seq,
            "rope_theta": self.rope_theta,
            "tie_embeddings": self.tie_embeddings,
            "eps": self.eps,
        }


def _rot_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """RoPE：y = x*cos + rot(x)*sin。rot 为斜对称算子，故 rot^T = -rot。"""
    out_data = x.data * cos + _rot_half(x.data) * sin
    out = Tensor(out_data, requires_grad=x.requires_grad, _prev=(x,))

    def _bw() -> None:
        g = out.grad
        x.grad = (x.grad + g * cos - _rot_half(g) * sin).astype(np.float32)

    out._backward = _bw
    return out


def _repeat_kv(t: Tensor, rep: int) -> Tensor:
    """GQA：把 (B, Hkv, T, hd) 沿 head 维复制 rep 次。反向为分组求和。"""
    out = Tensor(np.repeat(t.data, rep, axis=1), requires_grad=t.requires_grad, _prev=(t,))

    def _bw() -> None:
        if t.requires_grad:
            B, H, T, hd = out.grad.shape
            g = out.grad.reshape(B, H // rep, rep, T, hd).sum(axis=2)
            t.grad = (t.grad + g).astype(np.float32)

    out._backward = _bw
    return out


class Transformer:
    def __init__(self, cfg: ModelConfig, seed: int = 20260923, params: Optional[Dict[str, Tensor]] = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.params: Dict[str, Tensor] = params if params is not None else self._init_params()
        self._rope_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._mask_cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ 参数
    def _xavier(self, shape: Tuple[int, ...]) -> np.ndarray:
        fan_in, fan_out = shape[0], int(np.prod(shape[1:]))
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return (self.rng.normal(0.0, std, shape)).astype(np.float32)

    def _init_params(self) -> Dict[str, Tensor]:
        cfg = self.cfg
        D, V, hd = cfg.d_model, cfg.vocab_size, cfg.head_dim
        q_dim = cfg.n_heads * hd
        kv_dim = cfg.n_kv_heads * hd
        p: Dict[str, Tensor] = {}
        p["tok_emb"] = Tensor(self.rng.normal(0.0, 0.02, (V, D)).astype(np.float32), True, name="tok_emb")
        for i in range(cfg.n_layers):
            f = f"layers.{i}."
            p[f + "attn_norm"] = Tensor(np.ones(D, dtype=np.float32), True, name=f + "attn_norm")
            p[f + "ffn_norm"] = Tensor(np.ones(D, dtype=np.float32), True, name=f + "ffn_norm")
            p[f + "wq"] = Tensor(self._xavier((D, q_dim)), True, name=f + "wq")
            p[f + "wk"] = Tensor(self._xavier((D, kv_dim)), True, name=f + "wk")
            p[f + "wv"] = Tensor(self._xavier((D, kv_dim)), True, name=f + "wv")
            p[f + "wo"] = Tensor(self._xavier((q_dim, D)), True, name=f + "wo")
            p[f + "w1"] = Tensor(self._xavier((D, cfg.d_ff)), True, name=f + "w1")
            p[f + "w2"] = Tensor(self._xavier((cfg.d_ff, D)), True, name=f + "w2")
            p[f + "w3"] = Tensor(self._xavier((D, cfg.d_ff)), True, name=f + "w3")
        p["final_norm"] = Tensor(np.ones(D, dtype=np.float32), True, name="final_norm")
        if not cfg.tie_embeddings:
            p["lm_head"] = Tensor(self._xavier((D, V)), True, name="lm_head")
        return p

    def param_list(self) -> List[Tensor]:
        return [t for t in self.params.values() if t.requires_grad]

    def num_params(self) -> int:
        return sum(int(t.data.size) for t in self.params.values())

    # ------------------------------------------------------------------ 缓存
    def _rope(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        if t in self._rope_cache:
            return self._rope_cache[t]
        hd = self.cfg.head_dim
        inv = 1.0 / (self.cfg.rope_theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
        pos = np.arange(t, dtype=np.float64)
        ang = np.outer(pos, inv)  # (T, hd/2)
        ang = np.concatenate([ang, ang], axis=-1).astype(np.float32)  # (T, hd)
        cos, sin = np.cos(ang), np.sin(ang)
        self._rope_cache[t] = (cos[None, None, :, :], sin[None, None, :, :])
        return self._rope_cache[t]

    def _causal_mask(self, t: int) -> np.ndarray:
        if t not in self._mask_cache:
            m = np.triu(np.full((t, t), -1e9, dtype=np.float32), k=1)
            self._mask_cache[t] = m[None, None, :, :]
        return self._mask_cache[t]

    # ------------------------------------------------------------------ 前向
    def forward(
        self,
        ids: np.ndarray,
        targets: Optional[np.ndarray] = None,
        return_hidden: bool = False,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor]:
        """返回 (loss|None, logits, hidden)。ids: (B, T) int64"""
        cfg = self.cfg
        ids = np.asarray(ids, dtype=np.int64)
        B, T = ids.shape
        hd = cfg.head_dim
        H, Hkv = cfg.n_heads, cfg.n_kv_heads
        cos, sin = self._rope(T)
        mask = self._causal_mask(T)

        x = E.embedding(self.params["tok_emb"], ids)
        for i in range(cfg.n_layers):
            f = f"layers.{i}."
            h = E.rmsnorm(x, self.params[f + "attn_norm"], cfg.eps)
            q = E.linear(h, self.params[f + "wq"])
            k = E.linear(h, self.params[f + "wk"])
            v = E.linear(h, self.params[f + "wv"])
            q = E.permute(E.reshape(q, (B, T, H, hd)), (0, 2, 1, 3))
            k = E.permute(E.reshape(k, (B, T, Hkv, hd)), (0, 2, 1, 3))
            v = E.permute(E.reshape(v, (B, T, Hkv, hd)), (0, 2, 1, 3))
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if Hkv != H:
                k = _repeat_kv(k, H // Hkv)
                v = _repeat_kv(v, H // Hkv)
            scores = E.scale(E.matmul(q, E._t_last2_contig(k)), 1.0 / np.sqrt(hd))
            scores = E.add(scores, Tensor(mask))
            att = E.softmax(scores, axis=-1)
            o = E.matmul(att, v)
            o = E.reshape(E.permute(o, (0, 2, 1, 3)), (B, T, H * hd))
            x = E.add(x, E.linear(o, self.params[f + "wo"]))
            h2 = E.rmsnorm(x, self.params[f + "ffn_norm"], cfg.eps)
            gate = E.linear(h2, self.params[f + "w1"])
            up = E.linear(h2, self.params[f + "w3"])
            act = E.mul(E.silu(gate), up)
            x = E.add(x, E.linear(act, self.params[f + "w2"]))

        hidden = E.rmsnorm(x, self.params["final_norm"], cfg.eps)
        if cfg.tie_embeddings:
            # 展平成 2D 再做输出投影：3D×2D 的 numpy 路径会退化成逐 batch 小 GEMM
            hf = E.reshape(hidden, (-1, cfg.d_model))
            head_t = E._t_last2_contig(self.params["tok_emb"])
            logits = E.reshape(E.matmul(hf, head_t), (B, T, cfg.vocab_size))
        else:
            logits = E.linear(hidden, self.params["lm_head"])

        loss = None
        if targets is not None:
            loss = E.cross_entropy(logits, targets)
        return loss, logits, hidden

    def encode(self, ids: np.ndarray) -> Tensor:
        """只取最后隐藏状态，供嵌入/重排头使用。"""
        _, _, hidden = self.forward(ids)
        return hidden


__all__ = ["ModelConfig", "Transformer", "apply_rope"]
