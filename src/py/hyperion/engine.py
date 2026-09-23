"""Hyperion AI - Python 训练内核：极简自动微分引擎（numpy / float32）

作者：晨星

设计约束：
  * 只依赖 numpy，不依赖 torch / 任何需要本地编译的包。
  * 反向传播基于反向拓扑序的闭包链（micrograd 风格），张量语义为 numpy 广播语义。
  * 每个算子都可通过中心差分做梯度检验（tests/test_engine.py 的硬性不变量：maxRelErr < 1e-4，float32 下典型值 ~1e-6）。
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


def _reduce_grad(g: Array, shape: Tuple[int, ...]) -> Array:
    """把梯度还原到参数形状：处理广播新增轴与被广播的轴。"""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    if g.shape != shape:
        g = g.reshape(shape)
    return np.ascontiguousarray(g, dtype=np.float32)


class Tensor:
    """带反向闭包的张量。data 恒为 float32。"""

    __slots__ = ("data", "grad", "_prev", "_backward", "requires_grad", "name")

    def __init__(
        self,
        data: Array,
        requires_grad: bool = False,
        _prev: Tuple["Tensor", ...] = (),
        _backward: Optional[Callable[[], None]] = None,
        name: str = "",
    ) -> None:
        # 注意：numpy>=2 的 ascontiguousarray 会把 0 维数组提升为 (1,)，
        # 会破坏标量 loss 的语义，因此 0 维直接保留。
        arr = np.asarray(data, dtype=np.float32)
        self.data = arr if arr.ndim == 0 else np.ascontiguousarray(arr)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._prev = _prev
        self._backward = _backward if _backward is not None else (lambda: None)
        self.name = name

    # ---- 基础属性 -------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    def zero_grad(self) -> None:
        self.grad[...] = 0.0

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def numpy(self) -> Array:
        return self.data

    def astype(self, dtype) -> Array:
        return self.data.astype(dtype)

    # ---- 反向传播 -------------------------------------------------------
    def backward(self, grad: Optional[Array] = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("非标量反向必须显式传入 grad")
            grad = np.ones_like(self.data)
        self.grad = (self.grad + np.asarray(grad, dtype=np.float32)).astype(np.float32)

        topo: List[Tensor] = []
        seen = set()

        def build(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        for t in reversed(topo):
            t._backward()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Tensor(shape={self.shape}, name={self.name!r})"


def _maybe_grad(a: Tensor, b: Tensor) -> bool:
    return a.requires_grad or b.requires_grad


def _wrap(x: Tensor | Array | float, requires_grad: bool = False) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float32), requires_grad=requires_grad)


# ---------------------------------------------------------------- 元算子
def add(a: Tensor, b: Tensor) -> Tensor:
    a, b = _wrap(a), _wrap(b)
    out = Tensor(a.data + b.data, requires_grad=_maybe_grad(a, b), _prev=(a, b))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + _reduce_grad(out.grad, a.shape)).astype(np.float32)
        if b.requires_grad:
            b.grad = (b.grad + _reduce_grad(out.grad, b.shape)).astype(np.float32)

    out._backward = _bw
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    a, b = _wrap(a), _wrap(b)
    out = Tensor(a.data - b.data, requires_grad=_maybe_grad(a, b), _prev=(a, b))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + _reduce_grad(out.grad, a.shape)).astype(np.float32)
        if b.requires_grad:
            b.grad = (b.grad - _reduce_grad(out.grad, b.shape)).astype(np.float32)

    out._backward = _bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    a, b = _wrap(a), _wrap(b)
    out = Tensor(a.data * b.data, requires_grad=_maybe_grad(a, b), _prev=(a, b))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + _reduce_grad(out.grad * b.data, a.shape)).astype(np.float32)
        if b.requires_grad:
            b.grad = (b.grad + _reduce_grad(out.grad * a.data, b.shape)).astype(np.float32)

    out._backward = _bw
    return out


def scale(a: Tensor, s: float) -> Tensor:
    a = _wrap(a)
    out = Tensor(a.data * np.float32(s), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + out.grad * np.float32(s)).astype(np.float32)

    out._backward = _bw
    return out


def _t_last2(x: Array) -> Array:
    return np.swapaxes(x, -1, -2)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """支持 (..., M, K) @ (..., K, N) 与 (..., M, K) @ (K, N) 的批量矩阵乘。"""
    a, b = _wrap(a), _wrap(b)
    out = Tensor(np.matmul(a.data, b.data), requires_grad=_maybe_grad(a, b), _prev=(a, b))

    # 反向要用到的转置操作数在正向就物化成连续数组：
    # numpy 对非连续操作数的 matmul 走慢路径（实测慢 1.9 倍），
    # 而反向是热路径里被调用最频繁的地方。
    need_a = a.requires_grad
    need_b = b.requires_grad
    bt = np.ascontiguousarray(_t_last2(b.data)) if need_a else None
    at = np.ascontiguousarray(_t_last2(a.data)) if need_b else None

    def _bw() -> None:
        g = out.grad
        if need_a:
            ga = np.matmul(g, bt)
            ag = a.grad
            if ga.shape == ag.shape:
                ag += ga
            else:
                a.grad = (ag + _reduce_grad(ga, ag.shape)).astype(np.float32)
        if need_b:
            gb = np.matmul(at, g)
            bg = b.grad
            if gb.shape == bg.shape:
                bg += gb
            else:
                b.grad = (bg + _reduce_grad(gb, bg.shape)).astype(np.float32)

    out._backward = _bw
    return out


def _t_last2_contig(a: Tensor) -> Tensor:
    """转置最后两维并强制连续。

    numpy 的 matmul 遇到非连续操作数会走慢路径（实测慢 1.9 倍），
    注意力里 kᵀ 是热路径，必须显式物化。
    """
    out = Tensor(np.ascontiguousarray(_t_last2(a.data)), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + _t_last2(out.grad)).astype(np.float32)

    out._backward = _bw
    return out


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    """x @ w。x 为 3D 及以上时先展平成 2D 再乘。

    实测（本环境，OpenBLAS 0.3.34）：
        (16,96,128) @ (128,128)  15.0 ms   3.35 GFLOP/s
        (1536,128)  @ (128,128)   1.8 ms  27.53 GFLOP/s
    numpy 的 3D×2D 路径会退化成逐 batch 小 GEMM，慢 8 倍以上。展平是唯一正确写法。
    """
    if x.data.ndim > 2:
        orig = x.shape
        xf = reshape(x, (-1, orig[-1]))
        out = matmul(xf, w)
        out = reshape(out, (*orig[:-1], w.shape[1]))
    else:
        out = matmul(x, w)
    if b is not None:
        out = add(out, b)
    return out


def transpose_last2(a: Tensor) -> Tensor:
    a = _wrap(a)
    out = Tensor(_t_last2(a.data), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + _t_last2(out.grad)).astype(np.float32)

    out._backward = _bw
    return out


def reshape(a: Tensor, shape: Sequence[int]) -> Tensor:
    a = _wrap(a)
    out = Tensor(a.data.reshape(tuple(shape)), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + out.grad.reshape(a.shape)).astype(np.float32)

    out._backward = _bw
    return out


def sum_to(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    a = _wrap(a)
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis)
            a.grad = (a.grad + np.broadcast_to(g, a.shape).astype(np.float32) * np.float32(1.0)).astype(np.float32)

    out._backward = _bw
    return out


def mean(a: Tensor) -> Tensor:
    a = _wrap(a)
    n = np.float32(a.data.size)
    out = Tensor(a.data.mean(), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + np.broadcast_to(out.grad / n, a.shape)).astype(np.float32)

    out._backward = _bw
    return out


def exp(a: Tensor) -> Tensor:
    a = _wrap(a)
    out = Tensor(np.exp(a.data), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + out.grad * out.data).astype(np.float32)

    out._backward = _bw
    return out


def log(a: Tensor, eps: float = 1e-8) -> Tensor:
    a = _wrap(a)
    out = Tensor(np.log(a.data + eps), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + out.grad / (a.data + eps)).astype(np.float32)

    out._backward = _bw
    return out


def relu(a: Tensor) -> Tensor:
    a = _wrap(a)
    out = Tensor(np.maximum(a.data, 0), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + out.grad * (a.data > 0)).astype(np.float32)

    out._backward = _bw
    return out


def silu(a: Tensor) -> Tensor:
    """SiLU(x) = x * sigmoid(x)"""
    a = _wrap(a)
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = Tensor(a.data * s, requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            ds = s * (1.0 - s)
            a.grad = (a.grad + out.grad * (s + a.data * ds)).astype(np.float32)

    out._backward = _bw
    return out


def gelu(a: Tensor) -> Tensor:
    """tanh 近似 GELU。"""
    a = _wrap(a)
    k = np.float32(0.7978845608)
    inner = k * (a.data + np.float32(0.044715) * a.data ** 3)
    t = np.tanh(inner)
    out = Tensor(0.5 * a.data * (1.0 + t), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            d_inner = k * (1.0 + 3.0 * np.float32(0.044715) * a.data ** 2)
            dt = (1.0 - t * t) * d_inner
            a.grad = (a.grad + out.grad * (0.5 * (1.0 + t) + 0.5 * a.data * dt)).astype(np.float32)

    out._backward = _bw
    return out


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    a = _wrap(a)
    m = a.data.max(axis=axis, keepdims=True)
    e = np.exp(a.data - m)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(s, requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            g = out.grad
            dot = np.sum(g * out.data, axis=axis, keepdims=True)
            a.grad = (a.grad + out.data * (g - dot)).astype(np.float32)

    out._backward = _bw
    return out


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    a = _wrap(a)
    m = a.data.max(axis=axis, keepdims=True)
    z = a.data - m - np.log(np.exp(a.data - m).sum(axis=axis, keepdims=True))
    out = Tensor(z, requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            sm = np.exp(out.data)
            gsum = out.grad.sum(axis=axis, keepdims=True)
            a.grad = (a.grad + out.grad - sm * gsum).astype(np.float32)

    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: Array, ignore_index: int = -100) -> Tensor:
    """logits: (N, V) 或 (B, T, V)；targets 同前缀形状的整数数组。返回标量平均 NLL。"""
    logits = _wrap(logits)
    tgt = np.asarray(targets, dtype=np.int64)
    V = logits.shape[-1]
    flat_logits = logits.data.reshape(-1, V).astype(np.float64)
    flat_tgt = tgt.reshape(-1)
    mask = flat_tgt != ignore_index
    idx = np.where(mask, flat_tgt, 0)
    m = flat_logits.max(axis=1, keepdims=True)
    z = flat_logits - m - np.log(np.exp(flat_logits - m).sum(axis=1, keepdims=True))
    nll = -z[np.arange(flat_logits.shape[0]), idx]
    denom = max(int(mask.sum()), 1)
    loss = float((nll * mask).sum() / denom)
    out = Tensor(np.float32(loss), requires_grad=logits.requires_grad, _prev=(logits,))

    def _bw() -> None:
        if logits.requires_grad:
            p = np.exp(z)
            g = np.zeros_like(p)
            g[np.arange(p.shape[0]), idx] = -1.0
            g = (g + p) * mask[:, None] / np.float32(denom)
            logits.grad = (logits.grad + g.reshape(logits.shape).astype(np.float32)).astype(np.float32)

    out._backward = _bw
    return out


def embedding(w: Tensor, idx: Array) -> Tensor:
    """w: (V, D)；idx: 任意形状的整数数组 -> (..., D)"""
    w = _wrap(w)
    ii = np.asarray(idx, dtype=np.int64)
    out = Tensor(w.data[ii], requires_grad=w.requires_grad, _prev=(w,))

    def _bw() -> None:
        if w.requires_grad:
            g = out.grad.reshape(-1, w.shape[1]).astype(np.float32)
            np.add.at(w.grad, ii.reshape(-1), g)

    out._backward = _bw
    return out


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm：x / sqrt(mean(x^2)) * weight（LLaMA 风格）。"""
    x, weight = _wrap(x), _wrap(weight)
    ms = (x.data.astype(np.float32) ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    normed = x.data * inv
    out = Tensor(normed * weight.data, requires_grad=(x.requires_grad or weight.requires_grad), _prev=(x, weight))

    def _bw() -> None:
        g = out.grad
        gx = g * weight.data  # d/d normed
        if weight.requires_grad:
            weight.grad = (weight.grad + _reduce_grad(g * normed, weight.shape)).astype(np.float32)
        if x.requires_grad:
            d = x.shape[-1]
            gy = gx * inv - normed * (gx * normed).sum(axis=-1, keepdims=True) / np.float32(d) * inv
            x.grad = (x.grad + gy).astype(np.float32)

    out._backward = _bw
    return out


def concat(parts: Sequence[Tensor], axis: int = -1) -> Tensor:
    parts = [_wrap(p) for p in parts]
    sizes = [p.shape[axis] for p in parts]
    out = Tensor(
        np.concatenate([p.data for p in parts], axis=axis),
        requires_grad=any(p.requires_grad for p in parts),
        _prev=tuple(parts),
    )

    def _bw() -> None:
        splits = np.cumsum([0] + list(sizes))
        for i, p in enumerate(parts):
            if p.requires_grad:
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = slice(splits[i], splits[i + 1])
                p.grad = (p.grad + out.grad[tuple(sl)]).astype(np.float32)

    out._backward = _bw
    return out


def permute(a: Tensor, axes: Sequence[int]) -> Tensor:
    """通用轴置换。反向即逆置换。"""
    a = _wrap(a)
    ax = tuple(int(i) for i in axes)
    inv = [0] * len(ax)
    for i, p in enumerate(ax):
        inv[p] = i
    out = Tensor(np.transpose(a.data, ax), requires_grad=a.requires_grad, _prev=(a,))

    def _bw() -> None:
        if a.requires_grad:
            a.grad = (a.grad + np.transpose(out.grad, tuple(inv))).astype(np.float32)

    out._backward = _bw
    return out


def stack(parts: Sequence[Tensor], axis: int = 0) -> Tensor:
    parts = [_wrap(p) for p in parts]
    out = Tensor(
        np.stack([p.data for p in parts], axis=axis),
        requires_grad=any(p.requires_grad for p in parts),
        _prev=tuple(parts),
    )

    def _bw() -> None:
        for i, p in enumerate(parts):
            if p.requires_grad:
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = i
                p.grad = (p.grad + out.grad[tuple(sl)]).astype(np.float32)

    out._backward = _bw
    return out


def scalar(x: float) -> Tensor:
    return Tensor(np.float32(x))


__all__ = [
    "Tensor", "add", "sub", "mul", "scale", "matmul", "linear", "transpose_last2",
    "reshape", "sum_to", "mean", "exp", "log", "relu", "silu", "gelu", "softmax",
    "log_softmax", "cross_entropy", "embedding", "rmsnorm", "concat", "stack", "permute",
    "scalar",
]
