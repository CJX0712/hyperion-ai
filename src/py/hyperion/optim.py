"""Hyperion AI - 优化器与学习率调度（numpy 实现，无第三方依赖）

作者：晨星
"""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np

from .engine import Tensor


class AdamW:
    """解耦权重衰减的 Adam。betas 默认 (0.9, 0.95)，与主流 LLM 训练一致。"""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
    ) -> None:
        self.params: List[Tensor] = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def _clip(self) -> float:
        if self.grad_clip <= 0:
            return 0.0
        total = 0.0
        for p in self.params:
            total += float(np.sum(p.grad.astype(np.float64) ** 2))
        norm = math.sqrt(total)
        if norm > self.grad_clip:
            s = np.float32(self.grad_clip / (norm + 1e-12))
            for p in self.params:
                p.grad = (p.grad * s).astype(np.float32)
        return norm

    def step(self, lr_scale: float = 1.0) -> float:
        """返回裁剪前的全局梯度范数（可观测）。"""
        gnorm = self._clip()
        self.t += 1
        lr = np.float32(self.lr * lr_scale)
        b1c = np.float32(self.b1 ** self.t)
        b2c = np.float32(self.b2 ** self.t)
        for i, p in enumerate(self.params):
            g = p.grad.astype(np.float32)
            self.m[i] = (self.b1 * self.m[i] + (1.0 - self.b1) * g).astype(np.float32)
            self.v[i] = (self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).astype(np.float32)
            mhat = self.m[i] / (1.0 - b1c)
            vhat = self.v[i] / (1.0 - b2c)
            upd = mhat / (np.sqrt(vhat) + np.float32(self.eps))
            p.data = (p.data - lr * upd).astype(np.float32)
            if self.weight_decay != 0.0:
                p.data = (p.data - lr * np.float32(self.weight_decay) * p.data).astype(np.float32)
        return gnorm


Tuple_betas = tuple  # 兼容旧版类型注解写法


def cosine_schedule(
    step: int, total_steps: int, warmup: int, base_lr: float = 1.0, min_lr_ratio: float = 0.05
) -> float:
    """warmup 线性上升 -> 余弦衰减到 base_lr * min_lr_ratio。返回 lr_scale（相对 base_lr）。"""
    total_steps = max(int(total_steps), 1)
    warmup = max(int(warmup), 1)
    if step < warmup:
        return float(step + 1) / float(warmup)
    prog = (step - warmup) / float(max(total_steps - warmup, 1))
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cos)


__all__ = ["AdamW", "cosine_schedule"]
