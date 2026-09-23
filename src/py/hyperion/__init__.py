"""Hyperion AI - Python 训练内核

作者：晨星

模块：
    engine      极简自动微分引擎（numpy / float32）
    optim       AdamW + 余弦调度 + 梯度裁剪
    arch        Transformer（RMSNorm + RoPE + SwiGLU + GQA + 权重绑定）
    tokenizer   字节级 BPE 分词器
    corpus      确定性合成语料与查询真值
    export      HMF v1 模型导出（int8 / fp32）
    train       三阶段训练主入口
"""

__version__ = "0.1.0"
__author__ = "晨星"
