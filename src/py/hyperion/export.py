"""Hyperion AI - 模型导出：Hyperion Weight Model (.hwm v1)

作者：晨星

文件布局（小端，所有偏移自文件起始）：
    [0,  32)   固定 32 字节二进制头
                  magic      b"HWM1"        4B  偏移 0
                  version    uint16 = 1     2B  偏移 4
                  header_size uint16 = 32   2B  偏移 6
                  meta_len   uint32         4B  偏移 8
                  crc32      uint32         4B  偏移 12（覆盖 metadata + data 区）
                  data_offset uint64        8B  偏移 16
                  file_size  uint64         8B  偏移 24
    [32, 32+meta_len)       metadata JSON（UTF-8）：架构配置 + 张量表
    [pad 到 64 字节对齐)    零填充
    [data_offset, file_size) 张量数据区，每个张量起始 64 字节对齐

设计理由：
    * 固定头 + JSON 元数据：TS 侧 DataView 解析一次即可，不需要第三方库，
      也不需要在两侧维护两份易漂移的二进制解析代码。
    * 64 字节对齐 + 偏移 4 的倍数：TS 侧可以直接用 Float32Array/Int8Array
      建零拷贝视图，无需 memcpy。
    * 逐通道 int8 量化（权重矩阵按输出通道求 scale）：比逐张量量化的困惑度
      劣化显著更小，代价只是多一个 fp32 scale 张量。
    * CRC32：交付物在仓库里被 Git LFS / 传输损坏时能被机械发现，而不是
      表现为"生成乱码"这种不可诊断的故障。
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Dict, List, Tuple

import numpy as np

MAGIC = b"HWM1"
VERSION = 1
HEADER_SIZE = 32
TENSOR_ALIGN = 64
PAYLOAD_ALIGN = 4

# int8 导出时仍须保留为 f32 的权重：TS 推理侧直接以 .f32 读取这些权重
# （RMSNorm 的归一化权重、双塔/交叉编码器投影头），量化反而会丢失精度或取 null。
# 它们体量极小（每个 ~d_model 元素），保留 f32 对整体体积与速度影响可忽略，
# 而真正的量化收益来自 wq/wk/wv/wo/w1/w2/w3 等大矩阵乘。
F32_KEEP = frozenset({"emb_proj", "ce_score"})


def quantize_int8_perchannel(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """逐通道对称 int8 量化。

    2D 及以上：沿 axis=0（输出通道）求 scale；其余按整张量求 scale。
    返回 (int8 数据, float32 scale)。
    """
    arr = np.ascontiguousarray(w, dtype=np.float32)
    if arr.ndim >= 2:
        flat = arr.reshape(arr.shape[0], -1)
        amax = np.max(np.abs(flat), axis=1)
    else:
        amax = np.array([np.max(np.abs(arr))], dtype=np.float32)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(arr / scale.reshape([-1] + [1] * (arr.ndim - 1))), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    shape = q.shape
    s = np.asarray(scale, dtype=np.float32).reshape([-1] + [1] * (len(shape) - 1)) if len(shape) >= 2 else np.asarray(scale, dtype=np.float32)
    return q.astype(np.float32) * s


def _pad(n: int, align: int) -> int:
    return (-n) % align


#: 保持原始 (vocab, d_model) 布局的张量。它们是查表而不是矩阵乘。
KEEP_LAYOUT = frozenset({"tok_emb"})


def export_hwm(
    path: str,
    arch: Dict,
    tensors: Dict[str, np.ndarray],
    dtype: str = "i8",
    tokenizer_ref: str = "tokenizer.json",
    extra_meta: Dict | None = None,
) -> Dict:
    """导出 .hwm 文件。返回导出统计。

    布局契约：除 KEEP_LAYOUT 外，所有二维权重在导出时转置为 (out, in) 行主序。
    这样 TS 侧 tensor.matmulT1 能以内层沿 K 连续的方式访问，同时逐行 int8 scale
    正好落在 out 维上，可以从 m 循环外提出来。改动此契约必须同步 arch/index.ts。
    """
    if dtype not in ("f32", "i8"):
        raise ValueError("dtype 只能是 f32 或 i8")

    # 1) 先做布局变换，再量化，得到最终张量集合（含 scale 伴生张量）
    final: List[Tuple[str, str, List[int], bytes]] = []
    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name])
        if arr.ndim == 2 and name not in KEEP_LAYOUT:
            arr = np.ascontiguousarray(arr.T)
        if dtype == "i8" and not (name.endswith("_norm") or name in F32_KEEP):
            q, scale = quantize_int8_perchannel(arr)
            final.append((name + ".scale", "f32", list(scale.shape), scale.astype(np.float32).tobytes()))
            final.append((name, "i8", list(arr.shape), q.tobytes()))
        else:
            final.append((name, "f32", list(arr.shape), arr.astype(np.float32).tobytes()))

    # 2) 布局
    entries: List[Dict] = []
    body = bytearray()
    for name, dt, shape, blob in final:
        pad = _pad(len(body), TENSOR_ALIGN)
        if pad:
            body.extend(b"\x00" * pad)
        entries.append({"name": name, "dtype": dt, "shape": shape, "offset": len(body), "nbytes": len(blob)})
        body.extend(blob)
        tail = _pad(len(body), PAYLOAD_ALIGN)
        if tail:
            body.extend(b"\x00" * tail)

    meta: Dict = {
        "format": "hyperion-weight-model",
        "version": VERSION,
        "arch": arch,
        "dtype": dtype,
        "tokenizer": tokenizer_ref,
        "align": TENSOR_ALIGN,
        "n_tensors": len(entries),
        "tensors": entries,
    }
    if extra_meta:
        meta["meta"] = extra_meta

    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    data_offset = ((HEADER_SIZE + len(meta_bytes)) + TENSOR_ALIGN - 1) // TENSOR_ALIGN * TENSOR_ALIGN
    file_size = data_offset + len(body)

    head_wo_crc = MAGIC + struct.pack("<HHI", VERSION, HEADER_SIZE, len(meta_bytes))
    crc_input = meta_bytes + b"\x00" * (data_offset - HEADER_SIZE - len(meta_bytes)) + bytes(body)
    crc = zlib.crc32(crc_input) & 0xFFFFFFFF
    header = head_wo_crc + struct.pack("<I", crc) + struct.pack("<QQ", data_offset, file_size)

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta_bytes)
        f.write(b"\x00" * (data_offset - HEADER_SIZE - len(meta_bytes)))
        f.write(bytes(body))

    payload = sum(e["nbytes"] for e in entries)
    return {
        "path": path,
        "dtype": dtype,
        "n_tensors": len(entries),
        "payload_bytes": payload,
        "file_bytes": file_size,
        "crc32": f"{crc:08x}",
        "params": int(sum(int(np.prod(e["shape"])) for e in entries if not e["name"].endswith(".scale"))),
    }


__all__ = ["export_hwm", "quantize_int8_perchannel", "dequantize_int8", "MAGIC", "VERSION"]
