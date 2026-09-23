"""Hyperion AI - 模型加载：Hyperion Weight Model (.hwm v1) 的 Python 端读取

作者：晨星

存在的理由（不是「顺手加个读函数」，而是这座项目的一条硬纪律）：

双引擎架构里，Python 侧负责训练、TypeScript 侧负责推理，两者之间的契约
就是一个 .hwm 文件。契约一旦漂移，症状往往是「生成乱码」这种几乎无法诊断
的故障，而不是一个明确的报错。

有了加载器，就能做交叉验证：同一份权重分别在 Python（autograd 前向）与
TypeScript（手写推理内核）跑一遍，逐位比较 logits。
两者不一致时 bug 在推理侧；两者一致、但生成仍然不对，则 bug 在训练侧。
这个判据能把「模型坏了」这种模糊描述压缩成一个可机械定位的结论。

布局细节与导出侧严格互为逆操作，见 export.py 文件头与 KEEP_LAYOUT。
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Dict, Tuple

import numpy as np

from .export import MAGIC, VERSION, HEADER_SIZE, KEEP_LAYOUT, dequantize_int8

__all__ = ["load_hwm"]


def load_hwm(path: str, dequantize: bool = True, verify_crc: bool = True) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """读取 .hwm 文件，返回 (meta, tensors)。

    - meta：文件元数据（含 arch / dtype / tensors 表）
    - tensors：名字 -> float32 ndarray。int8 张量会被反量化回 fp32。
      二维权重已按 KEEP_LAYOUT 规则还原转置，因此返回的布局与训练侧 params
      一致（线性层为 (in, out)），可以直接灌给 Transformer。
    """
    with open(path, "rb") as f:
        blob = f.read()

    if blob[:4] != MAGIC:
        raise ValueError(f"不是合法的 .hwm 文件（magic={blob[:4]!r}）")
    version, header_size, meta_len = struct.unpack("<HHI", blob[4:12])
    crc_expected = struct.unpack("<I", blob[12:16])[0]
    data_offset, file_size = struct.unpack("<QQ", blob[16:32])
    if version != VERSION:
        raise ValueError(f"不支持的格式版本：{version}（本加载器支持 {VERSION}）")
    if len(blob) != file_size:
        raise ValueError(f"文件大小不符：头部声明 {file_size}，实际 {len(blob)}")

    meta_bytes = blob[32 : 32 + meta_len]
    meta = json.loads(meta_bytes.decode("utf-8"))

    if verify_crc:
        # CRC 覆盖 metadata + 填充 + data 区（与导出侧 crc_input 的构造完全一致）
        pad = data_offset - HEADER_SIZE - meta_len
        crc_actual = zlib.crc32(meta_bytes + b"\x00" * pad + blob[data_offset:file_size]) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise ValueError(f"CRC 校验失败：期望 {crc_expected:08x}，实际 {crc_actual:08x}")

    # 1) 原始读取（保持文件里的布局）
    raw: Dict[str, Tuple[np.ndarray, str]] = {}
    for e in meta["tensors"]:
        dt = np.float32 if e["dtype"] == "f32" else np.int8
        cnt = int(np.prod(e["shape"]))
        arr = np.frombuffer(blob, dtype=dt, count=cnt, offset=data_offset + e["offset"])
        raw[e["name"]] = (arr.reshape(e["shape"]), e["dtype"])

    # 2) 反量化 + 还原导出时做过的转置
    tensors: Dict[str, np.ndarray] = {}
    for name, (arr, dt) in raw.items():
        if name.endswith(".scale"):
            continue
        if dt == "i8":
            if not dequantize:
                tensors[name] = np.ascontiguousarray(arr)
                continue
            if name + ".scale" not in raw:
                raise ValueError(f"int8 张量 {name} 缺少伴生 scale")
            arr = dequantize_int8(arr, raw[name + ".scale"][0])
        if arr.ndim == 2 and name not in KEEP_LAYOUT:
            arr = arr.T
        tensors[name] = np.ascontiguousarray(arr, dtype=np.float32)

    return meta, tensors
