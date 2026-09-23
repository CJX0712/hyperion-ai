/**
 * M01 hwm - Hyperion Model Format (.hwm v1) 加载器（零第三方依赖）
 *
 * 作者：晨星
 *
 * 与 Python 侧 src/py/hyperion/export.py 是同一份契约的两端，必须逐字段对齐：
 *
 *   [0,  32)   固定二进制头
 *                magic        b"HWM1"   4B  偏移 0
 *                version      uint16  = 1   偏移 4
 *                header_size  uint16  = 32  偏移 6
 *                meta_len     uint32        偏移 8
 *                crc32        uint32        偏移 12（覆盖 metadata + data 区）
 *                data_offset  uint64        偏移 16
 *                file_size    uint64        偏移 24
 *   [32, 32+meta_len)  metadata JSON（UTF-8）
 *   [pad 到 64 字节)
 *   [data_offset, ...)  张量数据区，每个张量 64 字节对齐
 *
 * 关键实现点：
 *   * 张量数据用 buffer 上的 TypedArray **视图**暴露，不 memcpy。
 *     64 字节对齐 + 偏移为 4 的倍数，保证 Float32Array 视图合法。
 *   * uint64 用 getBigUint64 读取后转 Number（模型文件远小于 2^53，安全）。
 *   * CRC32 默认校验：损坏的权重在加载期直接报错，而不是表现为"生成乱码"
 *     这种无法诊断的故障。
 */

import { readFileSync } from "node:fs";

export interface TensorEntry {
  name: string;
  dtype: "f32" | "i8";
  shape: number[];
  offset: number;
  nbytes: number;
}

export interface HwmMeta {
  format: string;
  version: number;
  arch: Record<string, number | boolean>;
  dtype: string;
  tokenizer: string;
  align: number;
  n_tensors: number;
  tensors: TensorEntry[];
  meta?: Record<string, unknown>;
}

/** 一个已加载的权重视图。i8 张量附带逐行 scale（伴生张量 <name>.scale）。 */
export interface WeightView {
  name: string;
  shape: number[];
  f32: Float32Array | null;
  i8: Int8Array | null;
  scale: Float32Array | null;
}

export class HwmModel {
  readonly meta: HwmMeta;
  readonly weights = new Map<string, WeightView>();
  private readonly buf: ArrayBuffer;

  private constructor(buf: ArrayBuffer, meta: HwmMeta, dataOffset: number) {
    this.buf = buf;
    this.meta = meta;
    for (const t of meta.tensors) {
      const off = dataOffset + t.offset;
      let f32: Float32Array | null = null;
      let i8: Int8Array | null = null;
      if (t.dtype === "f32") {
        if (off % 4 !== 0) throw new Error(`f32 张量 ${t.name} 偏移未按 4 字节对齐：${off}`);
        f32 = new Float32Array(buf, off, t.nbytes / 4);
      } else {
        i8 = new Int8Array(buf, off, t.nbytes);
      }
      this.weights.set(t.name, { name: t.name, shape: t.shape, f32, i8, scale: null });
    }
    for (const [name, w] of this.weights) {
      if (name.endsWith(".scale")) {
        const owner = this.weights.get(name.slice(0, -".scale".length));
        if (owner && w.f32) owner.scale = w.f32;
      }
    }
  }

  /** 从文件加载。verifyCrc 默认开启。 */
  static load(path: string, opts: { verifyCrc?: boolean } = {}): HwmModel {
    const file = readFileSync(path);
    const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer;
    return HwmModel.fromBuffer(buf, opts);
  }

  static fromBuffer(buf: ArrayBuffer, opts: { verifyCrc?: boolean } = {}): HwmModel {
    const dv = new DataView(buf);
    const m = new Uint8Array(buf, 0, 4);
    if (!(m[0] === 0x48 && m[1] === 0x57 && m[2] === 0x4d && m[3] === 0x31)) {
      throw new Error(`不是 HWM1 文件：magic=${[...m].join(",")}`);
    }
    const version = dv.getUint16(4, true);
    if (version !== 1) throw new Error(`不支持的 HWM 版本：${version}`);
    const headerSize = dv.getUint16(6, true);
    const metaLen = dv.getUint32(8, true);
    const crc = dv.getUint32(12, true);
    const dataOffset = Number(dv.getBigUint64(16, true));
    const fileSize = Number(dv.getBigUint64(24, true));
    if (fileSize !== buf.byteLength) {
      throw new Error(`文件大小不符：头声明 ${fileSize}，实际 ${buf.byteLength}`);
    }
    const metaJson = new TextDecoder().decode(new Uint8Array(buf, headerSize, metaLen));
    const meta = JSON.parse(metaJson) as HwmMeta;

    if (opts.verifyCrc !== false) {
      const actual = crc32(new Uint8Array(buf, headerSize, buf.byteLength - headerSize));
      if (actual !== crc) {
        throw new Error(`CRC32 校验失败：期望 ${crc.toString(16)}，实际 ${actual.toString(16)}`);
      }
    }
    return new HwmModel(buf, meta, dataOffset);
  }

  /** 取权重，缺失即抛错（静默返回 undefined 会让模型退化成随机输出，必须早失败）。 */
  get(name: string): WeightView {
    const w = this.weights.get(name);
    if (!w) throw new Error(`权重缺失：${name}`);
    return w;
  }

  /** 取反量化后的 fp32 副本，用于与 int8 路径做精度对比（量化不变量校验）。 */
  dequantized(name: string): Float32Array {
    const w = this.get(name);
    if (w.f32) return w.f32;
    if (!w.i8 || !w.scale) throw new Error(`权重既不是 f32 也没有 scale：${name}`);
    const rows = w.shape[0];
    const cols = w.shape.length > 1 ? w.shape.slice(1).reduce((a, b) => a * b, 1) : 1;
    const out = new Float32Array(w.i8.length);
    for (let r = 0; r < rows; r++) {
      const s = w.scale.length === rows ? w.scale[r] : w.scale[0];
      for (let c = 0; c < cols; c++) out[r * cols + c] = w.i8[r * cols + c] * s;
    }
    return out;
  }
}

const CRC_TABLE: Uint32Array = ((): Uint32Array => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();

/** CRC32（IEEE 802.3 多项式），与 Python zlib.crc32 一致。 */
export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export const __all__ = ["HwmModel", "crc32"];
