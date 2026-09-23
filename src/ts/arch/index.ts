/**
 * M03 arch - Transformer 推理图（RMSNorm + RoPE + SwiGLU + GQA + KV Cache）
 *
 * 作者：晨星
 *
 * 复用的业界成果（架构层，非自研）：
 *   RMSNorm（LLaMA 预归一化）、RoPE（Su et al.）、SwiGLU（Shazeer / LLaMA）、
 *   GQA 分组查询注意力（Ainslie et al.）、输入/输出嵌入权重绑定（Press & Wolf）。
 * 自研部分：numpy 训练侧之外的 TS 推理实现、KV Cache 布局、int8 权重推理路径。
 *
 * 权重布局约定（与 export.py 强耦合，改动必须同步）：
 *   * tok_emb 保持 (vocab, dModel) —— 它是查表，不是矩阵乘。
 *   * 其余二维权重在导出时已转置为 (out, in)，配合 tensor.matmulT1 的
 *     m-outer / k-inner 循环，使内层沿 K 连续访问；int8 的逐行 scale 正好
 *     落在 out 维上，可以提到 m 循环外。
 *
 * KV Cache 布局：[layer][pos * kvDim + head * headDim + d]，预分配 maxSeq 长度，
 * 避免每步重新分配。多轮对话复用同一实例前必须调用 reset()。
 */

import {
  matmulT1, matmulT1Int8, rmsnormInplace, softmaxInplace, swigluInplace, ropeInPlace,
} from "../tensor/index.ts";
import type { HwmModel, WeightView } from "../tensor/hwm.ts";

export interface ArchConfig {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  max_seq: number;
  rope_theta: number;
  tie_embeddings: boolean;
  eps: number;
}

export class Transformer {
  readonly cfg: ArchConfig;
  private readonly model: HwmModel;
  private readonly headDim: number;
  private readonly kvDim: number;
  private readonly qDim: number;
  private readonly kvRepeat: number;

  // 复用缓冲区，热路径内零分配
  private readonly x: Float32Array;
  private readonly xb: Float32Array;
  private readonly xb2: Float32Array;
  private readonly hb: Float32Array;
  private readonly hb2: Float32Array;
  private readonly q: Float32Array;
  private readonly k: Float32Array;
  private readonly v: Float32Array;
  private readonly scores: Float32Array;
  private readonly logits: Float32Array;

  private kCache!: Float32Array[];
  private vCache!: Float32Array[];
  private cacheLen = 0;

  constructor(model: HwmModel) {
    this.model = model;
    this.cfg = model.meta.arch as unknown as ArchConfig;
    const c = this.cfg;
    this.headDim = c.d_model / c.n_heads;
    this.kvDim = c.n_kv_heads * this.headDim;
    this.qDim = c.n_heads * this.headDim;
    this.kvRepeat = c.n_heads / c.n_kv_heads;

    this.x = new Float32Array(c.d_model);
    this.xb = new Float32Array(c.d_model);
    this.xb2 = new Float32Array(c.d_model);
    this.hb = new Float32Array(c.d_ff);
    this.hb2 = new Float32Array(c.d_ff);
    this.q = new Float32Array(this.qDim);
    this.k = new Float32Array(this.kvDim);
    this.v = new Float32Array(this.kvDim);
    this.scores = new Float32Array(c.max_seq);
    this.logits = new Float32Array(c.vocab_size);
    this.reset();
  }

  reset(): void {
    const c = this.cfg;
    this.kCache = [];
    this.vCache = [];
    for (let i = 0; i < c.n_layers; i++) {
      this.kCache.push(new Float32Array(c.max_seq * this.kvDim));
      this.vCache.push(new Float32Array(c.max_seq * this.kvDim));
    }
    this.cacheLen = 0;
  }

  get position(): number {
    return this.cacheLen;
  }

  private lookup(id: number, out: Float32Array): void {
    const w = this.model.get("tok_emb");
    const D = this.cfg.d_model;
    if (w.f32) {
      out.set(w.f32.subarray(id * D, id * D + D));
    } else if (w.i8 && w.scale) {
      const s = w.scale.length === this.cfg.vocab_size ? w.scale[id] : w.scale[0];
      for (let d = 0; d < D; d++) out[d] = w.i8[id * D + d] * s;
    } else {
      throw new Error("tok_emb 权重格式不可识别");
    }
  }

  /** 线性层：out = W · x，W 为 (outDim, inDim) 行主序。 */
  private linear(name: string, x: Float32Array, out: Float32Array): void {
    const w: WeightView = this.model.get(name);
    const outDim = w.shape[0];
    const inDim = w.shape.length > 1 ? w.shape[1] : 1;
    if (w.f32) {
      matmulT1(out, 0, w.f32, 0, x, 0, outDim, inDim);
    } else if (w.i8 && w.scale) {
      matmulT1Int8(out, 0, w.i8, 0, w.scale, 0, x, 0, outDim, inDim);
    } else {
      throw new Error(`权重 ${name} 缺少可用数据`);
    }
  }

  /**
   * 计算一个 token 在当前位置的前向结果。
   * @param id   token id
   * @param pos  绝对位置（用于 RoPE 与 KV Cache 写入）
   * @returns    logits（长度 vocab_size）
   */
  step(id: number, pos: number): Float32Array {
    const c = this.cfg;
    const D = c.d_model;
    const hd = this.headDim;

    this.lookup(id, this.x);

    for (let l = 0; l < c.n_layers; l++) {
      const p = `layers.${l}.`;
      const kc = this.kCache[l];
      const vc = this.vCache[l];

      rmsnormInplace(this.xb, 0, this.model.get(p + "attn_norm").f32 as Float32Array, D, c.eps);
      this.linear(p + "wq", this.xb, this.q);
      this.linear(p + "wk", this.xb, this.k);
      this.linear(p + "wv", this.xb, this.v);

      ropeInPlace(this.q, 0, c.n_heads, this.k, 0, c.n_kv_heads, hd, pos, c.rope_theta);

      const kOff = pos * this.kvDim;
      kc.set(this.k, kOff);
      vc.set(this.v, kOff);

      // 注意力：逐头计算，只与 pos 及之前的 key 做因果注意力
      this.xb2.fill(0);
      const scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < c.n_heads; h++) {
        const kvHead = Math.floor(h / this.kvRepeat);
        const qOff = h * hd;
        const kvBase = kvHead * hd;
        const n = pos + 1;
        for (let t = 0; t < n; t++) {
          let s = 0;
          const base = t * this.kvDim + kvBase;
          for (let d = 0; d < hd; d++) s += this.q[qOff + d] * kc[base + d];
          this.scores[t] = s * scale;
        }
        softmaxInplace(this.scores, 0, n);
        const outOff = h * hd;
        for (let d = 0; d < hd; d++) this.xb2[outOff + d] = 0;
        for (let t = 0; t < n; t++) {
          const wgt = this.scores[t];
          if (wgt === 0) continue;
          const base = t * this.kvDim + kvBase;
          for (let d = 0; d < hd; d++) this.xb2[outOff + d] += wgt * vc[base + d];
        }
      }

      this.linear(p + "wo", this.xb2, this.xb);
      for (let d = 0; d < D; d++) this.x[d] += this.xb[d];

      rmsnormInplace(this.xb, 0, this.model.get(p + "ffn_norm").f32 as Float32Array, D, c.eps);
      this.linear(p + "w1", this.xb, this.hb);
      this.linear(p + "w3", this.xb, this.hb2);
      swigluInplace(this.hb, 0, this.hb, 0, this.hb2, 0, c.d_ff);
      this.linear(p + "w2", this.hb, this.xb);
      for (let d = 0; d < D; d++) this.x[d] += this.xb[d];
    }

    rmsnormInplace(this.x, 0, this.model.get("final_norm").f32 as Float32Array, D, c.eps);
    if (c.tie_embeddings) {
      const w = this.model.get("tok_emb");
      if (w.f32) {
        matmulT1(this.logits, 0, w.f32, 0, this.x, 0, c.vocab_size, D);
      } else if (w.i8 && w.scale) {
        matmulT1Int8(this.logits, 0, w.i8, 0, w.scale, 0, this.x, 0, c.vocab_size, D);
      }
    } else {
      this.linear("lm_head", this.x, this.logits);
    }
    this.cacheLen = pos + 1;
    return this.logits;
  }

  /** 预填充整段 prompt，返回最后一个位置的 logits。 */
  prefill(ids: number[], startPos = 0): Float32Array {
    let out = this.logits;
    for (let i = 0; i < ids.length; i++) out = this.step(ids[i], startPos + i);
    return out;
  }

  /**
   * 取最后一层归一化后的隐藏状态（当前位置），供嵌入头与重排头使用。
   * 必须在 step() 之后调用，返回内部缓冲区的副本。
   */
  lastHidden(): Float32Array {
    return Float32Array.from(this.x);
  }
}

export const __all__ = ["Transformer", "ArchConfig"];
