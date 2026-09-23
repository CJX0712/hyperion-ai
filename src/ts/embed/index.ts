/**
 * M06 embed - 文本嵌入与交叉编码器打分（零第三方依赖）
 *
 * 作者：晨星
 *
 * 本模块是"检索为什么能work"的算力来源。它复用已经训练好的 Transformer 编码器，
 * 只是换了个头（head），因此模型体量不翻倍：
 *
 *   双塔头   emb_proj (embedDim, dModel)：查询与文档各跑一次编码器 -> 均值池化
 *            -> 投影 -> L2 归一化 -> 余弦相似度即内积。
 *   交叉头   ce_score (1, dModel)：把「查询 <sep> 文档」拼成一条序列跑编码器
 *            -> 均值池化 -> 内积得标量相关性分数。
 *
 * 两个头都是在 Python 侧用同一份合成数据的真值标签训练出来的（InfoNCE / 组内 softmax），
 * 因此检索质量是可评测的，而不是靠人眼判断。
 */

import { matmulT1, matmulT1Int8, l2normalizeInplace } from "../tensor/index.ts";
import type { HwmModel } from "../tensor/hwm.ts";
import { Transformer } from "../arch/index.ts";
import { SEP_ID, Tokenizer } from "../tokenizer/index.ts";

/** 线性投影：out = W · x，W 为 (outDim, inDim) 行主序，兼容 int8 权重。 */
function project(model: HwmModel, name: string, x: Float32Array, out: Float32Array): void {
  const w = model.get(name);
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

export class Embedder {
  readonly dim: number;
  private readonly model: HwmModel;
  private readonly tok: Tokenizer;
  private readonly enc: Transformer;
  private readonly pooled: Float32Array;

  constructor(model: HwmModel, tok: Tokenizer) {
    this.model = model;
    this.tok = tok;
    this.enc = new Transformer(model);
    this.dim = model.get("emb_proj").shape[0];
    this.pooled = new Float32Array(model.meta.arch.d_model as number);
  }

  /**
   * 编码单条文本：均值池化 -> 投影 -> L2 归一化。
   * 均值池化与 Python 训练侧 mean_pool 语义一致（在有效 token 上求均值）。
   */
  embed(text: string, maxLen = 160): Float32Array {
    const ids = this.tok.encode(text).slice(0, maxLen);
    const D = this.model.meta.arch.d_model as number;
    this.pooled.fill(0);
    if (ids.length === 0) return new Float32Array(this.dim);

    this.enc.reset();
    for (let i = 0; i < ids.length; i++) {
      this.enc.step(ids[i], i);
      const h = this.enc.lastHidden();
      for (let d = 0; d < D; d++) this.pooled[d] += h[d];
    }
    const inv = 1 / ids.length;
    for (let d = 0; d < D; d++) this.pooled[d] *= inv;

    const out = new Float32Array(this.dim);
    project(this.model, "emb_proj", this.pooled, out);
    l2normalizeInplace(out, 0, this.dim);
    return out;
  }

  embedBatch(texts: readonly string[], maxLen = 160): Float32Array[] {
    return texts.map((t) => this.embed(t, maxLen));
  }
}

export class CrossEncoder {
  private readonly model: HwmModel;
  private readonly tok: Tokenizer;
  private readonly enc: Transformer;
  private readonly pooled: Float32Array;
  private readonly scoreOut = new Float32Array(1);

  constructor(model: HwmModel, tok: Tokenizer) {
    this.model = model;
    this.tok = tok;
    this.enc = new Transformer(model);
    this.pooled = new Float32Array(model.meta.arch.d_model as number);
  }

  /** 相关性打分：越大越相关。查询与文档用 <sep> 拼接。 */
  score(query: string, doc: string, maxQuery = 32, maxDoc = 128): number {
    const q = this.tok.encode(query).slice(0, maxQuery);
    const d = this.tok.encode(doc).slice(0, maxDoc);
    const D = this.model.meta.arch.d_model as number;
    const ids = [...q, SEP_ID, ...d];
    this.pooled.fill(0);
    this.enc.reset();
    for (let i = 0; i < ids.length; i++) {
      this.enc.step(ids[i], i);
      const h = this.enc.lastHidden();
      for (let k = 0; k < D; k++) this.pooled[k] += h[k];
    }
    const inv = 1 / ids.length;
    for (let k = 0; k < D; k++) this.pooled[k] *= inv;
    project(this.model, "ce_score", this.pooled, this.scoreOut);
    return this.scoreOut[0];
  }
}

export const __all__ = ["Embedder", "CrossEncoder"];
