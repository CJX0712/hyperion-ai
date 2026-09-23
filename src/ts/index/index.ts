/**
 * M07 index - 检索索引：稠密向量 + BM25 稀疏 + RRF 融合（零第三方依赖）
 *
 * 作者：晨星
 *
 * 复用的业界成果（算法层，非自研）：
 *   BM25（Robertson & Zaragoza）、倒数排名融合 RRF（Cormack et al.）。
 *   双通道融合而不是二选一，是因为两者失效模式互补：稠密通道负责语义泛化，
 *   稀疏通道负责专有名词与错误码这类字面精确匹配。
 *
 * 自研部分：中文按字符二元组建倒排（中文没有空格分词，而引入分词器会破坏
 * "零第三方依赖"这条交付红线）、确定性 tie-break、可复现的排序。
 *
 * 可复现性是硬要求（RAG 领域最大的差评来源就是"重启后结果变了"）：
 *   * 所有打分是纯函数，输入相同则输出逐位相同；
 *   * 排序比较器在分数相等时按 id 字典序兜底，杜绝依赖插入顺序；
 *   * 不使用任何随机数、不依赖 Map 的迭代顺序做排序。
 */

import { cosine } from "../tensor/index.ts";

export interface Hit {
  id: string;
  score: number;
}

/** 稳定性比较器：分数降序，同分按 id 升序。 */
export function compareHits(a: Hit, b: Hit): number {
  if (b.score !== a.score) return b.score - a.score;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * BM25 词元化。规则（与 tokenizer.pretokenize 的类别判定一致，但输出更利于检索）：
 *   * ASCII 词 -> 小写原词
 *   * CJK 串   -> 字符 + 字符二元组（中文用小粒度 n-gram 替代分词器）
 *   * 其它     -> 跳过（标点不参与倒排）
 */
export function tokenizeForBM25(text: string): string[] {
  const out: string[] = [];
  let i = 0;
  const n = text.length;
  const isWord = (c: number) =>
    (c >= 0x30 && c <= 0x39) || (c >= 0x41 && c <= 0x5a) || (c >= 0x61 && c <= 0x7a) || c === 0x5f;
  const isCjk = (c: number) =>
    (c >= 0x3400 && c <= 0x4dbf) || (c >= 0x4e00 && c <= 0x9fff) || (c >= 0xf900 && c <= 0xfaff);
  while (i < n) {
    const cp = text.codePointAt(i) as number;
    if (isWord(cp)) {
      let j = i;
      while (j < n && isWord(text.codePointAt(j) as number)) j++;
      out.push(text.slice(i, j).toLowerCase());
      i = j;
    } else if (isCjk(cp)) {
      let j = i;
      while (j < n && isCjk(text.codePointAt(j) as number)) j++;
      const seg = text.slice(i, j);
      for (let k = 0; k < seg.length; k++) out.push(seg[k]);
      for (let k = 0; k + 1 < seg.length; k++) out.push(seg.slice(k, k + 2));
      i = j;
    } else {
      i++;
    }
  }
  return out;
}

/** 稠密索引：暴力精确余弦。万级以内无需近似索引，精确性优先于吞吐。 */
export class DenseIndex {
  private readonly ids: string[] = [];
  private readonly vecs: Float32Array[] = [];
  readonly dim: number;

  constructor(dim: number) {
    this.dim = dim;
  }

  add(id: string, vec: Float32Array): void {
    if (vec.length !== this.dim) throw new Error(`向量维度不符：期望 ${this.dim}，实际 ${vec.length}`);
    this.ids.push(id);
    this.vecs.push(vec);
  }

  size(): number {
    return this.ids.length;
  }

  search(query: Float32Array, k: number): Hit[] {
    const hits: Hit[] = [];
    for (let i = 0; i < this.ids.length; i++) {
      hits.push({ id: this.ids[i], score: cosine(query, 0, this.vecs[i], 0, this.dim) });
    }
    hits.sort(compareHits);
    return hits.slice(0, Math.max(1, k));
  }

  vectorOf(id: string): Float32Array | undefined {
    const i = this.ids.indexOf(id);
    return i < 0 ? undefined : this.vecs[i];
  }
}

/** BM25 稀疏索引。k1/b 取信息服务领域常用默认值。 */
export class BM25Index {
  private readonly ids: string[] = [];
  private readonly tfs: Map<string, number>[] = [];
  private readonly lens: number[] = [];
  private readonly df = new Map<string, number>();
  private totalLen = 0;

  constructor(private readonly k1 = 1.2, private readonly b = 0.75) {}

  add(id: string, text: string): void {
    const toks = tokenizeForBM25(text);
    const tf = new Map<string, number>();
    for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    this.ids.push(id);
    this.tfs.push(tf);
    this.lens.push(toks.length);
    this.totalLen += toks.length;
  }

  size(): number {
    return this.ids.length;
  }

  private get avgLen(): number {
    return this.ids.length ? this.totalLen / this.ids.length : 1;
  }

  search(query: string, k: number): Hit[] {
    const N = this.ids.length;
    if (N === 0) return [];
    const qToks = Array.from(new Set(tokenizeForBM25(query)));
    const avg = this.avgLen || 1;
    const scores = new Float64Array(N);
    for (const t of qToks) {
      const df = this.df.get(t);
      if (!df) continue;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      for (let i = 0; i < N; i++) {
        const f = this.tfs[i].get(t);
        if (!f) continue;
        const denom = f + this.k1 * (1 - this.b + (this.b * this.lens[i]) / avg);
        scores[i] += idf * ((f * (this.k1 + 1)) / denom);
      }
    }
    const hits: Hit[] = [];
    for (let i = 0; i < N; i++) if (scores[i] > 0) hits.push({ id: this.ids[i], score: scores[i] });
    hits.sort(compareHits);
    return hits.slice(0, Math.max(1, k));
  }
}

/**
 * 倒数排名融合 RRF：score(d) = Σ_r w_r / (k + rank_r(d))。
 * 融合的是"排名"而不是"分数"，因此不需要把余弦相似度和 BM25 分数归一化到同一量纲
 * —— 这是它比加权求和更稳健的原因。
 */
export function rrfFuse(rankings: { hits: Hit[]; weight?: number }[], k = 60): Hit[] {
  const acc = new Map<string, number>();
  for (const r of rankings) {
    const w = r.weight ?? 1;
    r.hits.forEach((h, i) => {
      acc.set(h.id, (acc.get(h.id) ?? 0) + w / (k + i + 1));
    });
  }
  const hits: Hit[] = [...acc.entries()].map(([id, score]) => ({ id, score }));
  hits.sort(compareHits);
  return hits;
}

/** 混合索引：一入口管理双通道，返回融合后的候选。 */
export class HybridIndex {
  readonly dense: DenseIndex;
  readonly sparse: BM25Index;

  constructor(dim: number) {
    this.dense = new DenseIndex(dim);
    this.sparse = new BM25Index();
  }

  add(id: string, text: string, vec: Float32Array): void {
    this.dense.add(id, vec);
    this.sparse.add(id, text);
  }

  size(): number {
    return this.dense.size();
  }

  /**
   * 双通道检索 + RRF 融合。
   * @param depth 每个通道各取前 depth 条参与融合（候选深度，不是最终返回数）
   */
  search(query: string, queryVec: Float32Array, depth = 20, weights = { dense: 1, sparse: 1 }): Hit[] {
    const d = this.dense.search(queryVec, depth);
    const s = this.sparse.search(query, depth);
    return rrfFuse([
      { hits: d, weight: weights.dense },
      { hits: s, weight: weights.sparse },
    ]);
  }
}

export const __all__ = ["Hit", "DenseIndex", "BM25Index", "HybridIndex", "rrfFuse", "tokenizeForBM25", "compareHits"];
