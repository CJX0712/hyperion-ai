/**
 * M08 rerank - 交叉编码器重排 + 融合护栏（零第三方依赖）
 *
 * 作者：晨星
 *
 * 为什么需要"护栏"（这是本模块存在的全部理由，也是踩过坑之后的结论）：
 *
 *   朴素做法是把重排器的输出直接当作最终排序。但重排器在查询语言上力不从心时
 *   —— 最典型的是英文重排器遇到中文查询 —— 它会系统性地把正确答案压到后面，
 *   而且没有任何制衡：融合阶段辛苦保留的正确候选，在重排阶段被单方面丢弃。
 *   实测（本系统的回归集）表现为 top-1 命中率明显下降。更隐蔽的一类退化是：
 *   一条在混合检索（稠密 + BM25）里排第一的强词法/语义命中，会被重排器单方面的
 *   错误打分挤出 top-k，导致"用户刚入库的文档、用精确关键词却查不到"。
 *
 *   护栏的做法：**分数级归一化融合**，而不是"排名级 RRF 再融合"。
 *   两路信号各自做 min-max 归一化到 [0,1]，再按权重相加：
 *       final = w_fuse · fusedNorm + w_rerank · ceNorm
 *   默认 w_fuse : w_rerank = 4 : 1，融合通道主导。这样融合通道的"最高分者"
 *   始终保留主导权（它的 fusedNorm = 1.0，任何候选都追不上），交叉编码器只能在
 *   融合排序的近邻区间内做精细化重排，无法再把强命中挤出前列。
 *
 *   为什么不用"排名级 RRF 再融合"：RRF 只看排名不看分数差距，二次融合时只要交叉
 *   编码器把一个候选抬到 scored-rank-1，它就能同时吃满两路加成，反而可能反超融合
 *   通道的 rank-1 强命中（这正是 e2e 验收里 e2e-1 被挤出 top-5 的根因）。分数级
 *   融合尊重融合分数的"量级差"，从机制上杜绝这类退化。
 *
 *   不变量（由 selfcheck 机械校验）：在中文黄金集上，
 *   开启重排后的 top-1 命中数**不得低于**关闭重排时的 top-1 命中数。
 *   也就是说：重排可以不变好，但绝不允许变差。
 */

import { compareHits, type Hit } from "../index/index.ts";

/** 打分器接口：越大越相关。生产实现是 CrossEncoder，测试实现是任意确定性函数。 */
export type ScoreFn = (query: string, docId: string) => number;

export interface RerankOptions {
  /** 送入重排器的候选深度，默认 32。过大只会增加延迟。 */
  depth?: number;
  /** 是否开启融合护栏。默认开启——关掉它会复现上面描述的退化。 */
  guardrail?: boolean;
  /** 交叉编码器分数在分数级融合中的权重。默认 1，且恒小于 fuseWeight。 */
  rerankWeight?: number;
  /** 原始融合分数在分数级融合中的权重。默认 4，融合通道主导。 */
  fuseWeight?: number;
}

export const DEFAULT_RERANK: Required<RerankOptions> = {
  depth: 32,
  guardrail: true,
  rerankWeight: 1,
  fuseWeight: 4,
};

export class Reranker {
  private readonly opts: Required<RerankOptions>;

  constructor(private readonly score: ScoreFn, opts: RerankOptions = {}) {
    this.opts = { ...DEFAULT_RERANK, ...opts };
  }

  /** 全量重排（无护栏）。交叉编码器分数直接决定顺序——对照组，允许重排器夺冠。 */
  rerankPlain(query: string, fused: Hit[]): Hit[] {
    const depth = Math.min(fused.length, this.opts.depth);
    const head = fused.slice(0, depth);
    const scored: Hit[] = head.map((h) => ({ id: h.id, score: this.score(query, h.id) }));
    scored.sort(compareHits);
    return [...scored, ...fused.slice(depth)];
  }

  /** 带护栏的重排（默认路径）：分数级归一化融合，融合通道主导。 */
  rerank(query: string, fused: Hit[]): Hit[] {
    if (fused.length === 0) return [];
    const depth = Math.min(fused.length, this.opts.depth);
    const head = fused.slice(0, depth);
    const scored = head.map((h) => ({ id: h.id, fused: h.score, ce: this.score(query, h.id) }));

    if (!this.opts.guardrail) {
      scored.sort((a, b) => b.ce - a.ce);
      return [...scored.map((s) => ({ id: s.id, score: s.ce })), ...fused.slice(depth)];
    }

    // 两路分数各自 min-max 归一化到 [0,1]，再按权重相加。融合通道的"最高分者"
    // 其 fusedNorm = 1.0，在 w_fuse >= w_rerank 时任何候选都追不上它，
    // 从而保证强命中不会被交叉编码器的个别错误打分挤出前列。
    const fMin = Math.min(...scored.map((s) => s.fused));
    const fMax = Math.max(...scored.map((s) => s.fused));
    const cMin = Math.min(...scored.map((s) => s.ce));
    const cMax = Math.max(...scored.map((s) => s.ce));
    const fR = fMax - fMin || 1;
    const cR = cMax - cMin || 1;
    const out: Hit[] = scored.map((s) => ({
      id: s.id,
      score: this.opts.fuseWeight * ((s.fused - fMin) / fR) + this.opts.rerankWeight * ((s.ce - cMin) / cR),
    }));
    out.sort(compareHits);
    return [...out, ...fused.slice(depth)];
  }
}

export const __all__ = ["Reranker", "ScoreFn", "RerankOptions", "DEFAULT_RERANK"];
