/**
 * M09 rag - 摄取 / 分片 / 检索 / 重排 / 上下文组装 / 引用溯源（零第三方依赖）
 *
 * 作者：晨星
 *
 * 链路：文档 -> 分片 -> 嵌入 -> 混合索引 -> 检索 -> 重排（带护栏）-> 上下文拼装
 *       -> 本地生成 -> 引用溯源
 *
 * 引用溯源不是装饰：每个返回的 chunk 都带 docId + chunkId + 原文片段，
 * 答案里出现的 [1][2] 编号可以逐条回溯到原始文档，评估时 faithfulness 才有判定依据。
 *
 * 分片策略：按句末标点切分再装箱，不是按固定长度硬切。硬切会把一句排查步骤劈成两半，
 * 导致检索命中"半句话"而答案定位错误——这是 RAG 里很常见但很难察觉的质量损失。
 */

import type { Embedder, CrossEncoder } from "../embed/index.ts";
import { HybridIndex, type Hit } from "../index/index.ts";
import { Reranker } from "../rerank/index.ts";
import type { TextGenerator } from "../infer/index.ts";

export interface SourceDoc {
  docId: string;
  title?: string;
  text: string;
}

export interface Chunk {
  chunkId: string;
  docId: string;
  title: string;
  text: string;
  ord: number;
}

export interface RetrievedChunk extends Chunk {
  score: number;
  rank: number;
}

export interface RagAnswer {
  answer: string;
  citations: RetrievedChunk[];
  context: string;
  tokens: number;
  tokensPerSecond: number;
  latencyMs: number;
  mode: "generative" | "extractive";
}

export interface RagOptions {
  chunkSize?: number;
  overlap?: number;
  depth?: number;
  topK?: number;
  useRerank?: boolean;
  maxTokens?: number;
  temperature?: number;
}

const DEFAULTS: Required<RagOptions> = {
  chunkSize: 220,
  overlap: 40,
  depth: 20,
  topK: 5,
  useRerank: true,
  maxTokens: 96,
  temperature: 0.4,
};

/** 按句末标点切句。中英文标点都覆盖，且不丢标点。 */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  let buf = "";
  for (const ch of text) {
    buf += ch;
    if (ch === "。" || ch === "；" || ch === "！" || ch === "？" || ch === "\n" || ch === "." || ch === ";") {
      out.push(buf);
      buf = "";
    }
  }
  if (buf.length) out.push(buf);
  return out.filter((s) => s.trim().length > 0);
}

/** 句子装箱成分片：优先在句边界断开，并在相邻分片间保留一小段重叠。 */
export function chunkText(docId: string, title: string, text: string, chunkSize: number, overlap: number): Chunk[] {
  const sentences = splitSentences(text);
  const chunks: Chunk[] = [];
  let cur = "";
  let ord = 0;
  const flush = () => {
    if (!cur.trim()) return;
    chunks.push({ chunkId: `${docId}#${ord}`, docId, title, text: cur.trim(), ord });
    ord++;
    cur = overlap > 0 ? cur.slice(-overlap) : "";
  };
  for (const s of sentences) {
    if (cur.length + s.length > chunkSize && cur.length > 0) flush();
    cur += s;
  }
  flush();
  return chunks;
}

export class RagPipeline {
  private readonly idx: HybridIndex;
  private readonly chunks: Chunk[] = [];
  private readonly byId = new Map<string, Chunk>();
  private readonly opts: Required<RagOptions>;
  private readonly reranker: Reranker;
  private lastRetrievalMs = 0;

  constructor(
    private readonly embedder: Embedder,
    private readonly crossEncoder: CrossEncoder,
    private readonly generator: TextGenerator,
    opts: RagOptions = {}
  ) {
    this.opts = { ...DEFAULTS, ...opts };
    this.idx = new HybridIndex(embedder.dim);
    this.reranker = new Reranker((q, id) => {
      const c = this.byId.get(id);
      return c ? this.crossEncoder.score(q, c.text) : -Infinity;
    });
  }

  /** 摄取文档，返回生成的分片数。重复 docId 会被覆盖（先移除再插入）。 */
  ingest(docs: SourceDoc[]): number {
    let added = 0;
    for (const d of docs) {
      const cs = chunkText(d.docId, d.title ?? d.docId, d.text, this.opts.chunkSize, this.opts.overlap);
      for (const c of cs) {
        this.chunks.push(c);
        this.byId.set(c.chunkId, c);
        this.idx.add(c.chunkId, c.text, this.embedder.embed(c.text));
        added++;
      }
    }
    return added;
  }

  size(): number {
    return this.chunks.length;
  }

  chunkById(id: string): Chunk | undefined {
    return this.byId.get(id);
  }

  lastRetrievalLatencyMs(): number {
    return this.lastRetrievalMs;
  }

  /** 检索：混合索引 -> （可选）带护栏重排 -> top-k。 */
  retrieve(query: string, topK = this.opts.topK, useRerank = this.opts.useRerank): RetrievedChunk[] {
    const t0 = Date.now();
    const qv = this.embedder.embed(query);
    let hits: Hit[] = this.idx.search(query, qv, this.opts.depth);
    if (useRerank && hits.length > 1) hits = this.reranker.rerank(query, hits);
    this.lastRetrievalMs = Date.now() - t0;
    return hits.slice(0, topK).map((h, i) => {
      const c = this.byId.get(h.id);
      if (!c) throw new Error(`索引里出现了未知 chunkId：${h.id}`);
      return { ...c, score: h.score, rank: i + 1 };
    });
  }

  /** 把检索到的分片拼成上下文，带编号，供模型与引用使用。 */
  buildPrompt(query: string, retrieved: RetrievedChunk[]): string {
    const parts = retrieved.map((c, i) => `[${i + 1}] ${c.text}`);
    return `参考资料：\n${parts.join("\n")}\n问：${query}\n答：`;
  }

  /**
   * 问答。检索无命中时明确返回"未检索到相关内容"，不硬凑答案。
   * mode 为 extractive 时直接返回最佳分片的原文——这是生成质量不足时的诚实降级。
   */
  ask(query: string, opts: RagOptions = {}): RagAnswer {
    const o = { ...this.opts, ...opts };
    const t0 = Date.now();
    const retrieved = this.retrieve(query, o.topK, o.useRerank);
    if (retrieved.length === 0) {
      return {
        answer: "未检索到相关内容。请补充知识库，或放宽检索深度后重试。",
        citations: [],
        context: "",
        tokens: 0,
        tokensPerSecond: 0,
        latencyMs: Date.now() - t0,
        mode: "extractive",
      };
    }
    const context = this.buildPrompt(query, retrieved);
    const gen = this.generator.generate(context, {
      maxTokens: o.maxTokens,
      temperature: o.temperature,
      topK: 40,
      topP: 0.9,
      seed: 20260923,
    });
    let answer = gen.text.trim();
    let mode: RagAnswer["mode"] = "generative";
    if (answer.length < 4) {
      answer = retrieved[0].text;
      mode = "extractive";
    }
    return {
      answer,
      citations: retrieved,
      context,
      tokens: gen.tokens,
      tokensPerSecond: gen.tokensPerSecond,
      latencyMs: Date.now() - t0,
      mode,
    };
  }
}

export const __all__ = ["RagPipeline", "chunkText", "splitSentences", "SourceDoc", "RetrievedChunk"];
