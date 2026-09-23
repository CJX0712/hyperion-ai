/**
 * M12 eval - 可复现的检索与生成评估（零第三方依赖）
 *
 * 作者：晨星
 *
 * 三条纪律（都是踩过坑之后定下来的）：
 *
 *   1. **评估必须跑在全新管道上**。复用运行期已经写入数据的单例管道，
 *      会让 doc_id 重复、索引里混进上一次的残留，表现为"指标忽高忽低"。
 *      这里的 EvalRunner 每次都新建 RagPipeline 与索引。
 *
 *   2. **指标必须确定性**。固定 seed、固定温度、固定检索参数；
 *      任何随机性都会让 CI 里的阈值断言变成偶发失败。
 *
 *   3. **护栏必须有回归门禁**。重排护栏的核心承诺是"可以不变好，但绝不变差"，
 *      所以这里的 checks 里有一条硬断言：开启重排后的 top-1 命中数
 *      不得低于关闭重排时的 top-1 命中数。
 *
 * 指标定义：
 *   recall@k  ---- 前 k 条候选中包含标准答案文档的比例
 *   MRR       ---- 标准答案文档排名的倒数均值
 *   grounded  ---- 生成答案与引用片段的重叠度（4-gram 覆盖率），
 *                  作为 faithfulness 的可计算代理指标，不依赖外部裁判模型
 */

import { readFileSync } from "node:fs";
import { RagPipeline, type SourceDoc } from "../rag/index.ts";
import type { Embedder, CrossEncoder } from "../embed/index.ts";
import type { TextGenerator } from "../infer/index.ts";

export interface EvalCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface EvalReport {
  suite: string;
  seed: number;
  cases: number;
  metrics: {
    recallAt1: number;
    recallAt3: number;
    recallAt5: number;
    mrr: number;
    recallAt1NoRerank: number;
    recallAt3NoRerank: number;
    recallAt5NoRerank: number;
    mrrNoRerank: number;
    retrievalP50Ms: number;
    retrievalP95Ms: number;
  };
  invariants: EvalCheck[];
  generative: {
    cases: number;
    nonEmpty: number;
    grounded: number;
    avgTokensPerSecond: number;
    answerModeExtractive: number;
  };
}

interface QaRow { query: string; gold_doc_id: string }
interface DocRow { doc_id: string; title: string; text: string; kind: string }

function readJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as T);
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[idx];
}

/** 4-gram 覆盖率：答案有多少比例的内容能在引用片段里找到出处。 */
export function groundedRatio(answer: string, citations: string[], n = 4): number {
  const grams = new Set<string>();
  const norm = (s: string) => s.replace(/\s+/g, "");
  const a = norm(answer);
  if (a.length < n) return a.length === 0 ? 0 : 1;
  for (let i = 0; i + n <= a.length; i++) grams.add(a.slice(i, i + n));
  if (grams.size === 0) return 0;
  const ref = new Set<string>();
  for (const c of citations) {
    const t = norm(c);
    for (let i = 0; i + n <= t.length; i++) ref.add(t.slice(i, i + n));
  }
  let hit = 0;
  for (const g of grams) if (ref.has(g)) hit++;
  return hit / grams.size;
}

export class EvalRunner {
  constructor(
    private readonly embedder: Embedder,
    private readonly crossEncoder: CrossEncoder,
    private readonly generator: TextGenerator,
    private readonly corpusPath: string,
    private readonly qaPath: string,
    private readonly seed = 20260923
  ) {}

  /** 每次调用都构建全新索引与管道，保证不与运行期状态互相污染。 */
  private freshPipeline(): { rag: RagPipeline; docs: DocRow[]; qa: QaRow[] } {
    const docs = readJsonl<DocRow>(this.corpusPath);
    const qa = readJsonl<QaRow>(this.qaPath);
    const rag = new RagPipeline(this.embedder, this.crossEncoder, this.generator);
    rag.ingest(docs.map((d) => ({ docId: d.doc_id, title: d.title, text: d.text }) satisfies SourceDoc));
    return { rag, docs, qa };
  }

  run(opts: { generativeCases?: number; topK?: number } = {}): EvalReport {
    const generativeCases = opts.generativeCases ?? 40;
    const topK = opts.topK ?? 5;
    const { rag, qa } = this.freshPipeline();

    const acc = {
      r1: 0, r3: 0, r5: 0, mrr: 0,
      r1n: 0, r3n: 0, r5n: 0, mrrn: 0,
    };
    const latencies: number[] = [];

    for (const case_ of qa) {
      const withR = rag.retrieve(case_.query, topK, true);
      const noR = rag.retrieve(case_.query, topK, false);
      latencies.push(rag.lastRetrievalLatencyMs());

      const rankOf = (list: typeof withR) => {
        const i = list.findIndex((c) => c.docId === case_.gold_doc_id);
        return i < 0 ? Infinity : i + 1;
      };
      const rw = rankOf(withR);
      const rn = rankOf(noR);
      if (rw === 1) acc.r1++;
      if (rw <= 3) acc.r3++;
      if (rw <= 5) acc.r5++;
      if (Number.isFinite(rw)) acc.mrr += 1 / rw;
      if (rn === 1) acc.r1n++;
      if (rn <= 3) acc.r3n++;
      if (rn <= 5) acc.r5n++;
      if (Number.isFinite(rn)) acc.mrrn += 1 / rn;
    }

    const N = Math.max(qa.length, 1);
    const metrics = {
      recallAt1: acc.r1 / N,
      recallAt3: acc.r3 / N,
      recallAt5: acc.r5 / N,
      mrr: acc.mrr / N,
      recallAt1NoRerank: acc.r1n / N,
      recallAt3NoRerank: acc.r3n / N,
      recallAt5NoRerank: acc.r5n / N,
      mrrNoRerank: acc.mrrn / N,
      retrievalP50Ms: percentile(latencies, 50),
      retrievalP95Ms: percentile(latencies, 95),
    };

    // ---- 生成质量抽样
    let nonEmpty = 0;
    let grounded = 0;
    let extractive = 0;
    let tpsSum = 0;
    const sample = qa.slice(0, Math.min(generativeCases, qa.length));
    for (const c of sample) {
      const ans = rag.ask(c.query, { topK: 3, maxTokens: 64, temperature: 0.2 });
      if (ans.answer.trim().length > 0) nonEmpty++;
      if (ans.mode === "extractive") extractive++;
      const ratio = groundedRatio(ans.answer, ans.citations.map((x) => x.text));
      if (ratio >= 0.5) grounded++;
      tpsSum += ans.tokensPerSecond;
    }

    const invariants: EvalCheck[] = [
      {
        name: "检索可复现（同 query 两次结果逐条相同）",
        passed: (() => {
          const q = qa[0]?.query ?? "";
          const a = rag.retrieve(q, topK, true).map((c) => c.chunkId).join("|");
          const b = rag.retrieve(q, topK, true).map((c) => c.chunkId).join("|");
          return a === b && a.length > 0;
        })(),
        detail: "比较两次检索的 chunkId 序列",
      },
      {
        name: "重排护栏：top-1 命中数不劣于纯融合",
        passed: acc.r1 >= acc.r1n,
        detail: `开启重排 top-1=${acc.r1}，关闭重排 top-1=${acc.r1n}`,
      },
      {
        name: `recall@${topK} 达到基线 0.80`,
        passed: metrics.recallAt5 >= 0.8,
        detail: `实际 ${metrics.recallAt5.toFixed(4)}`,
      },
      {
        name: "MRR 达到基线 0.70",
        passed: metrics.mrr >= 0.7,
        detail: `实际 ${metrics.mrr.toFixed(4)}`,
      },
      {
        name: "生成答案非空率 100%",
        passed: nonEmpty === sample.length,
        detail: `${nonEmpty}/${sample.length}`,
      },
    ];

    return {
      suite: "hyperion-rag-golden",
      seed: this.seed,
      cases: qa.length,
      metrics,
      invariants,
      generative: {
        cases: sample.length,
        nonEmpty,
        grounded,
        avgTokensPerSecond: sample.length ? tpsSum / sample.length : 0,
        answerModeExtractive: extractive,
      },
    };
  }
}

/** 把报告格式化成终端可读的文本。不使用 emoji，统一用 [PASS]/[FAIL] 标记。 */
export function formatReport(r: EvalReport): string {
  const lines: string[] = [];
  const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
  lines.push(`评估套件：${r.suite}（seed=${r.seed}，用例 ${r.cases} 条）`);
  lines.push(`  recall@1  ${pct(r.metrics.recallAt1)}   （无重排 ${pct(r.metrics.recallAt1NoRerank)}）`);
  lines.push(`  recall@3  ${pct(r.metrics.recallAt3)}   （无重排 ${pct(r.metrics.recallAt3NoRerank)}）`);
  lines.push(`  recall@5  ${pct(r.metrics.recallAt5)}   （无重排 ${pct(r.metrics.recallAt5NoRerank)}）`);
  lines.push(`  MRR       ${r.metrics.mrr.toFixed(4)}      （无重排 ${r.metrics.mrrNoRerank.toFixed(4)}）`);
  lines.push(`  检索延迟  P50 ${r.metrics.retrievalP50Ms} ms / P95 ${r.metrics.retrievalP95Ms} ms`);
  lines.push(
    `  生成质量  非空 ${r.generative.nonEmpty}/${r.generative.cases}，` +
      `有据可依 ${r.generative.grounded}/${r.generative.cases}，` +
      `平均 ${r.generative.avgTokensPerSecond.toFixed(1)} tok/s，` +
      `降级为抽取式 ${r.generative.answerModeExtractive} 次`
  );
  lines.push("  不变量：");
  for (const c of r.invariants) {
    lines.push(`    [${c.passed ? "PASS" : "FAIL"}] ${c.name} —— ${c.detail}`);
  }
  return lines.join("\n");
}

export const __all__ = ["EvalRunner", "formatReport", "groundedRatio", "EvalReport"];
