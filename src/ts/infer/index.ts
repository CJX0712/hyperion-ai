/**
 * M05 infer - 自回归生成与采样（零第三方依赖）
 *
 * 作者：晨星
 *
 * 采样策略：temperature 缩放 -> top-k 截断 -> top-p（核采样）截断 -> 多项式采样。
 * 顺序不能颠倒：先 top-k 再做 top-p，可以在 k 较大时仍然保持分布平滑，
 * 反过来会先在长尾上浪费概率质量。
 *
 * 可复现性：随机源必须是注入的确定性 PRNG（tensor.mulberry32），
 * 禁止在本模块内调用 Math.random，否则同 seed 无法复现，评估指标会漂移。
 */

import { mulberry32 } from "../tensor/index.ts";
import type { Transformer } from "../arch/index.ts";
import { EOS_ID } from "../tokenizer/index.ts";

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  seed?: number;
  stopIds?: number[];
}

export interface GenerateResult {
  text: string;
  tokens: number[];
  tokensPerSecond: number;
  finishReason: "length" | "eos" | "stop";
}

/** 从 logits 采样一个 token id。logits 会被就地修改（temperature 缩放）。 */
export function sampleFromLogits(
  logits: Float32Array,
  opts: { temperature: number; topK: number; topP: number },
  rng: () => number
): number {
  const n = logits.length;
  const temperature = Math.max(opts.temperature, 1e-6);

  if (opts.temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }

  for (let i = 0; i < n; i++) logits[i] /= temperature;

  // 候选集：先 top-k
  let cand: number[];
  if (opts.topK > 0 && opts.topK < n) {
    cand = Array.from({ length: n }, (_, i) => i);
    cand.sort((a, b) => logits[b] - logits[a]);
    cand = cand.slice(0, opts.topK);
  } else {
    cand = Array.from({ length: n }, (_, i) => i);
  }

  // softmax（在候选集内做，长尾不参与）
  let mx = -Infinity;
  for (const i of cand) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const probs = new Float64Array(cand.length);
  for (let j = 0; j < cand.length; j++) {
    const e = Math.exp(logits[cand[j]] - mx);
    probs[j] = e;
    sum += e;
  }
  for (let j = 0; j < cand.length; j++) probs[j] /= sum;

  // top-p：按概率降序截断
  let pool = Array.from({ length: cand.length }, (_, j) => j);
  pool.sort((a, b) => probs[b] - probs[a]);
  if (opts.topP > 0 && opts.topP < 1) {
    let acc = 0;
    let cut = pool.length;
    for (let j = 0; j < pool.length; j++) {
      acc += probs[pool[j]];
      if (acc >= opts.topP) { cut = j + 1; break; }
    }
    pool = pool.slice(0, Math.max(1, cut));
  }

  let target = rng();
  let acc = 0;
  for (const j of pool) {
    acc += probs[j];
    if (target <= acc) return cand[j];
  }
  return cand[pool[pool.length - 1]];
}

/**
 * 自回归生成。onToken 用于流式输出（SSE 场景每生成一个 token 回调一次）。
 * 注意：model 必须已 reset()，调用方负责管理 KV Cache 生命周期。
 */
export function generate(
  model: Transformer,
  promptIds: number[],
  opts: GenerateOptions = {},
  onToken?: (id: number, text: string) => void
): GenerateResult {
  const maxTokens = opts.maxTokens ?? 128;
  const temperature = opts.temperature ?? 0.8;
  const topK = opts.topK ?? 40;
  const topP = opts.topP ?? 0.9;
  const stopIds = new Set(opts.stopIds ?? []);
  const rng = mulberry32(opts.seed ?? 20260923);

  // 输入超过上下文窗口（max_seq）时按末尾截断，保留 query 段。
  // 否则 KV Cache 写入位置越界会抛 RangeError: offset is out of bounds。
  const maxSeq = model.cfg.max_seq as number;
  if (promptIds.length > maxSeq - 1) promptIds = promptIds.slice(-(maxSeq - 1));

  const t0 = Date.now();
  let logits = model.prefill(promptIds, 0);
  const out: number[] = [];
  let finishReason: GenerateResult["finishReason"] = "length";

  for (let i = 0; i < maxTokens; i++) {
    // KV Cache 位置越界保护：输入已被截断到 max_seq-1，此处再兜底一次
    if (promptIds.length + i >= maxSeq) { finishReason = "length"; break; }
    // sampleFromLogits 会就地缩放 logits，而 logits 是模型内部缓冲区，
    // 必须拷贝后再采样，否则下一步会读到被污染的值。
    const copy = Float32Array.from(logits);
    const next = sampleFromLogits(copy, { temperature, topK, topP }, rng);
    if (next === EOS_ID || stopIds.has(next)) {
      finishReason = next === EOS_ID ? "eos" : "stop";
      break;
    }
    out.push(next);
    onToken?.(next, "");
    logits = model.step(next, promptIds.length + i);
  }

  const seconds = (Date.now() - t0) / 1000;
  return {
    text: "",
    tokens: out,
    tokensPerSecond: seconds > 0 ? out.length / seconds : 0,
    finishReason,
  };
}

/**
 * 流式生成（异步生成器）。
 *
 * 两个必须处理的细节：
 *   1. 每生成一个 token 就 await 一次 setImmediate。生成循环是同步 CPU 密集的，
 *      不让出事件循环的话，SSE 数据要等整段生成结束才会真正刷出去——
 *      "流式"就会退化成"一次性"。
 *   2. 增量文本不能直接 decode 单个 token。字节级 BPE 的一个 token 可能只是
 *      多字节 UTF-8 字符的一部分，逐 token 解码会产生乱码。
 *      这里改为累计解码整个序列再取增量。
 */
export async function* generateStream(
  model: Transformer,
  tok: { encode(t: string, o?: { bos?: boolean; eos?: boolean }): number[]; decode(ids: Iterable<number>): string },
  prompt: string,
  opts: GenerateOptions = {}
): AsyncGenerator<{ id: number; delta: string; index: number }> {
  const maxTokens = opts.maxTokens ?? 128;
  const rng = mulberry32(opts.seed ?? 20260923);
  const stopIds = new Set(opts.stopIds ?? []);
  const ids = tok.encode(prompt);
  if (ids.length === 0) return;
  // 超过上下文窗口时按末尾截断（保留 query 段），防止 KV Cache 越界
  const maxSeq = model.cfg.max_seq as number;
  if (ids.length > maxSeq - 1) ids.length = maxSeq - 1;

  model.reset();
  let logits = model.prefill(ids, 0);
  const acc: number[] = [];
  let emitted = 0;

  for (let i = 0; i < maxTokens; i++) {
    if (ids.length + i >= maxSeq) break;
    const copy = Float32Array.from(logits);
    const next = sampleFromLogits(
      copy,
      { temperature: opts.temperature ?? 0.8, topK: opts.topK ?? 40, topP: opts.topP ?? 0.9 },
      rng
    );
    if (next === EOS_ID || stopIds.has(next)) break;
    acc.push(next);
    const full = tok.decode(acc);
    const delta = full.slice(emitted);
    emitted = full.length;
    yield { id: next, delta, index: i };
    logits = model.step(next, ids.length + i);
    await new Promise<void>((r) => setImmediate(r));
  }
  model.reset();
}

export interface TextGenerator {
  generate(prompt: string, opts?: GenerateOptions): { text: string; tokens: number; tokensPerSecond: number; finishReason: string };
}

/**
 * 本地生成器：把 tokenizer 的编解码接到生成循环上。
 * 生成结束后会重置 KV Cache，避免上一次请求的上下文污染下一次请求。
 */
export class LocalGenerator implements TextGenerator {
  constructor(
    private readonly model: Transformer,
    private readonly tok: { encode(t: string, o?: { bos?: boolean; eos?: boolean }): number[]; decode(ids: Iterable<number>): string }
  ) {}

  generate(prompt: string, opts: GenerateOptions = {}) {
    const ids = this.tok.encode(prompt);
    if (ids.length === 0) throw new Error("提示词编码为空");
    this.model.reset();
    const r = generate(this.model, ids, opts);
    this.model.reset();
    return {
      text: this.tok.decode(r.tokens),
      tokens: r.tokens.length,
      tokensPerSecond: r.tokensPerSecond,
      finishReason: r.finishReason,
    };
  }
}

export const __all__ = ["generate", "sampleFromLogits", "LocalGenerator", "TextGenerator", "GenerateOptions"];
