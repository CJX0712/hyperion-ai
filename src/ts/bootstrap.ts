/**
 * 装配层 - 把各模块按依赖方向组装成可运行的服务
 *
 * 作者：晨星
 *
 * 依赖方向严格向下，本文件是唯一允许"跨层引用"的地方：
 *   tensor -> tokenizer / arch -> infer / embed -> index -> rerank -> rag -> agent / serve / eval
 *
 * 入口只做装配、不做业务：这里没有任何算法实现，只有 new、连接、注册。
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { HwmModel } from "./tensor/hwm.ts";
import { Tokenizer } from "./tokenizer/index.ts";
import { Transformer } from "./arch/index.ts";
import { LocalGenerator } from "./infer/index.ts";
import { CrossEncoder, Embedder } from "./embed/index.ts";
import { RagPipeline } from "./rag/index.ts";
import { ToolRegistry, runAgent, evaluateExpression, type AgentResult } from "./agent/index.ts";
import { EvalRunner } from "./eval/index.ts";

export interface BootstrapOptions {
  assetsDir: string;
  /** 是否把语料灌进检索索引（serve / e2e 需要，纯单模块校验不需要） */
  ingestCorpus?: boolean;
  /** 是否启用交叉编码器重排 */
  useRerank?: boolean;
}

export interface Bootstrap {
  assetsDir: string;
  model: HwmModel;
  tokenizer: Tokenizer;
  transformer: Transformer;
  generator: LocalGenerator;
  embedder: Embedder;
  crossEncoder: CrossEncoder;
  rag: RagPipeline;
  registry: ToolRegistry;
  evalRunner: EvalRunner;
  modelInfo: Record<string, unknown>;
  runAgentTask: (task: string, maxSteps: number) => AgentResult;
}

export function bootstrap(opts: BootstrapOptions): Bootstrap {
  const dir = opts.assetsDir;
  const modelPath = join(dir, "model.i8.hwm");
  const tokPath = join(dir, "tokenizer.json");
  const corpusPath = join(dir, "corpus.jsonl");
  const qaPath = join(dir, "qa.jsonl");

  for (const p of [modelPath, tokPath]) {
    if (!existsSync(p)) throw new Error(`缺少必需产物：${p}。请先运行 npm run train 生成。`);
  }

  const model = HwmModel.load(modelPath);
  const tokenizer = Tokenizer.load(tokPath);
  const transformer = new Transformer(model);
  const generator = new LocalGenerator(transformer, tokenizer);
  const embedder = new Embedder(model, tokenizer);
  const crossEncoder = new CrossEncoder(model, tokenizer);
  const rag = new RagPipeline(embedder, crossEncoder, generator, { useRerank: opts.useRerank !== false });

  if (opts.ingestCorpus && existsSync(corpusPath)) {
    const docs = readFileSync(corpusPath, "utf8")
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as { doc_id: string; title: string; text: string });
    rag.ingest(docs.map((d) => ({ docId: d.doc_id, title: d.title, text: d.text })));
  }

  // ---- 工具注册（每个工具都必须有确定性的、可单测的行为）
  const registry = new ToolRegistry();
  registry.register({
    name: "search_kb",
    description: "在本地知识库中检索与问题相关的片段，返回带来源的原文",
    run: (input: string) => {
      const hits = rag.retrieve(input, 3, true);
      if (hits.length === 0) return { ok: false, output: "知识库中没有检索到相关内容" };
      const out = hits
        .map((h) => `[${h.rank}]（${h.docId}，相关度 ${h.score.toFixed(4)}）${h.text}`)
        .join("\n");
      return { ok: true, output: out };
    },
  });
  registry.register({
    name: "calculator",
    description: "计算四则运算表达式，支持 + - * / % 与括号",
    run: (input: string) => {
      try {
        const v = evaluateExpression(input);
        return { ok: true, output: `${input.trim()} = ${v}` };
      } catch (err) {
        return { ok: false, output: `表达式无法求值：${(err as Error).message}` };
      }
    },
  });
  registry.register({
    name: "text_stats",
    description: "统计文本的字符数、非空白字符数与词元数",
    run: (input: string) => ({
      ok: true,
      output: `字符数 ${input.length}，非空白字符数 ${input.replace(/\s/g, "").length}`,
    }),
  });
  registry.register({
    name: "model_info",
    description: "查看当前加载的模型架构与参数量",
    run: () => ({
      ok: true,
      output: `模型 ${String(model.meta.arch.d_model)} 维 / ${String(model.meta.arch.n_layers)} 层 / 词表 ${String(model.meta.arch.vocab_size)}，权重 dtype=${model.meta.dtype}`,
    }),
  });

  const evalRunner = new EvalRunner(embedder, crossEncoder, generator, corpusPath, qaPath);

  const modelInfo: Record<string, unknown> = {
    name: "hyperion-micro",
    format: model.meta.format,
    dtype: model.meta.dtype,
    arch: model.meta.arch,
    tensors: model.meta.n_tensors,
    embedDim: embedder.dim,
    trainedTokens: (model.meta.meta as Record<string, unknown> | undefined)?.trained_tokens ?? null,
    finalLmLoss: (model.meta.meta as Record<string, unknown> | undefined)?.final_lm_loss ?? null,
    indexChunks: rag.size(),
  };

  return {
    assetsDir: dir,
    model,
    tokenizer,
    transformer,
    generator,
    embedder,
    crossEncoder,
    rag,
    registry,
    evalRunner,
    modelInfo,
    runAgentTask: (task: string, maxSteps: number) => runAgent(task, registry, { maxSteps }),
  };
}

export const __all__ = ["bootstrap", "Bootstrap", "BootstrapOptions"];
