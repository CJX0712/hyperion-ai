/**
 * M11 serve - HTTP / SSE 服务层（零第三方依赖）
 *
 * 作者：晨星
 *
 * 只用 node:http，不引入 express / fastify / koa。理由与整个系统的交付红线一致：
 * 运行时第三方依赖数必须为 0，否则"干净环境一键复现"不成立。
 *
 * 提供的接口分两组：
 *   原生接口 /api/v1/*     —— 覆盖 tokenize / embed / generate / rag / agent / eval
 *   OpenAI 兼容 /v1/chat/completions —— 让现成的 OpenAI 客户端可以直接接上来，
 *                                      这是生态复用，不是重复造一套协议
 *
 * 边界与安全：
 *   * 默认只监听 127.0.0.1，不提供鉴权（单机开发者工具定位，明确不暴露公网）
 *   * 令牌桶限流，超限返回 429 而不是静默排队
 *   * 请求体大小上限 8MB，防止单机被自己的调试脚本打爆
 */

import { createServer, type IncomingMessage, type ServerResponse, type Server } from "node:http";
import { readFileSync } from "node:fs";
import { generateStream, type GenerateOptions } from "../infer/index.ts";
import type { Tokenizer } from "../tokenizer/index.ts";
import type { Transformer } from "../arch/index.ts";
import type { RagPipeline } from "../rag/index.ts";
import type { AgentResult } from "../agent/index.ts";
import type { EvalReport } from "../eval/index.ts";

export interface Services {
  tokenizer: Tokenizer;
  transformer: Transformer;
  rag: RagPipeline;
  modelInfo: Record<string, unknown>;
  runAgentTask: (task: string, maxSteps: number) => AgentResult;
  runEval: () => EvalReport;
  consoleHtml: string;
}

export interface ServerHandle {
  server: Server;
  port: number;
  close: () => Promise<void>;
}

const MAX_BODY = 8 * 1024 * 1024;
const RATE_LIMIT = 120; // 每分钟
const RATE_WINDOW_MS = 60_000;

/** 极简令牌桶。单机场景不需要分布式限流。 */
class RateLimiter {
  private hits: number[] = [];
  allow(now = Date.now()): boolean {
    this.hits = this.hits.filter((t) => now - t < RATE_WINDOW_MS);
    if (this.hits.length >= RATE_LIMIT) return false;
    this.hits.push(now);
    return true;
  }
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > MAX_BODY) { reject(new Error("请求体超过 8MB 上限")); req.destroy(); return; }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function json(res: ServerResponse, code: number, body: unknown): void {
  const s = JSON.stringify(body);
  res.writeHead(code, { "content-type": "application/json; charset=utf-8", "content-length": Buffer.byteLength(s) });
  res.end(s);
}

function sseStart(res: ServerResponse): void {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });
}

function sseSend(res: ServerResponse, obj: unknown): void {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

export function createHyperionServer(svc: Services): ServerHandle {
  const limiter = new RateLimiter();
  let handle: ServerHandle;

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    const path = url.pathname;
    const method = req.method ?? "GET";

    if (!limiter.allow()) {
      json(res, 429, { error: { code: "RATE_429", message: "请求过于频繁，请稍后重试" } });
      return;
    }

    try {
      if (method === "GET" && (path === "/" || path === "/console")) {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        res.end(svc.consoleHtml);
        return;
      }

      if (method === "GET" && path === "/api/v1/health") {
        json(res, 200, {
          status: "ok",
          model: svc.modelInfo,
          index: { chunks: svc.rag.size() },
          uptimeSec: Math.round(process.uptime()),
          pid: process.pid,
        });
        return;
      }

      if (method === "GET" && path === "/api/v1/models") {
        json(res, 200, svc.modelInfo);
        return;
      }

      if (method === "POST" && path === "/api/v1/tokenize") {
        const body = JSON.parse(await readBody(req)) as { text?: string };
        const text = body.text ?? "";
        const ids = svc.tokenizer.encode(text);
        json(res, 200, { ids, count: ids.length, roundTrip: svc.tokenizer.decode(ids) });
        return;
      }

      if (method === "POST" && path === "/api/v1/rag/ingest") {
        const body = JSON.parse(await readBody(req)) as { docs?: { docId: string; title?: string; text: string }[] };
        const docs = body.docs ?? [];
        if (!Array.isArray(docs) || docs.length === 0) {
          json(res, 400, { error: { code: "INVALID_INPUT", message: "docs 必须是非空数组" } });
          return;
        }
        const chunks = svc.rag.ingest(docs);
        json(res, 200, { ingestedDocs: docs.length, chunks, totalChunks: svc.rag.size() });
        return;
      }

      if (method === "POST" && path === "/api/v1/rag/query") {
        const body = JSON.parse(await readBody(req)) as { query?: string; useRerank?: boolean };
        if (!body.query || !body.query.trim()) {
          json(res, 400, { error: { code: "EMPTY_QUERY", message: "query 不能为空" } });
          return;
        }
        const span = Date.now();
        const retrieved = svc.rag.retrieve(body.query, 5, body.useRerank !== false);
        json(res, 200, {
          query: body.query,
          hits: retrieved.map((c) => ({ rank: c.rank, chunkId: c.chunkId, docId: c.docId, title: c.title, score: c.score, text: c.text })),
          retrievalMs: Date.now() - span,
        });
        return;
      }

      if (method === "POST" && path === "/api/v1/rag/answer") {
        const body = JSON.parse(await readBody(req)) as { query?: string; useRerank?: boolean };
        if (!body.query || !body.query.trim()) {
          json(res, 400, { error: { code: "EMPTY_QUERY", message: "query 不能为空" } });
          return;
        }
        const ans = svc.rag.ask(body.query, { useRerank: body.useRerank !== false });
        json(res, 200, {
          answer: ans.answer,
          mode: ans.mode,
          latencyMs: ans.latencyMs,
          tokensPerSecond: Number(ans.tokensPerSecond.toFixed(2)),
          citations: ans.citations.map((c) => ({ rank: c.rank, chunkId: c.chunkId, docId: c.docId, title: c.title, score: Number(c.score.toFixed(4)) })),
        });
        return;
      }

      if (method === "POST" && path === "/api/v1/agent/run") {
        const body = JSON.parse(await readBody(req)) as { task?: string; maxSteps?: number };
        if (!body.task || !body.task.trim()) {
          json(res, 400, { error: { code: "EMPTY_TASK", message: "task 不能为空" } });
          return;
        }
        json(res, 200, svc.runAgentTask(body.task, body.maxSteps ?? 3));
        return;
      }

      if (method === "POST" && path === "/api/v1/eval/run") {
        json(res, 200, svc.runEval());
        return;
      }

      if (method === "POST" && (path === "/api/v1/generate" || path === "/v1/chat/completions")) {
        const raw = await readBody(req);
        const body = JSON.parse(raw) as {
          prompt?: string;
          messages?: { role: string; content: string }[];
          stream?: boolean;
          max_tokens?: number;
          temperature?: number;
          top_p?: number;
          seed?: number;
        };
        const prompt = body.prompt ?? (body.messages ?? []).map((m) => `${m.role}：${m.content}`).join("\n");
        if (!prompt || !prompt.trim()) {
          json(res, 400, { error: { code: "EMPTY_PROMPT", message: "prompt / messages 不能为空" } });
          return;
        }
        const opts: GenerateOptions = {
          maxTokens: body.max_tokens ?? 96,
          temperature: body.temperature ?? 0.8,
          topP: body.top_p ?? 0.9,
          topK: 40,
          seed: body.seed ?? 20260923,
        };
        const openai = path === "/v1/chat/completions";
        const id = `chatcmpl-${Date.now().toString(36)}`;
        const created = Math.floor(Date.now() / 1000);

        if (body.stream === false) {
          let text = "";
          for await (const ev of generateStream(svc.transformer, svc.tokenizer, prompt, opts)) text += ev.delta;
          if (openai) {
            json(res, 200, {
              id, object: "chat.completion", created, model: String(svc.modelInfo.name ?? "hyperion-micro"),
              choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
            });
          } else {
            json(res, 200, { text });
          }
          return;
        }

        sseStart(res);
        let tokens = 0;
        const t0 = Date.now();
        for await (const ev of generateStream(svc.transformer, svc.tokenizer, prompt, opts)) {
          tokens++;
          if (openai) {
            sseSend(res, {
              id, object: "chat.completion.chunk", created,
              model: String(svc.modelInfo.name ?? "hyperion-micro"),
              choices: [{ index: 0, delta: { content: ev.delta }, finish_reason: null }],
            });
          } else {
            sseSend(res, { type: "token", index: ev.index, delta: ev.delta });
          }
        }
        if (openai) {
          sseSend(res, { id, object: "chat.completion.chunk", created, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] });
          res.write("data: [DONE]\n\n");
        } else {
          const secs = (Date.now() - t0) / 1000;
          sseSend(res, { type: "done", tokens, seconds: Number(secs.toFixed(3)), tokensPerSecond: Number((tokens / Math.max(secs, 1e-6)).toFixed(2)) });
        }
        res.end();
        return;
      }

      json(res, 404, { error: { code: "NOT_FOUND", message: `没有这个路由：${method} ${path}` } });
    } catch (err) {
      const msg = (err as Error).message;
      if (!res.headersSent) json(res, 500, { error: { code: "INTERNAL", message: msg } });
      else res.end();
    }
  });

  handle = {
    server,
    port: 0,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
  return handle;
}

export function loadConsoleHtml(path: string): string {
  return readFileSync(path, "utf8");
}

export const __all__ = ["createHyperionServer", "loadConsoleHtml", "Services"];
