#!/usr/bin/env node
/**
 * 端到端验收 - 启动真实服务器，按用户旅程打全部接口
 *
 * 作者：晨星
 *
 * 前置：npm run build 已完成（dist/main.js 存在），assets/ 有训练产物。
 * 流程：随机端口启动 -> health -> tokenize -> ingest -> query -> answer ->
 *       agent -> SSE 流式生成 -> OpenAI 兼容接口 -> 错误流（404/400/空 docs）
 * 退出码：0 全过；1 有失败。
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const distMain = join(root, "dist", "main.js");
if (!existsSync(distMain)) {
  console.error("[e2e] 缺少 dist/main.js —— 请先运行 npm run build");
  process.exit(1);
}
if (!existsSync(join(root, "assets", "model.i8.hwm"))) {
  console.error("[e2e] 缺少 assets/model.i8.hwm —— 请先运行 npm run train");
  process.exit(1);
}

const results = [];
function check(name, cond, detail = "") {
  results.push({ name, ok: !!cond, detail });
  console.log(`  [${cond ? "PASS" : "FAIL"}] ${name}${detail ? "  " + detail : ""}`);
}

const PORT = 18000 + Math.floor(Math.random() * 20000);
const BASE = `http://127.0.0.1:${PORT}`;

const child = spawn(process.execPath, [distMain], {
  cwd: root,
  env: { ...process.env, PORT: String(PORT), HOST: "127.0.0.1", HYPERION_ASSETS: join(root, "assets") },
  stdio: ["ignore", "pipe", "pipe"],
});
let serverLog = "";
child.stdout.on("data", (d) => (serverLog += d));
child.stderr.on("data", (d) => (serverLog += d));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitHealthy(timeoutMs = 30000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(`${BASE}/api/v1/health`);
      if (r.ok) return true;
    } catch { /* 还没起来 */ }
    await sleep(300);
  }
  return false;
}

function post(path, body) {
  return fetch(BASE + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

try {
  const up = await waitHealthy();
  check("服务器在 30s 内就绪", up, `port=${PORT}`);
  if (!up) throw new Error("服务器未能启动");

  // 1. health
  {
    const h = await (await fetch(`${BASE}/api/v1/health`)).json();
    check("health：status=ok 且索引非空", h.status === "ok" && h.index.chunks > 0, `chunks=${h.index.chunks} pid=${h.pid}`);
  }

  // 2. tokenize 往返
  {
    const r = await (await post("/api/v1/tokenize", { text: "端到端验收 testing" })).json();
    check("tokenize：ids 非空且往返一致", Array.isArray(r.ids) && r.ids.length > 0 && r.roundTrip === "端到端验收 testing", `${r.count} tokens`);
  }

  // 3. ingest 新文档（含唯一 ASCII 词，BM25 词法精确匹配保证可被检索到）
  {
    const r = await (await post("/api/v1/rag/ingest", {
      docs: [{ docId: "e2e-1", title: "E2E", text: "hyperion e2e marker 端到端验收专用文档：超算中心的巡检时间为每周二上午十点。" }],
    })).json();
    check("ingest：分片入库", r.chunks > 0 && r.totalChunks > 0, `chunks=${r.chunks}`);
  }

  // 4. query 检索
  {
    const r = await (await post("/api/v1/rag/query", { query: "hyperion e2e marker" })).json();
    check("query：ingest 的文档出现在检索结果中", r.hits.length > 0 && r.hits.some((h) => h.docId === "e2e-1"), `top1=${r.hits[0]?.docId} hits=${r.hits.length} ${r.retrievalMs}ms`);
  }

  // 5. answer 问答
  {
    const r = await (await post("/api/v1/rag/answer", { query: "巡检时间是什么时候", max_tokens: 24 })).json();
    check("answer：有答案与引用", typeof r.answer === "string" && r.answer.length > 0 && r.citations.length > 0, `mode=${r.mode} ${r.tokensPerSecond} tok/s`);
  }

  // 6. agent
  {
    const r = await (await post("/api/v1/agent/run", { task: "计算 21*2" })).json();
    check("agent：calculator 路径", r.answer.includes("42"), `steps=${r.steps}`);
  }

  // 7. SSE 流式生成
  {
    const resp = await post("/api/v1/generate", { prompt: "知识库检索", stream: true, max_tokens: 12 });
    const text = await resp.text();
    const lines = text.split("\n").filter((l) => l.startsWith("data: "));
    const last = JSON.parse(lines[lines.length - 1].slice(6));
    check("SSE 流式：token 事件 + done 收尾", resp.status === 200 && last.type === "done" && last.tokens > 0, `${last.tokens} tokens @ ${last.tokensPerSecond} tok/s`);
  }

  // 8. OpenAI 兼容（非流式 + 流式）
  {
    const r = await (await post("/v1/chat/completions", { messages: [{ role: "user", content: "你好" }], stream: false, max_tokens: 8 })).json();
    check("OpenAI 兼容：chat.completion 结构", r.object === "chat.completion" && r.choices?.[0]?.message?.content !== undefined, `finish=${r.choices?.[0]?.finish_reason}`);

    const resp = await post("/v1/chat/completions", { messages: [{ role: "user", content: "你好" }], stream: true, max_tokens: 8 });
    const text = await resp.text();
    check("OpenAI 兼容：SSE 含 [DONE]", text.includes("chat.completion.chunk") && text.includes("[DONE]"), "");
  }

  // 9. 错误流
  {
    const r404 = await fetch(`${BASE}/no/such/route`);
    check("未知路由 -> 404", r404.status === 404, "");

    const r400 = await post("/api/v1/rag/query", { query: "" });
    check("空 query -> 400 EMPTY_QUERY", r400.status === 400, "");

    const r400b = await post("/api/v1/rag/ingest", { docs: [] });
    check("空 docs -> 400 INVALID_INPUT", r400b.status === 400, "");
  }
} catch (err) {
  check("流程异常中断", false, (err instanceof Error ? err.message : String(err)).slice(0, 200));
} finally {
  child.kill("SIGTERM");
  await sleep(500);
  if (!child.killed) child.kill("SIGKILL");
}

const fails = results.filter((r) => !r.ok);
console.log(`\ne2e 结果：${results.length - fails.length} PASS / ${fails.length} FAIL`);
if (fails.length === 0) process.exit(0);
console.log("服务器日志（末尾 40 行）：");
console.log(serverLog.split("\n").slice(-40).join("\n"));
process.exit(1);
