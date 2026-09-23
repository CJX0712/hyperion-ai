/**
 * 服务入口 - 只做装配与启动，零业务逻辑
 *
 * 作者：晨星
 *
 * 环境变量：
 *   PORT               监听端口，默认 8787
 *   HOST               监听地址，默认 127.0.0.1（单机工具，不暴露公网）
 *   HYPERION_ASSETS    产物目录，默认 assets
 *   HYPERION_CONSOLE   控制台 HTML 路径，默认 web/console.html
 */

import { existsSync } from "node:fs";
import { bootstrap } from "./bootstrap.ts";
import { createHyperionServer, loadConsoleHtml } from "./serve/server.ts";

const assetsDir = process.env.HYPERION_ASSETS ?? "assets";
const port = Number(process.env.PORT ?? 8787);
const host = process.env.HOST ?? "127.0.0.1";
const consolePath = process.env.HYPERION_CONSOLE ?? "web/console.html";

const t0 = Date.now();
const app = bootstrap({ assetsDir, ingestCorpus: true });
const loadMs = Date.now() - t0;

const consoleHtml = existsSync(consolePath)
  ? loadConsoleHtml(consolePath)
  : "<!doctype html><meta charset='utf-8'><title>Hyperion AI</title><p>控制台文件缺失：web/console.html</p>";

const handle = createHyperionServer({
  tokenizer: app.tokenizer,
  transformer: app.transformer,
  rag: app.rag,
  modelInfo: app.modelInfo,
  runAgentTask: app.runAgentTask,
  runEval: () => app.evalRunner.run({ generativeCases: 12 }),
  consoleHtml,
});

handle.server.listen(port, host, () => {
  const arch = app.model.meta.arch;
  console.log("Hyperion AI 服务已启动");
  console.log(`  地址        http://${host}:${port}`);
  console.log(`  权重        ${assetsDir}/model.i8.hwm（dtype=${app.model.meta.dtype}，张量 ${app.model.meta.n_tensors} 个）`);
  console.log(`  架构        d_model=${String(arch.d_model)} layers=${String(arch.n_layers)} heads=${String(arch.n_heads)} vocab=${String(arch.vocab_size)} max_seq=${String(arch.max_seq)}`);
  console.log(`  嵌入维度    ${app.embedder.dim}`);
  console.log(`  索引分片    ${app.rag.size()} 条`);
  console.log(`  注册工具    ${app.registry.names().join(", ")}`);
  console.log(`  加载耗时    ${loadMs} ms`);
  console.log("  运行期第三方依赖数  0");
});

const shutdown = () => {
  console.log("\n正在关闭服务……");
  handle.close().then(() => process.exit(0));
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
