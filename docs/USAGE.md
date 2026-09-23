# 使用指南

作者：晨星

## 三个入口

1. **控制台（推荐）**：`npm start` 后打开 `http://127.0.0.1:8787`，左侧五个视图：
   - **对话问答**：直接提问，回答带编号引用与来源文档，速度与延迟实时显示；
   - **知识库检索**：只看检索结果（每条带相关度分数），用于调参与核对召回；
   - **工具与追踪**：看 Agent 的完整执行轨迹（计划/执行/观测/批判四类事件）；
   - **评估**：一键跑全量 QA 评估，含带重排/无重排对照与不变量清单；
   - **健康**：模型架构、索引分片、量化 dtype、运行时长。
2. **HTTP API**：见 `docs/API.md`，含 SSE 流式与 OpenAI 兼容 `/v1/chat/completions`。
3. **Node 代码**：`import { bootstrap } from "./bootstrap.ts"` 后拿到完整对象图
   （generator / rag / registry / evalRunner），适合嵌入到已有 Node 服务。

## 换成自己的知识库

```js
// 任意 Node 进程里
const app = bootstrap({ assetsDir: "assets", ingestCorpus: false });
app.rag.ingest([
  { docId: "runbook-001", title: "重启流程", text: "你的文档正文，支持中英混排……" },
]);
const hits = app.rag.retrieve("怎么重启", 5, true);   // 检索
const ans  = app.rag.ask("怎么重启");                 // 带引用的问答
```

文档切分规则：按句子边界打包（中文句号/问号/叹号/分号与换行），块间带重叠，`chunkId` 形如 `docId#3`。重复 `docId` 会覆盖旧版本。

## Agent 工具扩展

```js
app.registry.register({
  name: "weather",                    // 工具名（路由器按关键词匹配）
  description: "查询城市天气",
  run: (input) => ({ ok: true, output: `${input}：晴，26 度` }),
});
```

注意：内置路由器是确定性规则匹配（关键词 -> 工具），不是让 1.3M 参数的模型自己发 JSON。要接入自定义路由逻辑，实现 `Planner` 接口并传给 `runAgent`。

## 评估怎么读

- **recall@k / MRR**（带重排 vs 无重排）：检索质量主指标；两列对照即重排增益；
- **不变量清单**：任何一条 FAIL 都表示系统性质被破坏（如重排护栏失效），必须修复而不是调参绕过；
- **忠实度（grounded）**：答案 4-gram 在引用片段中的覆盖率，低说明模型在"编"而不是"抄"知识库；
- **抽取式占比**：生成质量不足时的诚实降级比例，越低越好。

## 确定性与可复现

- 全部随机路径使用 mulberry32（TS）/ default_rng（Python），种子 20260923；
- 同一份数据 + 同一代码版本，训练产物与推理输出逐位一致；
- e2e 每次随机端口启动新服务器，不依赖运行期残留状态。
