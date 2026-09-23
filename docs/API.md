# HTTP API

作者：晨星

基础地址：`http://127.0.0.1:8787`（默认只绑回环地址，不暴露公网）。
限流：令牌桶 120 次/分钟，超限返回 429。请求体上限 8MB。

## 服务

### GET /api/v1/health
```json
{ "status": "ok", "model": { "...": "模型信息" }, "index": { "chunks": 312 },
  "uptimeSec": 42, "pid": 12345 }
```

### GET /api/v1/models
返回模型元信息（架构、参数量、dtype、嵌入维度、索引分片数）。

### POST /api/v1/tokenize
`{ "text": "你好" }` → `{ "ids": [..], "count": 2, "roundTrip": "你好" }`

## 检索与问答

### POST /api/v1/rag/ingest
`{ "docs": [{ "docId": "d1", "title": "标题", "text": "正文" }] }`
→ `{ "ingestedDocs": 1, "chunks": 3, "totalChunks": 315 }`
重复 docId 覆盖旧文档。空 docs 返回 400。

### POST /api/v1/rag/query
`{ "query": "日志级别怎么设置", "useRerank": true }`
→ `{ "query", "hits": [{ "rank", "chunkId", "docId", "title", "score", "text" }], "retrievalMs" }`
topK=5，混合检索（稠密+BM25+RRF+重排护栏）。

### POST /api/v1/rag/answer
`{ "query": "...", "useRerank": true }`
→ `{ "answer", "mode": "generative"|"extractive", "latencyMs", "tokensPerSecond", "citations": [...] }`
检索无命中时明确返回"未检索到相关内容"，不硬凑答案。

## Agent

### POST /api/v1/agent/run
`{ "task": "计算 12*3", "maxSteps": 3 }`
→ `{ "answer", "events": [{ "type": "plan|step|observation|critique|final", ... }], "steps", "toolCalls", "stoppedBy" }`
内置工具：`search_kb`、`calculator`、`text_stats`、`model_info`。

## 评估

### POST /api/v1/eval/run
每次调用在**全新索引**上跑全量 QA：
→ `{ "metrics": { recallAt1/3/5, mrr, 带重排/无重排两套, retrievalP50Ms, retrievalP95Ms },
    "invariants": [{ "name", "pass", "detail" }], "generative": { cases, nonEmpty, grounded, ... } }`

## 生成

### POST /api/v1/generate
`{ "prompt": "...", "stream": true, "max_tokens": 96, "temperature": 0.8 }`
- `stream: true`：SSE，`data: {"type":"token","index":n,"delta":"..."}`，收尾 `{"type":"done","tokens":n,"tokensPerSecond":x}`
- `stream: false`：`{ "text": "..." }`

### POST /v1/chat/completions（OpenAI 兼容）
`{ "messages": [{ "role": "user", "content": "..." }], "stream": false }`
非流式返回标准 `chat.completion`；流式返回 `chat.completion.chunk` 序列 + `data: [DONE]`。

## 错误格式

统一 `{ "error": { "code": "...", "message": "..." } }`，已知码：
`RATE_429`、`INVALID_INPUT`、`EMPTY_QUERY`、`EMPTY_TASK`、`EMPTY_PROMPT`、`NOT_FOUND`、`INTERNAL`。
