# 规格与不变量（SPEC）

作者：晨星

本文列出系统的**机械可验证规格**。每条不变量都能用独立实现交叉验证，
对应测试位于 `src/ts/selfcheck.ts`，验收门为 `npm run verify`（0 退出码）。

## 1. 权重格式 HWM v1

| 字段 | 规格 |
|---|---|
| magic | `HWM1`（0x48 0x57 0x4D 0x31） |
| version | u16 = 1 |
| header_size | u16 = 32 |
| meta_len / crc32 | u32；CRC32(IEEE 802.3) 覆盖 header 之后全部字节 |
| data_offset / file_size | u64；file_size 必须等于实际字节数 |
| 张量数据 | 64 字节对齐；int8 逐输出通道对称量化，`.scale` 伴随张量 |
| 布局契约 | 2D 权重导出为 (out,in)，唯一例外 `tok_emb`（查表布局） |

不变量：
- S1 `crc32("123456789") == 0xCBF43926`（与 Python zlib 一致）；
- S2 加载后每个 int8 张量的 `.scale` 非空；
- S3 CRC 或 file_size 不符必须抛错（早失败，不允许静默降级）。

## 2. 张量核

- S4 `matmulATB/T1/NT/NN` 与朴素参考实现逐元素一致（容差 1e-4）；
- S5 int8 路径与 f32 路径误差 <= 0.05 + 5%|ref|；
- S6 对称量化每行最大误差 <= max|w|/127；
- S7 `cosine(v,v)=1`（容差 1e-6）、正交向量=0、softmax 和=1；
- S8 mulberry32 同种子逐位一致。

## 3. 分词器

- S9 encode/decode 对任意 UTF-8（中英混排、emoji 除外）往返无损；
- S10 特殊 id（pad/bos/eos/unk/sep）解码时被跳过；
- S11 TS 实现与 Python 实现使用同一份 tokenizer.json，预分词规则逐字符一致。

## 4. 推理

- S12 KV Cache 一致性：`prefill(ids)` 与逐 `step(id,pos)` 的 logits 最大差 < 1e-3；
- S13 贪心解码（temperature=0）同输入同输出，与采样种子无关；
- S14 生成结束必重置 KV Cache（请求间零串扰）。

## 5. 检索与重排

- S15 BM25 分数降序，`compareHits` 同分按 id 字典序（全链路确定性）；
- S16 RRF 融合次序可复现（同输入同输出）；
- S17 **重排护栏**：`rerank()` 输出的 top1 相关度 >= 无重排融合的 top1（对抗性打分器下仍成立）；
- S18 检索空结果必须显式返回"未检索到相关内容"。

## 6. RAG

- S19 chunkId 形如 `docId#ord`，块长受控（chunkSize + 句子上溢）；
- S20 ingest 重复 docId 覆盖旧文档；
- S21 生成回答 < 4 字符时降级为抽取式并标记 `mode=extractive`。

## 7. Agent

- S22 maxSteps 钳制 [1,8]；
- S23 工具抛异常/返回失败不中断循环，最终有兜底 answer；
- S24 计算器为 shunting-yard 实现，拒绝一切非算术输入（绝不 eval）。

## 8. 服务

- S25 运行期第三方依赖 = 0（package.json 无 dependencies 字段）；
- S26 默认只绑 127.0.0.1；限流 120/min；请求体上限 8MB；
- S27 OpenAI 兼容流式以 `data: [DONE]` 收尾；
- S28 错误响应统一 `{error:{code,message}}` 结构；
- S29 e2e 用户旅程（health/tokenize/ingest/query/answer/agent/SSE/OpenAI/错误流）全通。

## 9. 设计红线（P0，`npm run scan:p0` 强制）

- R1 图标只用 SVG sprite，禁止 emoji；
- R2 禁止紫粉渐变；主色 #0D6E7E / 强调 #0E7490；
- R3 控制台单文件、零外部请求（离线可用）；
- R4 无占位文案。
