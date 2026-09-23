# hyperion-ai

**双引擎端到端可复现 AI 系统**：Python(numpy) 训练内核 + TypeScript 零运行期依赖推理编排内核。

作者：晨星（CJX0712） · License: MIT

## 这是什么

一套在 **16GB 内存、无 GPU、无编译器** 的普通 PC 上，从语料到服务**全链路可运行、可复现、可验收**的完整 AI 系统：

```
Python 训练内核（numpy）                TypeScript 推理编排内核（0 依赖）
┌─────────────────────────┐   HWM v1   ┌──────────────────────────────┐
│ BPE 分词器               │  int8/f32  │ 张量核（手写 GEMM/量化/RoPE）   │
│ Transformer LM           │  +CRC32 ─→ │ Transformer 推理（KV Cache）  │
│ 双编码器嵌入（InfoNCE）    │  64B 对齐  │ 混合检索 BM25+稠密+RRF+重排护栏 │
│ 交叉编码器重排器          │            │ RAG / 有界 Agent / HTTP+SSE   │
└─────────────────────────┘            └──────────────────────────────┘
```

设计原则：**复用业界成熟结构（RMSNorm/RoPE/SwiGLU/GQA/BPE/RRF），不自研轮子；每个模块单一职责、有机械可验证的不变量；运行期第三方依赖为 0。**

## 一键复现

前置：Node >= 20.10（需 `--experimental-transform-types`）、Python >= 3.11 + numpy。

```bash
# 1. Python 环境（任意含 numpy 的解释器均可，或项目内 venv）
python -m venv venv && venv/Scripts/pip install -r requirements.txt   # Windows
# 2. 安装 TS 开发依赖（仅 typescript 与 @types/node，运行期不需要）
npm install
# 3. 训练（约 12 分钟，CPU）——产出 assets/model.i8.hwm 等全部产物
npm run train
# 4. 全量验收：构建 -> 自检 -> e2e -> P0 扫描
npm run verify
# 5. 启动服务（默认 http://127.0.0.1:8787）
npm start
```

## 交付物

| 交付物 | 位置 |
|---|---|
| 完整源码 | `src/py/hyperion/`（训练内核）、`src/ts/`（推理编排内核） |
| 训练产物 | `assets/model.i8.hwm`（int8 权重）、`model.f32.hwm`、`tokenizer.json`、`corpus.jsonl`、`qa.jsonl`、`train-report.json` |
| 版本锁定 | `package.json`（devDependencies 精确版本）、`requirements.txt`（numpy 区间锁定） |
| 文档 | `docs/`：架构、规格、API、部署、使用指南、ADR 决策记录 |
| 质量门 | `npm run verify`：自检 29 项不变量 + 10 步 e2e + P0 设计红线扫描 |

## 自检与验收（全部机械可验证）

- 张量核：matmul 与朴素参考逐元素一致、int8 量化误差上界、cosine 自相似=1
- HWM：CRC32 惯例值、magic/版本/大小校验、scale 绑定
- 推理：KV Cache 的 prefill 与逐 token 路径等价、贪心解码确定性
- 检索：BM25/稠密/RRF 确定性、**重排护栏（重排后 top1 不差于重排前）**
- RAG/Agent：ingest→retrieve→ask 链路、有界循环封顶 8 步、非法输入拒绝
- e2e：真实服务器跑通 health/tokenize/ingest/query/answer/agent/SSE/OpenAI 兼容/错误流
