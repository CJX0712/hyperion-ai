# 系统架构

作者：晨星

## 1. 双引擎与权重边界

系统被一条**二进制权重边界**切成两个引擎，边界即契约：

- **训练内核（Python + numpy）**：负责一切需要梯度与 BLAS 的工作。自研极简 autograd（Tensor + backward 闭包），只用 numpy 一个第三方库。
- **推理编排内核（TypeScript，Node 内置模块）**：负责一切需要常驻服务的工作。只用 `node:http`、`node:fs`、`node:path` 等 builtin，运行期第三方依赖为 0。
- 两者通过 **HWM v1 权重格式**（见 ADR-001）解耦：Python 导出、TS 加载，CRC32 完整性校验，谁也不依赖谁的运行时。

## 2. 13 个模块与调用方向

依赖严格单向向下，`bootstrap.ts` 是唯一允许跨层引用的装配点：

```
tensor(张量核/HWM) ← tokenizer(BPE)
        ↑
      arch(Transformer)
        ↑
infer(采样/生成/KV)   embed(双编码器/交叉编码器)
        ↑                   ↑
      index(BM25/稠密/RRF 混合) → rerank(重排护栏)
        ↑
      rag(分块/检索/生成/引用)
        ↑
 agent(有界工具循环) · serve(HTTP/SSE/OpenAI 兼容) · eval(验收指标)
        ↑
    bootstrap(装配) → main(入口)
```

每个模块单一职责：`tensor` 只有数学，`index` 只有检索，`serve` 只有协议。模块可独立验证（`npm run test` 按模块跑不变量），也可组合成完整链路（`npm run e2e`）。

## 3. 训练三阶段（确定性，seed=20260923）

1. **LM**：字节级 BPE（词表 2048）→ Transformer（RMSNorm+RoPE+SwiGLU+GQA+tied embedding，约 1.31M 参数）→ 交叉熵，AdamW + cosine 调度 + 梯度裁剪。
2. **双编码器**：均值池化 → `emb_proj` → L2 归一 → InfoNCE 温度 0.07。
3. **交叉编码器**：`query <sep> doc` → 均值池化 → `ce_score` 标量头 → 组内 softmax。

全部基于确定性合成语料（101 篇知识库文档 + 404 条 QA），无外部数据依赖，任何机器重训结果逐位一致。

## 4. 关键性能决策（无 GPU、无编译器约束下的实测）

| 决策 | 依据（本机实测） |
|---|---|
| Python 侧 3D 张量先 reshape 成 2D 再 matmul | numpy 3D×2D 路径 3.35 GFLOP/s，展平 2D 27.5 GFLOP/s，**8 倍差距** |
| matmul 反向缓存连续转置、原地累加梯度 | 训练步时 29.3s → 1.21s（24 倍） |
| TS 侧权重统一转置为 (out,in) 存储（`tok_emb` 除外） | 解码热路径只需 Aᵀ·b 形式的矩阵-向量乘（matmulT1） |
| int8 逐输出通道对称量化 + `.scale` 伴随张量 | 1.31M 参数权重约 1.1MB，内存映射零拷贝加载 |
| BLAS 线程数默认 2 | 小矩阵多线程反而慢，且降低内存 commit 压力 |

## 5. 检索管线与融合护栏

```
query ─┬─ 稠密召回（嵌入余弦，brute-force）
       ├─ 稀疏召回（BM25，k1=1.2 b=0.75，CJK 单字+二元组）
       └─→ RRF 融合（k=60）→ 重排器（交叉编码器）
                                  ↓
                     融合护栏：rrfFuse(重排结果, 原融合)  ← 不变量：重排后 top1 不差于重排前
```

护栏的动机（ADR-006）：1M 参数交叉编码器并不总是比混合融合更准，但用户要的是「重排**可以没有增益，但绝不能变差**」。护栏把重排从「替换排序」降级为「提供一路 RRF 信号」，从而把该性质变成机械可验证的不变量。

## 6. Agent 的诚实设计

1.3M 参数的 LM 无法可靠输出结构化工具调用（JSON 解析失败率高），因此 Agent 使用**确定性规则路由器**（计算器→知识库检索→文本统计），LLM 只在生成最终回答时参与。循环有界：maxSteps 钳制在 [1,8]，工具失败不中断循环，最终兜底话术明确承认失败（ADR-008）。

## 7. 内存标定（为什么会 OOM、怎么根治）

自研 autograd 保留全部中间激活，反向传播时内存峰值 ≈ `batch × seq × d_ff × 层数 × 常数`。16GB Windows 主机（Docker/WSL/浏览器共存，可用 commit 约 6GB）实测标定：LM/BE 按 `batch*seq*d_ff ≈ 4e5 元素`（batch=8、seq=96）；交叉编码器的注意力分数 `(G·K, H, L, L)` 全部进激活图，L=q_len+1+doc_len=24+1+96=121 时峰值约 1.9MB。这组默认值（batch=8、ce-groups=2、q-len=24、doc-len=96）保证同类机器 `npm run train` 一次通过；三次 OOM 复盘见 ADR-009。
