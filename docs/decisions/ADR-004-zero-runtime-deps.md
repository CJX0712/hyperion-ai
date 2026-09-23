# ADR-004 推理内核运行期零第三方依赖

状态：已采纳 · 作者：晨星

## 背景
目标环境包含离线/受限网络机器（npm 镜像被改写、外网不稳）。
推理是常驻服务，依赖供应链风险 = 可用性风险。

## 决策
推理/编排内核只用 Node 内置模块（http/fs/path/url）。
数学核（GEMM、RoPE、softmax、量化、cosine）全部手写并带不变量测试；
`package.json` 禁止出现 `dependencies` 字段（scan:p0 机械检查）。
开发依赖（typescript / @types/node）仅构建期使用。

## 后果
- +：`npm start` 不需要任何运行期安装；审计面 = 0；
- +：性能完全可控（SimSIMD 式 8 路展开 cosine、m-outer/k-inner 解码 GEMM）；
- −：不能白嫖成熟库的边界处理，由 SPEC S1-S29 与自检补偿。
