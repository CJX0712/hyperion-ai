# ADR-009 内存标定：autograd 保留全部激活，默认值按 16GB 机器标定

状态：已采纳 · 作者：晨星

## 背景
自研 autograd 不做激活检查点，反向传播内存峰值 ≈ batch × seq × d_ff × 层数 × 常数。
本机 16GB，与 Docker/WSL/浏览器共存时可用 commit 约 6GB。三次实测 OOM：
- 首次：LM batch 32 / seq 128 下 `_ArrayMemoryError`（10.0 MiB 分配失败）；
- 第二次：CE 6 组 × 4 序列 × L=193 下 2.26 MiB 分配失败
  ——哪怕物理空闲 2.7GB，**commit（页面文件预算）才是真正约束**；
- 第三次：CE 已降到 2 组，注意力分数 (8,4,193,193)=4.55MiB 仍分配失败
  ——长进程后期 Python 堆保留 LM 阶段的 commit 水位，CE 峰值必须另算。

## 决策
- LM/BE：`batch*seq*d_ff ≈ 4e5 元素`（batch=8、seq=96）；
- CE：注意力分数峰值压到约 1.9MB —— groups=2、q_len=24、doc_len=96（L=121）；
- BLAS 线程数默认 2（小矩阵多线程更慢且抬高内存峰值）。

## 后果
- +：同类机器 `npm run train` 一次通过，无需手工调参；
- −：CE 每步组数变少，训练步数等效减半（由步数与学习率补偿）；
- 大内存机器可通过 CLI 覆盖（DEPLOY.md）。
