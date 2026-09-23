# ADR-003 numpy 3D matmul 必须先展平为 2D

状态：已采纳（实测驱动） · 作者：晨星

## 背景
初始训练 29.3 s/步。cProfile 显示 matmul 占主导；微基准测得：

| 路径 | 吞吐 |
|---|---|
| numpy 3D × 2D（(B,T,D) @ (D,D)） | 3.35 GFLOP/s |
| numpy 展平 2D × 2D | 27.5 GFLOP/s |
| JS 手写 GEMM | 1.35 GFLOP/s（说明训练必须留在 numpy 侧） |

## 决策
- `linear()` / LM head / 注意力 k 转置在进入 matmul 前 reshape 为 2D；
- matmul 反向在**前向时**缓存连续转置（backward 里做 ascontiguousarray 代价相同）；
- 梯度累加用原地 `+=`。

## 后果
步时 29.3s → 1.21s（24 倍）。该优化对使用者不可见，但任何绕过 `linear()` 直接写
`matmul(3D, 2D)` 的新代码都会慢 8 倍——在 engine.matmul 的文档注释中标注此红线。
