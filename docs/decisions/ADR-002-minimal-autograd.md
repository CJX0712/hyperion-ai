# ADR-002 自研极简 autograd（约 300 行 numpy），不引入 PyTorch

状态：已采纳 · 作者：晨星

## 背景
1.3M 参数、三阶段训练、CPU only。PyTorch 安装包 2GB+，且本沙箱曾出现内存 commit 紧张。
需求只有：matmul/softmax/cross_entropy/rmsnorm/embedding 的前向与反向。

## 决策
自研 Tensor 类（backward 闭包风格），仅实现训练所需算子。

## 后果
- +：依赖只剩 numpy，全环境（含离线沙箱）可复现；
- +：性能优化直接可控（见 ADR-003）；
- −：无 checkpoint / 分布式 / 混合精度等高级能力（本规模用不到）；
- −：算子少写一个反向都会错，因此用「数值梯度 vs 解析梯度」的梯度检验守护（rel err ~1e-7 量级）。
