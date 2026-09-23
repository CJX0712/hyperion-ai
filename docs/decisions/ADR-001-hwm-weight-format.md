# ADR-001 自定义 HWM v1 权重格式，而不是复用 safetensors / GGUF

状态：已采纳 · 作者：晨星

## 背景
训练内核是 Python、推理内核是 TypeScript 零依赖。需要跨语言权重交换。
safetensors 解析需要第三方库；GGUF 量化生态丰富但格式复杂（元数据 KV、多种量化布局），
在零依赖约束下实现成本高且大部分能力用不到。

## 决策
自定义 32 字节头的 HWM v1：magic + version + header_size + meta_len + CRC32 +
data_offset + file_size，元数据为 JSON，张量数据 64 字节对齐，
int8 权重逐输出通道对称量化、`.scale` 伴随张量命名。

## 后果
- +：TS 侧约 150 行完成加载，零拷贝视图（f32 直接 Float32Array(buf, off)）；
- +：CRC32 与 zlib.crc32 惯例一致，跨语言校验零成本；
- −：不兼容现有生态工具（可接受的代价：1.3M 参数模型无需生态）；
- 布局契约（(out,in) 转置、tok_emb 例外）在 Python 导出与 TS 加载两侧同步维护，SPEC 第 1 节固化。
