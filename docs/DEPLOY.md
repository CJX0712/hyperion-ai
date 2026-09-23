# 部署指南

作者：晨星

## 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| OS | Windows 10/11、Linux、macOS | 开发与验证均在 Windows 11 / 16GB RAM / 无 GPU 完成 |
| Node | >= 20.10 | 需要 `--experimental-transform-types`（22.x 推荐） |
| Python | >= 3.11 + numpy >= 2.0 | 仅训练阶段需要；推理服务不需要 Python |
| 编译器 | 不需要 | 全栈无原生编译依赖 |
| 网络 | 仅 `npm install` 时需要 | 训练与服务完全离线可跑 |

## 标准部署（源码方式）

```bash
git clone https://github.com/CJX0712/hyperion-ai.git
cd hyperion-ai

# 1) Python 虚拟环境（训练内核）
python -m venv venv
venv/Scripts/pip install -r requirements.txt      # Windows
# venv/bin/pip install -r requirements.txt        # Linux / macOS

# 2) TS 构建依赖（精确版本锁定）
npm install                                       # 运行期不加载任何 npm 包

# 3) 训练（约 12 分钟 CPU；确定性种子 20260923）
npm run train
# 产出：assets/model.i8.hwm / model.f32.hwm / tokenizer.json /
#       corpus.jsonl / qa.jsonl / train-report.json

# 4) 全量验收（构建 + 自检 + e2e + P0 扫描）
npm run verify

# 5) 启动服务
npm start                                         # http://127.0.0.1:8787
```

### 训练命令的位置参数

`npm run train -- <参数>` 透传给 `hyperion.train`，常用：

- `--lm-steps 900 --emb-steps 150 --ce-steps 150`：三阶段步数
- `--batch 8 --seq 96`：LM 批量/序列长（内存标定见 ARCHITECTURE.md 第 7 节）
- `--d-model 128 --layers 4`：模型规模（改大前先确认内存）
- 内存充裕（>=24GB 可用）可加 `--batch 16 --ce-groups 6`

## Docker 部署

```bash
docker build -t hyperion-ai .
docker run -p 8787:8787 -v hyperion-assets:/app/assets hyperion-ai
```

镜像内分两个阶段：`train` 目标完成训练，`runtime` 目标只跑 Node 服务。
权重通过 volume 持久化；挂入已训练的 `assets/` 可跳过训练。

## 配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 8787 | 监听端口 |
| `HOST` | 127.0.0.1 | 监听地址；单机工具默认不暴露公网 |
| `HYPERION_ASSETS` | assets | 产物目录 |
| `HYPERION_CONSOLE` | web/console.html | 控制台文件路径 |
| `HYPERION_PYTHON` | 自动探测 | 训练用解释器（venv > PATH） |

## 验收标准（部署后必做）

`npm run verify` 必须 0 退出码，包含：
1. `tsc` 严格模式编译零错误；
2. 自检 29 项不变量全 PASS（含 KV Cache 等价、重排护栏、CRC32）；
3. e2e 十步用户旅程全 PASS（含 SSE 流式与 OpenAI 兼容协议）;
4. P0 扫描：无 emoji 图标、无紫粉渐变、运行期依赖 0、控制台零外部请求。

## 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 训练 OOM（`_ArrayMemoryError`） | commit 耗尽（WSL/Docker/浏览器挤占） | 保持默认 batch=8；关停大内存进程；见 ARCHITECTURE.md 第 7 节 |
| `缺少必需产物` | 未训练或 HYPERION_ASSETS 指错 | 先 `npm run train`，核对 assets 路径 |
| CRC32 校验失败 | 权重文件损坏/截断 | 重跑 `npm run train` |
| 启动即退出 | Node < 20.10 | 升级 Node |
| 429 Too Many Requests | 超过 120 次/分钟 | 客户端退避重试 |
