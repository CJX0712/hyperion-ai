# syntax=docker/dockerfile:1
# Hyperion AI 双引擎镜像
# 作者：晨星
#
# 训练：  docker build --target train  -t hyperion-train .
# 服务：  docker build --target runtime -t hyperion-ai .

# ---------- 阶段 1：训练 ----------
FROM python:3.12-slim AS train
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/py ./src/py
COPY scripts ./scripts
RUN PYTHONPATH=src/py OPENBLAS_NUM_THREADS=2 python -m hyperion.train --out assets
# 产物在 /app/assets

# ---------- 阶段 2：运行时（Node，零运行期依赖） ----------
FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production PORT=8787 HOST=0.0.0.0
COPY package.json tsconfig.json ./
COPY src/ts ./src/ts
COPY web ./web
COPY --from=train /app/assets ./assets
# 开发依赖仅用于把 TS 编译成 dist（运行期不加载任何 npm 包）。
# NODE_ENV=production 会让 npm 跳过 devDependencies，故显式 --include=dev。
RUN npm install --ignore-scripts --include=dev \
 && ./node_modules/.bin/tsc -p tsconfig.json \
 && rm -rf node_modules src/ts
EXPOSE 8787
CMD ["node", "dist/main.js"]
