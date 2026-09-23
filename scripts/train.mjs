#!/usr/bin/env node
/**
 * 训练启动器 - 调用 Python 训练内核，把产物写到 assets/
 *
 * 作者：晨星
 *
 * Python 解释器查找顺序：
 *   1. 环境变量 HYPERION_PYTHON
 *   2. 项目内虚拟环境 ./venv（Windows / POSIX 两种布局）
 *   3. PATH 上的 python
 *
 * 用法：node scripts/train.mjs [任意 hyperion.train 参数]
 *   例：node scripts/train.mjs --lm-steps 900 --batch 8
 */

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function findPython() {
  if (process.env.HYPERION_PYTHON) return process.env.HYPERION_PYTHON;
  const candidates = [
    join(root, "venv", "Scripts", "python.exe"),
    join(root, "venv", "bin", "python"),
  ];
  for (const c of candidates) if (existsSync(c)) return c;
  // PATH 上探测一次，拿不到就让 spawn 报错
  const probe = spawnSync("python", ["--version"], { stdio: "ignore" });
  if (probe.status === 0) return "python";
  const probe3 = spawnSync("python3", ["--version"], { stdio: "ignore" });
  if (probe3.status === 0) return "python3";
  throw new Error("找不到 Python。请设置 HYPERION_PYTHON 或在项目根目录创建 ./venv（见 docs/DEPLOY.md）");
}

const py = findPython();
const args = ["-m", "hyperion.train", "--out", "assets", ...process.argv.slice(2)];
const env = {
  ...process.env,
  PYTHONPATH: join(root, "src", "py"),
  // 小矩阵场景下多线程 BLAS 反而慢，且省内存（详见 docs/ARCHITECTURE.md 性能一节）
  OPENBLAS_NUM_THREADS: process.env.OPENBLAS_NUM_THREADS ?? "2",
  OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "2",
};

console.log(`[train] ${py} ${args.join(" ")}`);
const child = spawn(py, args, { stdio: "inherit", env, cwd: root });
child.on("exit", (code) => process.exit(code ?? 1));
