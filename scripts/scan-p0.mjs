#!/usr/bin/env node
/**
 * P0 扫描 - 设计红线与依赖红线的机械检查
 *
 * 作者：晨星
 *
 * 检查项：
 *   1. 代码与文档中不得出现 emoji（P0 规则：图标只用 SVG sprite）
 *   2. 不得出现紫粉渐变（P0 规则：主色 #0D6E7E / 强调 #0E7490）
 *   3. 运行期依赖必须为 0（package.json 不得有 dependencies 字段）
 *   4. console.html 不得引用任何外部 URL（单文件、离线可用）
 *
 * 退出码：0 通过；1 有违规。
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const root = join(new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const violations = [];

// 扫描目标：源码、脚本、文档、控制台（不含 node_modules / dist / assets）
function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name === "dist" || name === "assets" || name === ".git" || name === "venv") continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, acc);
    else if (/\.(ts|mjs|js|html|md|css|json|yml|yaml|py|txt)$/.test(name)) acc.push(p);
  }
  return acc;
}

const files = walk(root);

// 1. emoji 检测（覆盖主要 emoji Unicode 区段）
const emojiRe = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{1F000}-\u{1F02F}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}]/u;
// 注：箭头区 2190-21FF 太宽（会误伤 => 等 ASCII？不会，那是 0x3D）。排除常见符号：
const safeArrows = new Set(["\u2192", "\u2190", "\u2191", "\u2193", "\u2194"]);
for (const f of files) {
  const text = readFileSync(f, "utf8");
  for (const ch of text) {
    if (safeArrows.has(ch)) continue;
    if (emojiRe.test(ch)) {
      violations.push(`emoji：${f} 含字符 U+${ch.codePointAt(0).toString(16)}`);
      break;
    }
  }
}

// 2. 紫粉渐变检测（P0 红线色）
const purplePink = /linear-gradient[^;)]*(#(8b5cf6|a855f7|c084fc|d946ef|e879f9|ec4899|f472b6|a21caf|9333ea|7e22ce|86198f|f0abfc|e879f9)|purple|violet|fuchsia|magenta|pink)/i;
for (const f of files) {
  const text = readFileSync(f, "utf8");
  const m = text.match(purplePink);
  if (m) violations.push(`紫粉渐变：${f} 匹配 ${m[0].slice(0, 60)}`);
}

// 3. 零运行期依赖
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
if (pkg.dependencies && Object.keys(pkg.dependencies).length > 0) {
  violations.push(`运行期依赖：package.json dependencies 含 ${Object.keys(pkg.dependencies).join(", ")}`);
}

// 4. console.html 离线性：不得有外部 http(s) 引用（w3.org 命名空间除外）
const consolePath = join(root, "web", "console.html");
if (!existsLocal(consolePath)) {
  violations.push("缺文件：web/console.html（控制台必须存在）");
} else {
  const html = readFileSync(consolePath, "utf8");
  const urlRe = /https?:\/\/(?!www\.w3\.org)[^\s"'<>)]+/g;
  const urls = html.match(urlRe) ?? [];
  for (const u of urls) violations.push(`外部引用：web/console.html -> ${u.slice(0, 80)}`);
}

function existsLocal(p) {
  try { statSync(p); return true; } catch { return false; }
}

if (violations.length > 0) {
  console.log(`P0 扫描：${violations.length} 处违规`);
  for (const v of violations) console.log(`  [FAIL] ${v}`);
  process.exit(1);
} else {
  console.log(`P0 扫描：通过（扫描 ${files.length} 个文件；无 emoji、无紫粉渐变、0 运行期依赖、控制台完全离线）`);
}
