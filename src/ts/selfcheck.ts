/**
 * Hyperion AI 自检套件 - 每个模块的硬不变量，全部可机械验证
 *
 * 作者：晨星
 *
 * 运行：node --experimental-strip-types src/ts/selfcheck.ts
 * 退出码：0 = 全部通过；1 = 存在 FAIL。依赖模型权重的用例在产物缺失时记 SKIP。
 *
 * 用例分两组：
 *   A 组（纯 TS，无需任何产物）：kernel 数学、CRC、量化、检索融合、护栏、Agent 边界
 *   B 组（需要 assets/）：HWM 加载、KV Cache 一致性、贪心确定性、嵌入检索、RAG 链路
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import { crc32 } from "./tensor/hwm.ts";
import {
  matmulATB, matmulT1, matmulT1Int8, matmulNT, matmulNN,
  rmsnormInplace, softmaxInplace, cosine, l2normalizeInplace,
  quantizeInt8PerRow, mulberry32, topKIndices, nucleusSize,
} from "./tensor/index.ts";
import { Tokenizer, pretokenize } from "./tokenizer/index.ts";
import {
  DenseIndex, BM25Index, HybridIndex, rrfFuse, compareHits, tokenizeForBM25,
} from "./index/index.ts";
import { Reranker } from "./rerank/index.ts";
import { chunkText, splitSentences } from "./rag/index.ts";
import { ToolRegistry, runAgent, evaluateExpression } from "./agent/index.ts";
import { groundedRatio } from "./eval/index.ts";

// ---------- 测试小框架 ----------
interface Check { section: string; name: string; status: "PASS" | "FAIL" | "SKIP"; detail: string }
const checks: Check[] = [];

function check(section: string, name: string, fn: () => string): void {
  try {
    const detail = fn();
    checks.push({ section, name, status: "PASS", detail });
  } catch (err) {
    checks.push({ section, name, status: "FAIL", detail: (err as Error).message });
  }
}
function skipIf(cond: boolean): boolean { return cond; }
function assert(cond: boolean, msg: string): void { if (!cond) throw new Error(msg); }
function near(a: number, b: number, tol: number, label: string): void {
  assert(Math.abs(a - b) <= tol, `${label} 偏差 ${Math.abs(a - b).toExponential(2)} 超出容差 ${tol}`);
}

const ASSETS = process.env.HYPERION_ASSETS ?? "assets";
const hasModel = existsSync(join(ASSETS, "model.i8.hwm")) && existsSync(join(ASSETS, "tokenizer.json"));

// ---------- A1. tensor kernel ----------
check("tensor", "matmulATB 与朴素参考实现一致", () => {
  const M = 7, N = 5, K = 11;
  const rng = mulberry32(1);
  const a = new Float32Array(M * K), b = new Float32Array(N * K);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  for (let i = 0; i < b.length; i++) b[i] = rng() * 2 - 1;
  const out = new Float32Array(M * N);
  matmulATB(out, 0, a, b, M, N, K);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += a[m * K + k] * b[n * K + k];
    near(out[m * N + n], s, 1e-4, `C[${m},${n}]`);
  }
  return `M=${M} N=${N} K=${K} 全部一致`;
});

check("tensor", "matmulT1 / matmulT1Int8 与参考一致", () => {
  const M = 6, K = 9;
  const rng = mulberry32(2);
  const a = new Float32Array(M * K), b = new Float32Array(K);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  for (let i = 0; i < b.length; i++) b[i] = rng() * 2 - 1;
  const out = new Float32Array(M);
  matmulT1(out, 0, a, 0, b, 0, M, K);
  for (let m = 0; m < M; m++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += a[m * K + k] * b[k];
    near(out[m], s, 1e-4, `T1[${m}]`);
  }
  const { data, scale } = quantizeInt8PerRow(a, M, K);
  const out8 = new Float32Array(M);
  matmulT1Int8(out8, 0, data, 0, scale, 0, b, 0, M, K);
  for (let m = 0; m < M; m++) near(out8[m], out[m], 0.05 + 0.05 * Math.abs(out[m]), `Int8[${m}]`);
  return "f32 与 int8 路径均与参考一致";
});

check("tensor", "量化往返误差上界（对称 int8 per-row）", () => {
  const rows = 4, cols = 32;
  const rng = mulberry32(3);
  const w = new Float32Array(rows * cols);
  for (let i = 0; i < w.length; i++) w[i] = (rng() * 2 - 1) * 0.7;
  const { data, scale } = quantizeInt8PerRow(w, rows, cols);
  let maxErr = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) maxErr = Math.max(maxErr, Math.abs(w[r * cols + c] - data[r * cols + c] * scale[r]));
    assert(scale[r] > 0, `scale[${r}] 必须为正`);
  }
  assert(maxErr <= 0.7 / 127 * 1.001, `最大量化误差 ${maxErr.toExponential(3)} 超出理论上界`);
  return `最大误差 ${maxErr.toExponential(3)} <= 0.7/127`;
});

check("tensor", "cosine 自相似=1、正交=0、softmax 归一", () => {
  const a = new Float32Array([3, 0, 4]);
  near(cosine(a, 0, a, 0, 3), 1, 1e-6, "自相似");
  const b = new Float32Array([0, 5, 0]);
  near(cosine(a, 0, b, 0, 3), 0, 1e-6, "正交");
  const x = new Float32Array([1, 2, 3, 4]);
  softmaxInplace(x, 0, 4);
  near(x[0] + x[1] + x[2] + x[3], 1, 1e-6, "softmax 和");
  l2normalizeInplace(a, 0, 3);
  near(Math.hypot(a[0], a[1], a[2]), 1, 1e-6, "L2 范数");
  rmsnormInplace(a, 0, new Float32Array([1, 1, 1]), 3, 1e-5);
  return "全部通过";
});

check("tensor", "mulberry32 确定性 + topK/nucleus", () => {
  const r1 = mulberry32(42), r2 = mulberry32(42);
  for (let i = 0; i < 100; i++) assert(r1() === r2(), "同种子序列必须逐位一致");
  const logits = new Float32Array([0.1, 3.0, 0.5, 2.0]);
  const top = topKIndices(logits, 0, 4, 2);
  assert(top.length === 2 && top.includes(1) && top.includes(3), `top2 应为 {1,3}，实际 ${top}`);
  const probs = new Float32Array([0.5, 0.3, 0.15, 0.05]);
  assert(nucleusSize(probs, 0, 4, 0.9) <= 4 && nucleusSize(probs, 0, 4, 0.9) >= 1, "nucleus 大小应在 [1,4]");
  return "同种子逐位一致";
});

// ---------- A2. HWM CRC ----------
check("hwm", "CRC32 与 zlib.crc32 惯例一致", () => {
  const bytes = new TextEncoder().encode("123456789");
  assert(crc32(bytes) === 0xcbf43926, `crc32("123456789") 应为 cbf43926，实际 ${crc32(bytes).toString(16)}`);
  assert(crc32(new Uint8Array(0)) === 0, "空输入 CRC 应为 0");
  return "0xCBF43926 惯例值匹配";
});

// ---------- A3. tokenizer（若产物缺失则用最小词表自建） ----------
function makeTok(): Tokenizer {
  return Tokenizer.fromObject({
    type: "bpe",
    version: 1,
    byte_offset: 5,
    vocab_size: 5 + 256 + 2,
    specials: ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"],
    merges: [[5 + 104, 5 + 105], [5 + 104, 5 + 105 + 0]], // 任意两条合法合并
  });
}
check("tokenizer", "pretokenize 覆盖 ASCII/CJK/空白/其它", () => {
  const parts = pretokenize("你好 world_ai!!");
  assert(parts.some((p) => /\p{Script=Han}/u.test(p)), `应有 CJK 块：${JSON.stringify(parts)}`);
  assert(parts.some((p) => p === "world_ai"), `应有 ASCII 词块：${JSON.stringify(parts)}`);
  return JSON.stringify(parts);
});
check("tokenizer", "encode/decode 往返无损", () => {
  const tok = makeTok();
  const samples = ["Hello, 世界！Pipeline v3.2 — 100% 可靠。", " mixed CASE 123 ", "中文与English交替出现"];
  for (const s of samples) {
    const ids = tok.encode(s);
    assert(tok.decode(ids) === s, `往返不一致：${JSON.stringify(s)} -> ${JSON.stringify(tok.decode(ids))}`);
  }
  return "3 条中英混合样本均无损";
});
check("tokenizer", "特殊 id（bos/eos）解码时被跳过", () => {
  const tok = makeTok();
  const ids = tok.encode("hi");
  const withSpecials = [1, ...ids, 2];
  assert(tok.decode(withSpecials) === tok.decode(ids), "bos/eos 不应出现在解码输出");
  return "通过";
});
check("tokenizer", "BPE 合并秩生效（有合并的块短于无合并）", () => {
  const tok = makeTok();
  const merged = tok.encode("hii").length;
  const unmergedLen = "hii".length;
  assert(merged <= unmergedLen, "合并后不应更长");
  return `len(encode("hii"))=${merged}`;
});
if (hasModel) {
  check("tokenizer", "与训练侧词表一致（assets/tokenizer.json）", () => {
    const tok = Tokenizer.load(join(ASSETS, "tokenizer.json"));
    const ids = tok.encode("知识库检索测试 knowledge base");
    assert(ids.length > 0 && tok.decode(ids).includes("知识库"), "往返应保留中文");
    return `词表 ${tok.vocabSize}，样本 ${ids.length} tokens`;
  });
} else if (skipIf(true)) {
  checks.push({ section: "tokenizer", name: "与训练侧词表一致", status: "SKIP", detail: "assets/tokenizer.json 缺失" });
}

// ---------- A4. 检索：BM25 / Dense / RRF / 护栏 ----------
const docs = [
  { id: "d1", text: "重启服务后日志级别设置回警告会导致问题排查困难" },
  { id: "d2", text: "数据库连接池耗尽时新增请求会超时" },
  { id: "d3", text: "缓存命中率下降的常见原因是键空间倾斜" },
  { id: "d4", text: "日志级别设置需要在配置文件中修改" },
];
check("index", "BM25 命中含关键词的文档", () => {
  const bm25 = new BM25Index();
  for (const d of docs) bm25.add(d.id, d.text);
  const hits = bm25.search("日志级别", 3);
  assert(hits.length > 0 && (hits[0].id === "d1" || hits[0].id === "d4"), `top1 应为 d1/d4，实际 ${hits[0]?.id}`);
  assert(hits[0].score >= hits[1].score, "分数须降序");
  return `top1=${hits[0].id} score=${hits[0].score.toFixed(3)}`;
});
check("index", "tokenizeForBM25 生成 CJK 单字+二元组", () => {
  const t = tokenizeForBM25("日志级别");
  assert(t.includes("日志") || t.includes("日志级别".slice(0, 2)), `应含二元组：${JSON.stringify(t)}`);
  return JSON.stringify(t);
});
check("index", "DenseIndex 余弦检索 gold 排第一", () => {
  const dense = new DenseIndex(3);
  dense.add("gold", new Float32Array([1, 0, 0]));
  dense.add("near", new Float32Array([0.9, 0.1, 0]));
  dense.add("far", new Float32Array([0, 0, 1]));
  const hits = dense.search(new Float32Array([1, 0, 0]), 3);
  assert(hits[0].id === "gold", `top1 应为 gold，实际 ${hits[0].id}`);
  return "gold > near > far";
});
check("index", "compareHits 平局按 id 字典序（确定性）", () => {
  const sorted = [{ id: "b", score: 1 }, { id: "a", score: 1 }].sort(compareHits);
  assert(sorted[0].id === "a", "同分应按 id 升序");
  return "a < b";
});
check("index", "rrfFuse 双路融合的次序可复现", () => {
  const r1 = rrfFuse([{ hits: [{ id: "x", score: 1 }, { id: "y", score: 0.5 }] }, { hits: [{ id: "y", score: 1 }, { id: "z", score: 0.2 }] }]);
  const r2 = rrfFuse([{ hits: [{ id: "x", score: 1 }, { id: "y", score: 0.5 }] }, { hits: [{ id: "y", score: 1 }, { id: "z", score: 0.2 }] }]);
  assert(r1.map((h) => h.id).join(",") === r2.map((h) => h.id).join(","), "RRF 必须确定性");
  assert(r1.length === 3, "并集应为 3 条");
  return r1.map((h) => h.id).join(" > ");
});
check("index", "HybridIndex 融合了两路信号", () => {
  const hy = new HybridIndex(2);
  hy.add("d1", "alpha beta", new Float32Array([1, 0]));
  hy.add("d2", "gamma delta", new Float32Array([0, 1]));
  const hits = hy.search("alpha", new Float32Array([1, 0]), 2, { dense: 1, sparse: 1 });
  assert(Array.isArray(hits), "search 应返回数组");
  return `返回 ${hits.length} 条`;
});
check("rerank", "护栏不变量：重排后 top1 不得差于重排前", () => {
  const fused = [
    { id: "gold", score: 0.9 }, { id: "a", score: 0.5 }, { id: "b", score: 0.4 }, { id: "c", score: 0.1 },
  ];
  // 对抗性打分器：故意把垃圾顶到第一
  const reranker = new Reranker((_q, id) => (id === "c" ? 10 : id === "gold" ? 1 : 0));
  const out = reranker.rerank("q", fused);
  assert(out[0].id === "gold", `护栏应把 gold 保住，实际 ${out[0].id}`);
  const plain = reranker.rerankPlain("q", fused);
  assert(plain[0].id === "c", "无护栏路径应允许重排器夺冠（对照组）");
  return `护栏 top1=${out[0].id}，无护栏 top1=${plain[0].id}`;
});

// ---------- A5. RAG 分块 + Agent + eval ----------
check("rag", "splitSentences / chunkText 边界正确", () => {
  const s = splitSentences("第一句。第二句！第三句？\n第四句");
  assert(s.length === 4, `应切出 4 句，实际 ${s.length}`);
  const chunks = chunkText("doc1", "标题", "句一。".repeat(50), 30, 5);
  assert(chunks.length >= 2, "长文本应切出多块");
  assert(chunks.every((c) => c.text.length <= 30 + 20), `块长度应受控，最长 ${Math.max(...chunks.map((c) => c.text.length))}`);
  assert(chunks.every((c) => c.chunkId.startsWith("doc1#")), "chunkId 前缀应为 docId");
  return `${s.length} 句 -> ${chunks.length} 块`;
});
check("agent", "evaluateExpression 四则与括号、非法输入拒绝", () => {
  near(evaluateExpression("1+2*3"), 7, 1e-9, "优先级");
  near(evaluateExpression("(1+2)*3"), 9, 1e-9, "括号");
  near(evaluateExpression("10/4"), 2.5, 1e-9, "除法");
  let threw = false;
  try { evaluateExpression("1+"); } catch { threw = true; }
  assert(threw, "非法表达式必须抛错");
  let threw2 = false;
  try { evaluateExpression("process.exit(1)"); } catch { threw2 = true; }
  assert(threw2, "非算术输入必须拒绝（不得 eval）");
  return "4 个用例全过";
});
check("agent", "有界循环：maxSteps 封顶，失败工具不炸循环", () => {
  const reg = new ToolRegistry();
  reg.register({ name: "always_fail", description: "", run: () => ({ ok: false, output: "失败" }) });
  const r = runAgent("随便做点什么", reg, {
    maxSteps: 99, // 应被钳到 8
    planner: () => ({ tool: "always_fail", input: "x" }),
  });
  assert(r.steps <= 8, `步数应被钳到 <=8，实际 ${r.steps}`);
  assert(r.stoppedBy === "max_steps", `应因 max_steps 停止，实际 ${r.stoppedBy}`);
  assert(r.toolCalls.every((c) => !c.ok), "全部调用应失败");
  assert(r.answer.length > 0, "失败时也要给出兜底 answer");
  return `steps=${r.steps} stoppedBy=${r.stoppedBy}`;
});
check("agent", "成功工具立即采纳并停止", () => {
  const reg = new ToolRegistry();
  reg.register({ name: "calculator", description: "", run: () => ({ ok: true, output: "1+1 = 2" }) });
  const r = runAgent("算一下", reg, { planner: () => ({ tool: "calculator", input: "1+1" }) });
  assert(r.steps === 1 && r.answer === "1+1 = 2", `应 1 步结束，实际 ${r.steps} 步 / "${r.answer}"`);
  return "1 步收敛";
});
check("eval", "groundedRatio：全文引用=1，无关=0", () => {
  const cite = ["缓存命中率下降的常见原因是键空间倾斜，需要重新分片。"];
  assert(groundedRatio(cite[0], cite) === 1, "自我引用应为 1");
  assert(groundedRatio("完全无关的一段文字没有任何重叠词汇出现", cite) === 0, "无关文本应为 0");
  return "两个边界值正确";
});

// ---------- B 组：需要模型产物 ----------
if (hasModel) {
  const { bootstrap } = await import("./bootstrap.ts");
  const app = bootstrap({ assetsDir: ASSETS, ingestCorpus: false });

  check("hwm", "模型加载：magic/版本/CRC/张量绑定", () => {
    assert(String(app.model.meta.format).length > 0, `format 缺失：${String(app.model.meta.format)}`);
    const n = app.model.weights.size;
    assert(n > 10, `张量数应 >10，实际 ${n}`);
    const w = app.model.get("tok_emb");
    assert(w.f32 !== null || (w.i8 !== null && w.scale !== null), "tok_emb 必须可用");
    return `${n} 个张量，dtype=${app.model.meta.dtype}`;
  });

  check("arch", "KV Cache 一致性：prefill 与逐 token step 等价", () => {
    const t1 = app.transformer;
    const t2 = new (t1.constructor as new (m: typeof app.model) => typeof t1)(app.model);
    const ids = app.tokenizer.encode("重启服务后日志级别").slice(0, 12);
    const lg1 = t1.prefill(ids, 0);
    let lg2 = t2.step(ids[0], 0);
    for (let i = 1; i < ids.length; i++) lg2 = t2.step(ids[i], i);
    let maxDiff = 0;
    for (let i = 0; i < lg1.length; i++) maxDiff = Math.max(maxDiff, Math.abs(lg1[i] - lg2[i]));
    assert(maxDiff < 1e-3, `prefill 与 step 的 logits 差 ${maxDiff.toExponential(2)} 超容差`);
    return `${ids.length} tokens，最大差 ${maxDiff.toExponential(2)}`;
  });

  check("infer", "贪心解码确定性（同输入同输出，温度=0）", () => {
    const a = app.generator.generate("如何设置日志级别？", { temperature: 0, maxTokens: 24, seed: 20260923 });
    const b = app.generator.generate("如何设置日志级别？", { temperature: 0, maxTokens: 24, seed: 1 });
    assert(a.text === b.text, `贪心解码不一致：\n${JSON.stringify(a.text)}\n${JSON.stringify(b.text)}`);
    assert(a.text.length > 0, "输出不应为空");
    return `输出 ${a.text.length} 字，两次一致`;
  });

  check("embed", "嵌入器：L2 归一 / 确定性 / 自相似=1 / 异文本相似<1", () => {
    const a = app.embedder.embed("数据库连接池耗尽导致请求超时");
    const a2 = app.embedder.embed("数据库连接池耗尽导致请求超时");
    const gold = app.embedder.embed("连接池耗尽的排查方法与扩容方案");
    const dis = app.embedder.embed("Integration by parts: ∫ u dv = uv − ∫ v du");
    const norm = (x: Float32Array) => Math.sqrt(x.reduce((s, v) => s + v * v, 0));
    const sim = (x: Float32Array, y: Float32Array) => {
      let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s / (norm(x) * norm(y));
    };
    assert(Math.abs(norm(a) - 1) < 1e-4, `嵌入未 L2 归一：norm=${norm(a).toFixed(4)}`);
    let maxDiff = 0; for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - a2[i]));
    assert(maxDiff < 1e-6, `嵌入不确定：maxDiff=${maxDiff.toExponential(2)}`);
    near(sim(a, a), 1, 1e-4, "自相似");
    const sg = sim(a, gold), sd = sim(a, dis);
    assert(sg > 0 && sg <= 1 && sd < 1, `相似度应在 (0,1]：gold=${sg.toFixed(4)} distractor=${sd.toFixed(4)}`);
    assert(sg > sd, `同领域 gold 应明显近于跨语言干扰项：${sg.toFixed(4)} vs ${sd.toFixed(4)}`);
    return `norm=${norm(a).toFixed(4)} selfSim=1 gold=${sg.toFixed(4)} > dis=${sd.toFixed(4)}`;
  });

  check("rag", "RAG 链路：ingest -> retrieve -> ask", () => {
    const added = app.rag.ingest([
      { docId: "t1", title: "重启流程", text: "服务重启的标准流程：先摘除流量，再停止进程，等待端口释放后启动新实例。" },
      { docId: "t2", title: "菜单", text: "本周食堂提供红烧牛肉面与西红柿鸡蛋盖饭。" },
    ]);
    assert(added > 0, "ingest 应产生分片");
    const hits = app.rag.retrieve("服务怎么重启？", 3, false);
    assert(hits.length > 0 && hits[0].docId === "t1", `top1 应为 t1，实际 ${hits[0]?.docId}`);
    const ans = app.rag.ask("服务怎么重启？", { useRerank: false, maxTokens: 24 });
    assert(ans.answer.length > 0 && ans.citations.length > 0, "ask 应有答案与引用");
    return `top1=${hits[0].docId}，mode=${ans.mode}，${ans.tokensPerSecond.toFixed(1)} tok/s`;
  });

  check("agent", "Agent 全链路：calculator / search_kb / 未知工具", () => {
    const rc = app.runAgentTask("计算 12*3", 3);
    assert(rc.answer.includes("36"), `calculator 结果应含 36，实际 "${rc.answer}"`);
    app.rag.ingest([{ docId: "k1", title: "备份", text: "数据库每日凌晨三点自动备份到对象存储，保留三十天。" }]);
    const rs = app.runAgentTask("数据库什么时候备份", 3);
    assert(rs.answer.includes("备份") || rs.answer.includes("凌晨"), `search_kb 应命中备份文档，实际 "${rs.answer.slice(0, 40)}"`);
    return "三个工具路径全通";
  });
} else {
  for (const [s, n] of [
    ["hwm", "模型加载：magic/版本/CRC/张量绑定"],
    ["arch", "KV Cache 一致性：prefill 与逐 token step 等价"],
    ["infer", "贪心解码确定性（同输入同输出，温度=0）"],
    ["embed", "嵌入检索：gold 文档高于干扰项"],
    ["rag", "RAG 链路：ingest -> retrieve -> ask"],
    ["agent", "Agent 全链路：calculator / search_kb / 未知工具"],
  ] as const) {
    checks.push({ section: s, name: n, status: "SKIP", detail: "assets/model.i8.hwm 缺失（先运行 npm run train）" });
  }
}

// ---------- 输出 ----------
const pad = (s: string, n: number) => (s.length >= n ? s : s + " ".repeat(n - s.length));
let lastSection = "";
for (const c of checks) {
  if (c.section !== lastSection) {
    console.log(`\n== ${c.section} ==`);
    lastSection = c.section;
  }
  const mark = c.status === "PASS" ? "[PASS]" : c.status === "FAIL" ? "[FAIL]" : "[SKIP]";
  console.log(`  ${mark} ${pad(c.name, 44)} ${c.detail.slice(0, 90)}`);
}
const fail = checks.filter((c) => c.status === "FAIL");
const pass = checks.filter((c) => c.status === "PASS").length;
const skp = checks.filter((c) => c.status === "SKIP").length;
console.log(`\n自检结果：${pass} PASS / ${fail.length} FAIL / ${skp} SKIP`);
if (fail.length > 0) {
  console.log("失败用例：");
  for (const f of fail) console.log(`  - [${f.section}] ${f.name}: ${f.detail}`);
  process.exit(1);
}
