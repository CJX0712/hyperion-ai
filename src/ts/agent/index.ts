/**
 * M10 agent - 有界工具调用循环（Planner / Executor / Critic）
 *
 * 作者：晨星
 *
 * 一个必须说清楚的设计取舍：
 *
 *   主流 agent 框架用「让 LLM 输出结构化工具调用」的 ReAct 范式。本系统没有采用，
 *   原因是硬件与模型规模的硬约束——CPU 上单步推理是秒级，而 1M 参数级的模型
 *   无法可靠地生成符合 JSON Schema 的工具调用，强行做只会得到一个"看起来有 agent、
 *   实际上大部分时间在解析失败"的东西。实测数据（CPU 上 10 步 agent 约 5 分钟）
 *   也说明无界循环在这个硬件上根本不可用。
 *
 *   因此这里的 Planner 是**确定性规则路由器**，而不是 LLM 规划器：
 *     * 结果可复现（同一输入必然走同一条工具路径）；
 *     * 步数有界（maxSteps 默认 3，硬上限 8）；
 *     * 失败可诊断（每一步都记录选了什么工具、为什么、观测是什么）。
 *
 *   LLM 规划器可以通过 Planner 接口注入替换（Planner 是一个纯函数），
 *   在有 GPU 或更大的模型时无需改动 Executor 与 Critic。
 *
 * 反剧场铁律：每个工具必须是纯的、可单测的、有明确错误语义的。
 * 没有观测结果的工具不允许注册。
 */

export interface ToolResult {
  ok: boolean;
  output: string;
}

export interface Tool {
  name: string;
  description: string;
  run(input: string): ToolResult;
}

export interface AgentEvent {
  type: "plan" | "step" | "observation" | "critique" | "final";
  step?: number;
  text: string;
  tool?: string;
  ok?: boolean;
}

export interface AgentResult {
  answer: string;
  events: AgentEvent[];
  steps: number;
  toolCalls: { tool: string; input: string; ok: boolean }[];
  stoppedBy: "converged" | "max_steps" | "no_tool";
}

export type Planner = (task: string, history: AgentEvent[]) => { tool: string; input: string } | null;

// ------------------------------------------------------------------ 内置工具

/** 安全四则运算求值器。不使用 eval，手写调度场算法，杜绝任意代码执行。 */
export function evaluateExpression(expr: string): number {
  const tokens: (number | string)[] = [];
  let i = 0;
  while (i < expr.length) {
    const c = expr[i];
    if (c === " ") { i++; continue; }
    if ("+-*/()%".includes(c)) { tokens.push(c); i++; continue; }
    if ((c >= "0" && c <= "9") || c === ".") {
      let j = i;
      while (j < expr.length && ((expr[j] >= "0" && expr[j] <= "9") || expr[j] === ".")) j++;
      tokens.push(parseFloat(expr.slice(i, j)));
      i = j;
      continue;
    }
    throw new Error(`表达式含非法字符：${c}`);
  }
  const prec: Record<string, number> = { "+": 1, "-": 1, "*": 2, "/": 2, "%": 2 };
  const output: (number | string)[] = [];
  const ops: string[] = [];
  for (const t of tokens) {
    if (typeof t === "number") output.push(t);
    else if (t === "(") ops.push(t);
    else if (t === ")") {
      while (ops.length && ops[ops.length - 1] !== "(") output.push(ops.pop() as string);
      if (!ops.length) throw new Error("括号不匹配");
      ops.pop();
    } else {
      while (ops.length && ops[ops.length - 1] !== "(" && prec[ops[ops.length - 1]] >= prec[t]) {
        output.push(ops.pop() as string);
      }
      ops.push(t);
    }
  }
  while (ops.length) output.push(ops.pop() as string);
  const st: number[] = [];
  for (const t of output) {
    if (typeof t === "number") { st.push(t); continue; }
    const b = st.pop(); const a = st.pop();
    if (a === undefined || b === undefined) throw new Error("表达式结构错误");
    if (t === "+") st.push(a + b);
    else if (t === "-") st.push(a - b);
    else if (t === "*") st.push(a * b);
    else if (t === "/") { if (b === 0) throw new Error("除以零"); st.push(a / b); }
    else if (t === "%") st.push(a % b);
  }
  if (st.length !== 1) throw new Error("表达式求值失败");
  return st[0];
}

export class ToolRegistry {
  private readonly tools = new Map<string, Tool>();

  register(tool: Tool): void {
    if (!tool.name) throw new Error("工具必须有名字");
    if (typeof tool.run !== "function") throw new Error(`工具 ${tool.name} 缺少 run 实现`);
    this.tools.set(tool.name, tool);
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  names(): string[] {
    return [...this.tools.keys()].sort();
  }

  describe(): { name: string; description: string }[] {
    return this.names().map((n) => ({ name: n, description: (this.tools.get(n) as Tool).description }));
  }
}

/** 默认规划器：确定性意图路由。命中顺序固定，保证可复现。 */
export function defaultPlanner(task: string, history: AgentEvent[]): { tool: string; input: string } | null {
  const used = new Set(history.filter((e) => e.type === "observation").map((e) => e.tool as string));
  const exprMatch = task.match(/[-+*/%().\d\s]{5,}/);
  if (exprMatch && /[\d]/.test(exprMatch[0]) && /[+\-*/%]/.test(exprMatch[0]) && !used.has("calculator")) {
    return { tool: "calculator", input: exprMatch[0].trim() };
  }
  if (!used.has("search_kb")) {
    return { tool: "search_kb", input: task };
  }
  if (!used.has("text_stats")) {
    return { tool: "text_stats", input: task };
  }
  return null;
}

export interface AgentOptions {
  maxSteps?: number;
  planner?: Planner;
}

/** 有界 Agent 循环。 */
export function runAgent(
  task: string,
  registry: ToolRegistry,
  opts: AgentOptions = {}
): AgentResult {
  const maxSteps = Math.max(1, Math.min(opts.maxSteps ?? 3, 8));
  const planner = opts.planner ?? defaultPlanner;
  const events: AgentEvent[] = [];
  const toolCalls: AgentResult["toolCalls"] = [];
  let best = "";
  let steps = 0;
  let stoppedBy: AgentResult["stoppedBy"] = "max_steps";

  events.push({ type: "plan", text: `接收任务，最大步数 ${maxSteps}，可用工具 ${registry.names().join(", ")}` });

  for (let s = 1; s <= maxSteps; s++) {
    const decision = planner(task, events);
    if (!decision) { stoppedBy = "no_tool"; break; }
    const tool = registry.get(decision.tool);
    steps = s;
    events.push({ type: "step", step: s, text: `选择工具 ${decision.tool}`, tool: decision.tool });

    if (!tool) {
      events.push({ type: "observation", step: s, text: `工具 ${decision.tool} 未注册`, tool: decision.tool, ok: false });
      toolCalls.push({ tool: decision.tool, input: decision.input, ok: false });
      events.push({ type: "critique", step: s, text: "工具不存在，改用其余可用工具重试" });
      continue;
    }

    let res: ToolResult;
    try {
      res = tool.run(decision.input);
    } catch (err) {
      res = { ok: false, output: `工具执行抛出异常：${(err as Error).message}` };
    }
    toolCalls.push({ tool: decision.tool, input: decision.input, ok: res.ok });
    events.push({ type: "observation", step: s, text: res.output, tool: decision.tool, ok: res.ok });

    if (res.ok && res.output.trim().length > best.trim().length) {
      best = res.output;
      events.push({ type: "critique", step: s, text: "观测有效，采纳为当前最佳结果" });
      if (decision.tool === "search_kb" || decision.tool === "calculator") break;
    } else {
      events.push({ type: "critique", step: s, text: res.ok ? "观测信息量不足，继续尝试其它工具" : "工具失败，继续尝试" });
    }
  }

  const answer = best.trim() || "未能通过可用工具得到有效结果。可尝试换个问法，或确认知识库已被摄取。";
  events.push({ type: "final", text: answer });
  return { answer, events, steps, toolCalls, stoppedBy };
}

export const __all__ = ["ToolRegistry", "runAgent", "defaultPlanner", "evaluateExpression", "Tool", "AgentEvent"];
