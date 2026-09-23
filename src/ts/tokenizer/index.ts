/**
 * M02 tokenizer - 字节级 BPE 分词器（推理侧，零第三方依赖）
 *
 * 作者：晨星
 *
 * 与 Python 侧 src/py/hyperion/tokenizer.py 共用同一份 tokenizer.json，
 * 两侧必须产出完全一致的 token id —— 这是"训练在 Python、推理在 TypeScript"
 * 这套双引擎成立的前提。因此下面三处判定必须与 Python 逐字符对齐：
 *
 *   1. 预分词规则（pretokenize）：ASCII 词 / 空白串 / CJK 串 / 其它单字符
 *   2. 基础符号映射：字节 b 的 id = b + BYTE_OFFSET(5)
 *   3. 合并符号 id：N_SPECIAL(5) + 256 + mergeRank
 *
 * 不变量（由 selfcheck 机械校验）：
 *   * 对任意文本，encode 再 decode 必须还原原文（往返恒等）
 *   * 同一文本在 Python 与 TypeScript 侧编码结果必须逐 id 相等
 */

import { readFileSync } from "node:fs";

export const SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"] as const;
export const N_SPECIAL = SPECIALS.length;
export const BYTE_OFFSET = N_SPECIAL;
export const PAD_ID = 0;
export const BOS_ID = 1;
export const EOS_ID = 2;
export const UNK_ID = 3;
export const SEP_ID = 4;

function isAsciiWord(cp: number): boolean {
  return (
    (cp >= 0x30 && cp <= 0x39) || // 0-9
    (cp >= 0x41 && cp <= 0x5a) || // A-Z
    (cp >= 0x61 && cp <= 0x7a) || // a-z
    cp === 0x5f || // _
    cp === 0x27 // '
  );
}

function isCjk(cp: number): boolean {
  return (
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x3040 && cp <= 0x30ff) ||
    (cp >= 0xac00 && cp <= 0xd7af)
  );
}

function isSpace(cp: number): boolean {
  return cp === 0x20 || cp === 0x09 || cp === 0x0a || cp === 0x0b || cp === 0x0c || cp === 0x0d || cp === 0x3000;
}

/** 确定性预分词。与 Python 实现逐字符一致。 */
export function pretokenize(text: string): string[] {
  const out: string[] = [];
  const n = text.length;
  let i = 0;
  while (i < n) {
    const cp = text.codePointAt(i) as number;
    if (isAsciiWord(cp)) {
      let j = i;
      while (j < n && isAsciiWord(text.codePointAt(j) as number)) j++;
      out.push(text.slice(i, j));
      i = j;
    } else if (isSpace(cp)) {
      let j = i;
      while (j < n && isSpace(text.codePointAt(j) as number)) j++;
      out.push(text.slice(i, j));
      i = j;
    } else if (isCjk(cp)) {
      let j = i;
      while (j < n && isCjk(text.codePointAt(j) as number)) j++;
      out.push(text.slice(i, j));
      i = j;
    } else {
      out.push(text[i]);
      i++;
    }
  }
  return out;
}

export interface TokenizerFile {
  type: string;
  version: number;
  specials: string[];
  byte_offset: number;
  vocab_size: number;
  merges: [number, number][];
}

export class Tokenizer {
  readonly vocabSize: number;
  /** key = a * 65536 + b，value = 合并后的 id。用数值 key 避免字符串拼接开销。 */
  private readonly mergeId = new Map<number, number>();
  /** key 同上，value = 合并秩（越小越先合并）。 */
  private readonly mergeRank = new Map<number, number>();
  private readonly idToString: string[] = [];

  constructor(file: TokenizerFile) {
    this.vocabSize = file.vocab_size;
    for (const s of file.specials) this.idToString.push(s);
    for (let b = 0; b < 256; b++) this.idToString.push(String.fromCharCode(b));
    file.merges.forEach(([a, b], i) => {
      const key = a * 65536 + b;
      this.mergeId.set(key, N_SPECIAL + 256 + i);
      this.mergeRank.set(key, i);
      this.idToString.push(this.idToString[a] + this.idToString[b]);
    });
  }

  static load(path: string): Tokenizer {
    return new Tokenizer(JSON.parse(readFileSync(path, "utf8")) as TokenizerFile);
  }

  static fromObject(o: TokenizerFile): Tokenizer {
    return new Tokenizer(o);
  }

  /** 单块（预分词结果）编码为 id 序列。 */
  private encodeChunk(chunk: string): number[] {
    const bytes = Buffer.from(chunk, "utf8");
    const ids: number[] = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) ids[i] = bytes[i] + BYTE_OFFSET;
    while (ids.length > 1) {
      let bestRank = Infinity;
      let bestPos = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.mergeRank.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPos = i;
        }
      }
      if (bestPos < 0) break;
      const key = ids[bestPos] * 65536 + ids[bestPos + 1];
      ids.splice(bestPos, 2, this.mergeId.get(key) as number);
    }
    return ids;
  }

  encode(text: string, opts: { bos?: boolean; eos?: boolean } = {}): number[] {
    const ids: number[] = [];
    if (opts.bos) ids.push(BOS_ID);
    for (const chunk of pretokenize(text)) {
      for (const id of this.encodeChunk(chunk)) ids.push(id);
    }
    if (opts.eos) ids.push(EOS_ID);
    return ids;
  }

  decode(ids: Iterable<number>): string {
    const buf: number[] = [];
    for (const id of ids) {
      if (id < N_SPECIAL) continue;
      const s = this.idToString[id];
      if (s === undefined) continue;
      for (let i = 0; i < s.length; i++) {
        const cp = s.charCodeAt(i);
        // idToString 里存的是字节字符，直接按字节回写
        buf.push(cp & 0xff);
      }
    }
    return Buffer.from(Uint8Array.from(buf)).toString("utf8");
  }

  /** 取某个 id 的可读表示，用于调试与引用展示。 */
  tokenString(id: number): string {
    return this.idToString[id] ?? `<未知:${id}>`;
  }
}

export const __all__ = ["Tokenizer", "pretokenize", "SPECIALS", "PAD_ID", "SEP_ID", "EOS_ID"];
