"""Hyperion AI - 字节级 BPE 分词器（Python 侧训练，TS 侧复用同一份词表）

作者：晨星

设计要点：
  * 字节级 BPE（GPT-2 思路）：基础符号 = 256 个字节，天然支持中英混排与任意 Unicode。
  * 预分词规则在 Python 与 TypeScript 中逐字符一致（见 CHUNK_ 判定函数），保证两侧编码结果完全对齐。
  * 词表构造顺序确定：特殊符号 -> 256 字节 -> 按合并秩排序的合并符号，因此只需导出 merges 即可重建词表。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

SPECIALS: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]
N_SPECIAL = len(SPECIALS)
BYTE_OFFSET = N_SPECIAL  # 字节 b 的 id = b + BYTE_OFFSET


def _is_ascii_word(ch: str) -> bool:
    o = ord(ch)
    return (
        (0x30 <= o <= 0x39) or (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A) or ch == "_" or ch == "'"
    )


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x3040 <= o <= 0x30FF
        or 0xAC00 <= o <= 0xD7AF
    )


def _is_space(ch: str) -> bool:
    return ch in " \t\r\n\v\f" or ord(ch) == 0x3000


def pretokenize(text: str) -> List[str]:
    """确定性预分词：ASCII 词 / 空白串 / CJK 串 / 其它单字符。两侧实现必须一致。"""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _is_ascii_word(ch):
            j = i
            while j < n and _is_ascii_word(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
        elif _is_space(ch):
            j = i
            while j < n and _is_space(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
        elif _is_cjk(ch):
            j = i
            while j < n and _is_cjk(text[j]):
                j += 1
            out.append(text[i:j])
            i = j
        else:
            out.append(ch)
            i += 1
    return out


def chunk_to_base(chunk: str) -> Tuple[int, ...]:
    return tuple(b + BYTE_OFFSET for b in chunk.encode("utf-8"))


class BPETokenizer:
    def __init__(self, merges: Sequence[Tuple[int, int]], vocab_size: int) -> None:
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.vocab_size = int(vocab_size)
        self.ranks: Dict[Tuple[int, int], int] = {tuple(m): i for i, m in enumerate(self.merges)}  # type: ignore[misc]
        self.id_to_token: Dict[int, str] = {}
        for i, s in enumerate(SPECIALS):
            self.id_to_token[i] = s
        for b in range(256):
            self.id_to_token[b + BYTE_OFFSET] = chr(b)
        for i, (a, b) in enumerate(self.merges):
            self.id_to_token[N_SPECIAL + 256 + i] = self.id_to_token[a] + self.id_to_token[b]
        # 预先展开合并结果，加速编码
        self._merged: Dict[Tuple[int, int], int] = {
            tuple(m): N_SPECIAL + 256 + i for i, m in enumerate(self.merges)  # type: ignore[misc]
        }

    # ---------------------------------------------------------------- 编码
    def _encode_chunk(self, chunk: str) -> List[int]:
        ids = list(chunk_to_base(chunk))
        while len(ids) > 1:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_pos < 0:
                break
            a, b = ids[best_pos], ids[best_pos + 1]
            ids[best_pos : best_pos + 2] = [self._merged[(a, b)]]
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(SPECIALS.index("<bos>"))
        for chunk in pretokenize(text):
            ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(SPECIALS.index("<eos>"))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < N_SPECIAL:
                continue
            tok = self.id_to_token.get(i)
            if tok is None:
                continue
            buf.extend(tok.encode("utf-8", errors="ignore"))
        return buf.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> Dict:
        return {
            "type": "bpe-byte-level",
            "version": 1,
            "specials": SPECIALS,
            "byte_offset": BYTE_OFFSET,
            "vocab_size": self.vocab_size,
            "merges": [[a, b] for a, b in self.merges],
        }

    @staticmethod
    def from_dict(d: Dict) -> "BPETokenizer":
        return BPETokenizer([tuple(m) for m in d["merges"]], d["vocab_size"])  # type: ignore[misc]


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 2048,
    min_chunk_freq: int = 2,
    verbose: bool = False,
) -> BPETokenizer:
    """标准字节级 BPE：统计相邻对频率 -> 合并最高频对 -> 迭代。"""
    if vocab_size <= N_SPECIAL + 256:
        raise ValueError("vocab_size 必须大于 261")
    n_merges = vocab_size - N_SPECIAL - 256

    freq: Counter = Counter()
    for t in texts:
        freq.update(pretokenize(t))

    words: Dict[str, List[int]] = {}
    for chunk, c in freq.items():
        if c >= min_chunk_freq or len(chunk) <= 1:
            words[chunk] = list(chunk_to_base(chunk))
    counts: Dict[str, int] = {k: freq[k] for k in words}

    pairs: Counter = Counter()
    where: Dict[Tuple[int, int], set] = defaultdict(set)
    for k, ids in words.items():
        c = counts[k]
        for i in range(len(ids) - 1):
            p = (ids[i], ids[i + 1])
            pairs[p] += c
            where[p].add(k)

    merges: List[Tuple[int, int]] = []
    next_id = N_SPECIAL + 256
    for step in range(n_merges):
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(best)
        new_id = next_id + step

        affected = where.pop(best, set())
        for k in list(affected):
            ids = words[k]
            if not ids:
                continue
            c = counts[k]
            # 移除旧对统计
            for i in range(len(ids) - 1):
                p = (ids[i], ids[i + 1])
                pairs[p] -= c
                if pairs[p] <= 0:
                    pairs.pop(p, None)
                if p in where:
                    where[p].discard(k)
            # 应用合并
            new_ids: List[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            words[k] = new_ids
            # 新增对统计
            for i in range(len(new_ids) - 1):
                p = (new_ids[i], new_ids[i + 1])
                pairs[p] += c
                where[p].add(k)
        if verbose and (step + 1) % 200 == 0:
            print(f"  BPE merges: {step + 1}/{n_merges}", flush=True)

    return BPETokenizer(merges, vocab_size)


def save_tokenizer(tok: BPETokenizer, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok.to_dict(), f, ensure_ascii=False, separators=(",", ":"))


def load_tokenizer(path: str) -> BPETokenizer:
    with open(path, "r", encoding="utf-8") as f:
        return BPETokenizer.from_dict(json.load(f))


__all__ = ["BPETokenizer", "train_bpe", "pretokenize", "save_tokenizer", "load_tokenizer", "SPECIALS"]
