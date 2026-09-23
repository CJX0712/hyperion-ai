"""Hyperion AI - 确定性合成语料生成器

作者：晨星

为什么用合成语料：
    目标环境 huggingface.co 不可达，拿不到任何公开语料或预训练权重。为保证
    「干净环境一键复现 + 指标可计算」，本模块用固定 seed 生成一份自洽的技术
    知识库（服务 / 故障 / 配置 / 手册 / FAQ 五类文档），并为每篇文档配套生成
    带标准答案的查询对。好处：
      1. 完全离线、可复现，任何机器生成结果逐字节一致；
      2. 查询-文档相关性有真值标签，检索/重排/问答的 recall@k、MRR、faithfulness
         都是可计算的，而不是靠人眼判断；
      3. 领域自洽，小参数模型在自己的领域内能达到低的困惑度与连贯生成。

    注意：这是**领域专用**语料，不是通用语料。通用能力不在本系统 MVP 范围内。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

SERVICES: List[Tuple[str, str]] = [
    ("api-gateway", "API 网关"),
    ("vector-store", "向量库"),
    ("inference-runtime", "推理运行时"),
    ("ingest-queue", "摄取队列"),
    ("eval-service", "评估服务"),
    ("auth-service", "认证服务"),
    ("scheduler", "调度器"),
    ("cache-layer", "缓存层"),
    ("object-store", "对象存储"),
    ("trace-collector", "链路采集器"),
]

DEPENDENCIES: Dict[str, List[str]] = {
    "api-gateway": ["auth-service", "inference-runtime", "cache-layer"],
    "vector-store": ["object-store", "ingest-queue"],
    "inference-runtime": ["vector-store", "cache-layer", "object-store"],
    "ingest-queue": ["object-store", "vector-store"],
    "eval-service": ["vector-store", "inference-runtime", "trace-collector"],
    "auth-service": ["cache-layer"],
    "scheduler": ["ingest-queue", "inference-runtime"],
    "cache-layer": ["object-store"],
    "object-store": [],
    "trace-collector": ["api-gateway", "inference-runtime"],
}

METRICS: List[Tuple[str, str, str]] = [
    ("p95_latency", "毫秒", "P95 延迟"),
    ("error_rate", "百分比", "错误率"),
    ("throughput_qps", "每秒请求数", "吞吐"),
    ("cpu_util", "百分比", "CPU 利用率"),
    ("mem_rss_mb", "兆字节", "常驻内存"),
    ("queue_depth", "条", "队列深度"),
    ("cache_hit_ratio", "百分比", "缓存命中率"),
    ("token_per_sec", "每秒 token 数", "生成速度"),
]

ERRORS: List[Tuple[str, str, str]] = [
    ("ETIMEDOUT", "上游响应超时", "把超时阈值从默认值提高到两倍，并开启指数退避重试"),
    ("ECONNREFUSED", "目标端口没有进程监听", "确认目标服务已就绪，检查健康检查端点与端口映射"),
    ("OOM-409", "常驻内存超过容器上限", "降低批处理大小并启用 int8 量化，必要时水平扩容"),
    ("IDX-CORRUPT", "向量索引文件校验失败", "删除损坏分片后重建索引，重建期间切换到暴力检索"),
    ("RATE-429", "请求速率超过配额", "启用令牌桶限流，并把重试间隔改为抖动退避"),
    ("AUTH-401", "令牌过期或签名不匹配", "轮换密钥并清理本地令牌缓存"),
    ("SCHEMA-MISMATCH", "写入字段与索引模式不一致", "先做模式迁移再恢复写入，禁止直接改字段类型"),
]

CONFIGS: List[Tuple[str, str, str, str]] = [
    ("max_seq", "192", "64 到 2048", "单次上下文长度上限，调大会线性增加注意力显存与算力开销"),
    ("top_k", "8", "1 到 64", "检索返回的候选片段数量，调大提升召回率但会稀释上下文密度"),
    ("batch_size", "48", "1 到 512", "训练与推理的批大小，调大提升吞吐但增加峰值内存"),
    ("cache_ttl_sec", "300", "0 到 86400", "缓存条目存活时间，设为 0 等于关闭缓存"),
    ("thread_count", "4", "1 到 16", "推理线程数，小模型在内存带宽瓶颈下并非越多越好"),
    ("similarity_threshold", "0.62", "0 到 1", "向量检索的最低相似度门槛，调高会减少噪声但可能漏召"),
    ("chunk_size", "512", "128 到 2048", "文档分片字符数，过小丢上下文，过大稀释向量语义"),
    ("rerank_depth", "32", "4 到 128", "送入交叉编码器的候选深度，越大越准但延迟越高"),
]

RUNBOOKS: List[Tuple[str, List[str], str]] = [
    (
        "重建向量索引",
        ["暂停摄取队列", "导出当前索引快照到对象存储", "删除损坏分片", "以 batch_size 512 重建索引", "回放暂停期间的写入"],
        "索引文档数与源文档数一致，且抽查 20 条查询的 recall@5 不低于重建前",
    ),
    (
        "回滚一次失败发布",
        ["确认当前版本号与上一版本号", "冻结调度器的新任务下发", "切换镜像标签到上一版本", "等待健康检查连续 5 次通过", "解冻任务下发"],
        "健康检查连续通过且 P95 延迟回到基线区间",
    ),
    (
        "扩容推理运行时",
        ["确认 CPU 利用率与队列深度", "计算目标副本数", "滚动扩容并观察启动耗时", "逐步放开流量比例"],
        "队列深度回落到基线且错误率低于 0.1 个百分点",
    ),
    (
        "开启 int8 量化推理",
        ["在评估集上跑基线指标", "对权重做逐通道 int8 量化", "重跑评估集对比困惑度劣化", "劣化小于 8 个百分点则全量切换"],
        "量化后困惑度劣化不超过 8 个百分点",
    ),
    (
        "清理过期会话数据",
        ["列出超过保留期的会话", "导出审计摘要", "分批删除并校验计数", "压缩对象存储的生命周期规则"],
        "删除条数与审计摘要一致，且存储占用回落到预算内",
    ),
]

FAQ: List[Tuple[str, str]] = [
    ("检索结果里为什么会出现无关片段", "相似度门槛过低会让噪声进入候选集，应提高 similarity_threshold 并开启重排"),
    ("为什么开启重排之后效果反而变差", "重排器在查询语言上能力不足时会压掉正确答案，应把重排结果与融合结果再做一次 RRF 融合"),
    ("CPU 上小模型为什么线程越多越慢", "瓶颈在内存带宽而不是算力，线程数应锁在 2 到 4 之间"),
    ("向量检索和关键词检索该用哪个", "两者是互补通道，建议同时开启并用 RRF 融合，关键词通道兜住专有名词"),
    ("困惑度下降但答案质量没变好", "困惑度只衡量语言建模，不衡量事实性，应同时看 faithfulness 与引用命中率"),
    ("分片太大会有什么后果", "向量语义被稀释，检索命中整段但答案定位不准，应控制在 512 字符左右"),
    ("如何判断是否需要重排", "看融合结果的 top-1 是否被 top-5 中的正确片段压制，是则重排有收益"),
    ("评估指标忽高忽低是什么原因", "评估复用了运行期已写入数据的单例管道，应在全新管道上跑以保证确定性"),
]


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    kind: str
    entities: List[str] = field(default_factory=list)


@dataclass
class QAPair:
    query: str
    gold_doc_id: str
    answer: str
    hard_negatives: List[str] = field(default_factory=list)


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


# --------------------------------------------------------------------- 文档生成
def _gen_overview(r: random.Random, svc: str, cn: str) -> Doc:
    deps = DEPENDENCIES.get(svc, [])
    m1, m2 = r.sample(METRICS, 2)
    dep_txt = "、".join(deps) if deps else "无外部依赖"
    text = (
        f"{cn}（{svc}）是 Hyperion AI 的{'入口' if svc == 'api-gateway' else '核心'}组件，"
        f"负责{'请求路由与协议转换' if svc == 'api-gateway' else '该组件声明的职责'}。"
        f"它依赖以下组件：{dep_txt}。"
        f"运行期需要重点观测两个指标：{m1[2]}（单位 {m1[1]}）与 {m2[2]}（单位 {m2[1]}）。"
        f"当 {m1[2]} 连续三个采样周期超过基线时，应先检查依赖组件的健康状态，再检查自身资源占用。"
        f"{cn}的配置通过环境变量注入，变更需要滚动重启生效。"
    )
    return Doc(f"ov-{svc}", f"{cn}组件说明", text, "overview", [svc])


def _gen_trouble(r: random.Random, svc: str, cn: str, err: Tuple[str, str, str]) -> Doc:
    code, cause, fix = err
    m = r.choice(METRICS)
    text = (
        f"故障现象：{cn}（{svc}）在执行请求时返回错误码 {code}，同时 {m[2]}出现异常抬升。"
        f"根因：{cause}。排查步骤：第一步，确认错误码出现的比例与时间分布；"
        f"第二步，检查 {cn} 与依赖组件之间的连通性；第三步，查看最近一次配置变更记录。"
        f"修复动作：{fix}。修复后需要观察至少十分钟，确认错误率回落到基线区间。"
    )
    return Doc(f"tr-{svc}-{code}", f"{cn} {code} 故障排查", text, "troubleshooting", [svc, code])


def _gen_config(r: random.Random, cfg: Tuple[str, str, str, str]) -> Doc:
    key, default, rng_txt, impact = cfg
    text = (
        f"配置项 {key} 的默认值为 {default}，取值范围是 {rng_txt}。"
        f"调整影响：{impact}。修改方式：在环境变量中设置 {key}，重启对应服务后生效。"
        f"生产环境建议先在评估集上验证，再全量切换，避免直接在线上调整。"
    )
    return Doc(f"cf-{key}", f"配置项 {key}", text, "config", [key])


def _gen_runbook(r: random.Random, rb: Tuple[str, List[str], str]) -> Doc:
    title, steps, accept = rb
    steps_txt = "；".join(f"第 {i + 1} 步，{s}" for i, s in enumerate(steps))
    text = f"操作手册：{title}。执行步骤：{steps_txt}。验收标准：{accept}。执行前需要确认已备份当前状态。"
    return Doc(f"rb-{title}", f"操作手册：{title}", text, "runbook", [title])


def _gen_faq(r: random.Random, qa: Tuple[str, str]) -> Doc:
    q, a = qa
    text = f"问：{q}。答：{a}。相关建议是在变更前先跑一遍评估集，用数据决定而不是凭直觉。"
    return Doc(f"faq-{q[:8]}", f"FAQ：{q}", text, "faq", [q[:8]])


# --------------------------------------------------------------------- 查询生成
def _query_for(doc: Doc, r: random.Random) -> str:
    if doc.kind == "overview":
        svc = doc.entities[0]
        cn = dict(SERVICES)[svc]
        return r.choice([
            f"{cn} {svc} 的职责是什么，依赖哪些组件",
            f"{svc} 需要重点观测哪些指标",
            f"{cn}的配置变更怎么生效",
        ])
    if doc.kind == "troubleshooting":
        svc, code = doc.entities[0], doc.entities[1]
        cn = dict(SERVICES)[svc]
        return r.choice([
            f"{cn}返回 {code} 是什么原因，怎么修复",
            f"{svc} 出现 {code} 的排查步骤是什么",
            f"{code} 错误码的根因和修复动作",
        ])
    if doc.kind == "config":
        key = doc.entities[0]
        return r.choice([
            f"配置项 {key} 的默认值和取值范围是多少",
            f"调整 {key} 会有什么影响",
            f"怎么修改 {key}",
        ])
    if doc.kind == "runbook":
        title = doc.entities[0]
        return r.choice([
            f"{title}的具体步骤是什么",
            f"{title}的验收标准是什么",
            f"如何执行{title}",
        ])
    return r.choice([
        doc.title.replace("FAQ：", ""),
        doc.title.replace("FAQ：", "") + "，应该怎么处理",
    ])


def _answer_for(doc: Doc) -> str:
    """真值答案：直接取文档正文的核心句，保证 faithfulness 可判定。"""
    return doc.text


def build_corpus(seed: int = 20260923, variants: int = 4) -> Tuple[List[Doc], List[str], List[QAPair]]:
    """返回 (知识库文档, 语言建模文本列表, 查询-答案对)。

    - 知识库文档：每个模板项只保留一份规范表述，作为检索索引与评估的真值来源，
      避免「同一内容的多个改写互相抢召回」导致指标失真。
    - 语言建模文本：包含知识库文档、额外的改写变体、以及问答对，用于 LM 训练。
    - 查询-答案对：由知识库文档反向生成，gold_doc_id 有真值标签。
    """
    r = _rng(seed)
    kb: List[Doc] = []
    for svc, cn in SERVICES:
        kb.append(_gen_overview(r, svc, cn))
    for svc, cn in SERVICES:
        for err in ERRORS:
            kb.append(_gen_trouble(r, svc, cn, err))
    for cfg in CONFIGS:
        kb.append(_gen_config(r, cfg))
    for rb in RUNBOOKS:
        kb.append(_gen_runbook(r, rb))
    for qa in FAQ:
        kb.append(_gen_faq(r, qa))

    lm_texts: List[str] = [d.text for d in kb]
    for v in range(max(variants - 1, 0)):
        for svc, cn in SERVICES:
            lm_texts.append(_gen_overview(r, svc, cn).text)
        for svc, cn in SERVICES:
            for err in r.sample(ERRORS, 4):
                lm_texts.append(_gen_trouble(r, svc, cn, err).text)
        for cfg in CONFIGS:
            lm_texts.append(_gen_config(r, cfg).text)
        for rb in RUNBOOKS:
            lm_texts.append(_gen_runbook(r, rb).text)
        for qa in FAQ:
            lm_texts.append(_gen_faq(r, qa).text)

    pairs: List[QAPair] = []
    for doc in kb:
        same_kind = [d.doc_id for d in kb if d.kind == doc.kind and d.doc_id != doc.doc_id]
        neg = r.sample(same_kind, min(3, len(same_kind)))
        base = _query_for(doc, r)
        for q in [base, base + "？", "请问" + base, base.replace("，", " 并且 ", 1)]:
            pairs.append(QAPair(q, doc.doc_id, _answer_for(doc), neg))

    for p in pairs[: len(kb)]:
        lm_texts.append(f"问：{p.query}\n答：{p.answer}")
    return kb, lm_texts, pairs


def corpus_text(texts: Sequence[str]) -> str:
    return "\n".join(texts)


def export_corpus(kb: Sequence[Doc], lm_texts: Sequence[str], pairs: Sequence[QAPair], out_dir: str) -> Dict:
    import os

    os.makedirs(out_dir, exist_ok=True)
    doc_path = os.path.join(out_dir, "corpus.jsonl")
    qa_path = os.path.join(out_dir, "qa.jsonl")
    lm_path = os.path.join(out_dir, "lm.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        for d in kb:
            f.write(json.dumps({"doc_id": d.doc_id, "title": d.title, "text": d.text, "kind": d.kind}, ensure_ascii=False) + "\n")
    with open(qa_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"query": p.query, "gold_doc_id": p.gold_doc_id, "hard_negatives": p.hard_negatives}, ensure_ascii=False) + "\n")
    with open(lm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lm_texts))
    return {
        "docs": len(kb),
        "pairs": len(pairs),
        "lm_texts": len(lm_texts),
        "doc_path": doc_path,
        "qa_path": qa_path,
        "lm_path": lm_path,
    }


__all__ = ["Doc", "QAPair", "build_corpus", "corpus_text", "export_corpus"]
