#!/usr/bin/env python3
"""Search arXiv for papers related to docs/modi_steer_opd.md."""

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree


API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
REQUEST_INTERVAL = 3
CAT_FILTER = "(cat:cs.CL OR cat:cs.LG OR cat:cs.AI)"


def title_only(*phrases):
    return "(" + " OR ".join(f'ti:"{phrase}"' for phrase in phrases) + ")"


def title_or_abstract(*phrases):
    terms = [f'{field}:"{phrase}"' for phrase in phrases for field in ("ti", "abs")]
    return "(" + " OR ".join(terms) + ")"


# Apply exclusions to titles so that mentions in the abstract still match.
EXCLUDE_TITLE = title_only(
    "diffusion", "image", "video", "audio", "speech", "vision-language",
    "multimodal", "VLM", "3D", "robotics", "autonomous", "embodied",
    "dataset distillation", "graph", "federated", "recommender",
    "object detection", "segmentation", "quantization", "pruning",
    "unlearning", "watermark", "backdoor", "molecule", "protein", "quantum",
)

LANGUAGE_MODELS = title_or_abstract(
    "language model", "language models", "LLM", "LLMs", "thinking models", "reasoning models",
)
DISTILLATION = title_or_abstract(
    "distillation", "self-distillation", "on-policy distillation",
    "on-policy self-distillation", "OPD", "OPSD",
)
ON_POLICY = title_or_abstract(
    "on-policy distillation", "on-policy knowledge distillation",
    "on-policy self-distillation", "generalized knowledge distillation", "OPSD",
)
STEERING = title_or_abstract(
    "steering vector", "steering vectors", "activation steering",
    "representation steering", "contrastive activation addition",
    "activation-conditioned", "steering the teacher", "activation addition",
    "spherical steering", "angular steering",
)

TOPICS = {
    "teacher-steering": (
        "核心竞品：教师激活干预与蒸馏/训练信号",
        f"({STEERING} AND ({DISTILLATION} OR "
        + title_or_abstract("training signal", "post-training", "teacher", "logit")
        + f") AND {LANGUAGE_MODELS})"
        ' OR ti:"Steering the Teacher"'
        ' OR ti:"Steering Vectors as a Training Signal"'
        ' OR ti:"Activation-Conditioned Self-Distillation"',
    ),
    "geometry-dynamic-steering": (
        "几何保范数(Slerp/球面)、防副作用与动态门控 Steering",
        f"{STEERING} AND {LANGUAGE_MODELS} AND " + title_or_abstract(
            "spherical", "Slerp", "angular", "geodesic", "norm-preserving",
            "collateral damage", "selective", "gating", "gate",
            "dynamic", "adaptive", "online", "annealing", "steering strength",
        ),
    ),
    "cross-tokenizer-opd": (
        "跨家族/跨 tokenizer OPD 与词表对齐",
        f"({DISTILLATION} AND {LANGUAGE_MODELS} AND " + title_or_abstract(
            "cross-tokenizer", "tokenizer barrier", "different tokenizers",
            "tokenizer mismatch", "cross-family", "within-family",
            "vocabulary alignment", "vocabulary mismatch", "dual-space",
            "universal logit", "span alignment", "supervision reliability",
        ) + ') OR ti:"CompassOPD" OR ti:"SimCT"',
    ),
    "opd-correctness": (
        "OPD 正确性过滤、教师误导与奖惩对齐",
        f"{ON_POLICY} AND {LANGUAGE_MODELS} AND " + title_or_abstract(
            "correctness", "reward-aligned", "reward-aware", "misleads",
            "rollout filtering", "selective distillation", "verifiable rewards",
            "self-generated mistakes", "failure mode", "privileged teacher",
        ),
    ),
    "opd-warmup-gap": (
        "OPD 机理、SFT 热身、师生差距与遗忘",
        f"{ON_POLICY} AND {LANGUAGE_MODELS} AND " + title_or_abstract(
            "warm-up", "warmup", "cold start", "supervised fine-tuning", "SFT",
            "teacher-student gap", "distribution mismatch", "distribution shift",
            "catastrophic forgetting", "out-of-distribution", "reverse KL",
            "teacher evolution",
        ) + ' OR ti:"Simple-OPD" OR ti:"Rethinking On-Policy Distillation"',
    ),
    "opsd-thinking": (
        "OPSD、教师演化与思考模型蒸馏",
        f"({ON_POLICY} AND " + title_or_abstract(
            "thinking", "reasoning", "teacher evolution", "without any supervision",
            "self-teacher", "chain-of-thought", "mode mismatch",
        ) + ")",
    ),
}
TOPIC_ALIASES = {
    "dynamic-steering": "geometry-dynamic-steering",
    "cross-tokenizer": "cross-tokenizer-opd",
    "correctness": "opd-correctness",
    "opd-warmup": "opd-warmup-gap",
}

SAFETY_NOISE_RE = re.compile(
    r"\b(jailbreak|refusal|safety|harmful|toxic|detoxif|red-team|sycophancy|persona|bias|fairness|privacy|hallucinat)\w*",
    re.IGNORECASE,
)
KEEP_METHOD_RE = re.compile(
    r"\b(distill|opd|opsd|slerp|spherical|angular|geodesic|norm-preserving|collateral|anneal|training signal|teacher)\w*",
    re.IGNORECASE,
)


def wrap_query(raw_query):
    return f"({raw_query}) AND {CAT_FILTER} ANDNOT {EXCLUDE_TITLE}"


def is_safety_noise(paper):
    title = paper["title"]
    return bool(SAFETY_NOISE_RE.search(title) and not KEEP_METHOD_RE.search(title))


def query_url(query, per_query, sort, start=0):
    return API_URL + "?" + urlencode({
        "search_query": wrap_query(query),
        "start": start,
        "max_results": per_query,
        "sortBy": "submittedDate" if sort == "newest" else "relevance",
        "sortOrder": "descending",
    })


def fetch_feed(url):
    request = Request(url, headers={
        "Accept": "application/atom+xml",
        "User-Agent": "Steer-OPD-Literature-Search/1.0",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                return ElementTree.parse(response).getroot()
        except (URLError, TimeoutError) as error:
            delay = REQUEST_INTERVAL * 2**attempt
            if isinstance(error, HTTPError):
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = error.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            if attempt == 2:
                raise
            time.sleep(delay)


def parse_papers(feed):
    if feed.tag != ATOM + "feed":
        raise ValueError("arXiv 返回了非 Atom 搜索结果")
    for entry in feed.findall(ATOM + "entry"):
        path = urlsplit(entry.findtext(ATOM + "id", "")).path
        title = " ".join(entry.findtext(ATOM + "title", "").split())
        if not path.startswith("/abs/"):
            reason = " ".join(entry.findtext(ATOM + "summary", title).split())
            raise ValueError(reason)
        paper_id = re.sub(r"v\d+$", "", path.removeprefix("/abs/"))
        yield {
            "arxiv_id": paper_id,
            "title": title,
            "published": entry.findtext(ATOM + "published", "")[:10],
            "abstract": " ".join(entry.findtext(ATOM + "summary", "").split()),
            "score": 0.0,
        }


def iter_papers(query, per_query, sort, all_results):
    start = 0
    while True:
        feed = fetch_feed(query_url(query, per_query, sort, start))
        page = list(parse_papers(feed))
        yield from page
        if not all_results:
            return

        page_start = int(feed.findtext(OPENSEARCH + "startIndex", "-1"))
        total = int(feed.findtext(OPENSEARCH + "totalResults", "-1"))
        if page_start != start or total < start + len(page):
            raise ValueError(f"arXiv 返回了不一致的分页信息，起始位置 {start}")
        start += len(page)
        if start == total:
            return
        if not page:
            raise ValueError(f"arXiv 在取完 {total} 条结果前返回了空页，起始位置 {start}")
        time.sleep(REQUEST_INTERVAL)


def search(queries, per_query, sort, all_results=False):
    papers = {}
    for number, (label, query) in enumerate(queries):
        if number:
            time.sleep(REQUEST_INTERVAL)
        try:
            fetched = list(iter_papers(query, per_query, sort, all_results))
            results = [
                (rank, paper)
                for rank, paper in enumerate(fetched, 1)
                if not is_safety_noise(paper)
            ]
        except (URLError, TimeoutError, ValueError, ElementTree.ParseError) as error:
            raise RuntimeError(f"{label} 的检索失败。{error}") from error
        print(f"{label} · 取回 {len(fetched)} 篇，保留 {len(results)} 篇", file=sys.stderr)
        seen_ids = set()
        for rank, result in results:
            paper_id = result["arxiv_id"]
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            # Reciprocal rank fusion across the selected arXiv queries.
            papers.setdefault(paper_id, result)["score"] += 1 / (60 + rank)

    sort_fields = ("published", "score") if sort == "newest" else ("score", "published")
    return sorted(papers.values(), key=lambda paper: tuple(paper[field] for field in sort_fields),
                  reverse=True)


def format_report(papers):
    blocks = [
        f'Title: {paper["title"]}\n'
        f'ArXiv ID: {paper["arxiv_id"]}\n'
        f'Date: {paper["published"]}\n'
        f'Abstract: {paper["abstract"]}'
        for paper in papers
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description="检索与动态教师 SV 和 OPD 相关的 arXiv 论文。")
    parser.add_argument("--topic", action="append", choices=TOPICS,
                        type=lambda name: TOPIC_ALIASES.get(name, name),
                        help="限定主题，可重复指定。默认搜索全部主题。")
    parser.add_argument("--query", action="append", help="自定义 arXiv 查询式，可重复指定。")
    parser.add_argument("--since", type=date.fromisoformat, help="最早提交日期，YYYY-MM-DD。")
    parser.add_argument("--until", type=date.fromisoformat, help="最晚提交日期，YYYY-MM-DD。")
    parser.add_argument("--all", dest="all_results", action="store_true",
                        help="逐页取完每个查询的结果，输出全部去重后的论文。")
    parser.add_argument("--per-query", type=positive_int, default=20,
                        help="每个查询取回的篇数。全量模式下为每页篇数。默认 20。")
    parser.add_argument("--limit", type=positive_int, default=25,
                        help="普通模式去重后最多输出的篇数。默认 25。")
    parser.add_argument("--sort", choices=("relevance", "newest"), default="relevance")
    parser.add_argument("-o", "--output", type=Path,
                        help="保存到 UTF-8 .txt 文件。未指定时写到标准输出。")
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() != ".txt":
        parser.error("--output 需要使用 .txt 文件名")

    selected = args.topic or (() if args.query else TOPICS)
    queries = [TOPICS[topic] for topic in dict.fromkeys(selected)]
    queries.extend((f"自定义查询 {number}", query)
                   for number, query in enumerate(args.query or (), 1))
    if args.since or args.until:
        start = args.since or date(1991, 1, 1)
        end = args.until or datetime.now(timezone.utc).date()
        if start > end:
            parser.error("--since 必须早于或等于 --until")
        date_range = f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
        queries = [(label, f"({query}) AND {date_range}") for label, query in queries]

    try:
        papers = search(queries, args.per_query, args.sort, all_results=args.all_results)
    except RuntimeError as error:
        parser.exit(1, str(error) + "\n")
    if not args.all_results:
        papers = papers[:args.limit]
    if not papers:
        print("未找到符合条件的论文。", file=sys.stderr)
    report = format_report(papers)
    if args.output is None:
        sys.stdout.write(report)
        return
    try:
        with TemporaryDirectory(dir=args.output.parent, prefix=".arxiv-report-") as directory:
            pending = Path(directory) / args.output.name
            pending.write_text(report, encoding="utf-8", newline="\n")
            pending.replace(args.output)
    except OSError as error:
        parser.exit(1, f"写入 {args.output} 失败。{error}\n")
    print(f"已保存 {len(papers)} 篇论文到 {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
