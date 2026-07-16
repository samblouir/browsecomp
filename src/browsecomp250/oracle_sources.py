from __future__ import annotations

import asyncio
import hashlib
import re
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

from .agent.runner import AgentRunner
from .browser.fetcher import PageFetcher
from .question_planning import compile_question_discovery_profile
from .search.base import SearchProvider
from .types import PageDocument, SearchResult

_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_COMPOSITE_SEPARATOR = re.compile(r"\s*(?:[,;|]|\band\b)\s*", re.I)
_LOW_VALUE_HOSTS = {
    "brainly.com",
    "etsy.com",
    "facebook.com",
    "instagram.com",
    "pinterest.com",
    "quora.com",
    "scribd.com",
    "tiktok.com",
    "tripadvisor.com",
    "x.com",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "between",
    "by",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "less",
    "more",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "to",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_WEAK_EVIDENCE_TERMS = {
    "12-hour",
    "after",
    "all",
    "another",
    "answer",
    "article",
    "before",
    "clock",
    "clues",
    "community",
    "different",
    "directly",
    "each",
    "first",
    "format",
    "following",
    "further",
    "go",
    "hour",
    "included",
    "least",
    "last",
    "local",
    "location",
    "mentioned",
    "name",
    "next",
    "one",
    "other",
    "particular",
    "person",
    "previous",
    "previously",
    "provide",
    "pulled",
    "question",
    "questions",
    "report",
    "some",
    "than",
    "there",
    "those",
    "three",
    "time",
    "together",
    "two",
    "using",
    "within",
    "year",
}
_COMMON_NUMERIC_ANSWER = re.compile(
    r"^(?:\d{1,4}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|"
    r"\d+(?:\.\d+)?\s*(?:percent|%|years?|miles?|km|meters?|metres?))$",
    re.I,
)


def _space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(_space(value))
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def _question_evidence_terms(question: str, aliases: list[str]) -> set[str]:
    alias_terms = set().union(*(_tokens(alias) for alias in aliases)) if aliases else set()
    return {
        token
        for token in _tokens(question)
        if token not in alias_terms and token not in _WEAK_EVIDENCE_TERMS
    }


def _is_common_numeric_answer(aliases: list[str]) -> bool:
    return any(_COMMON_NUMERIC_ANSWER.fullmatch(_space(alias)) for alias in aliases)


def _aliases(record: dict[str, Any]) -> list[str]:
    oracle = record.get("oracle") or {}
    values = [oracle.get("gold_answer"), *(oracle.get("comparison_aliases") or [])]
    aliases = [_space(value) for value in values if _space(value)]
    return list(dict.fromkeys(aliases))


def _answer_search_forms(record: dict[str, Any]) -> list[str]:
    forms: list[str] = []
    for alias in _aliases(record):
        forms.append(alias)
        if not re.fullmatch(r"(?:\d{1,4}[:./-]?)+\s*(?:am|pm)?", alias, re.I):
            forms.extend(
                part
                for part in _COMPOSITE_SEPARATOR.split(alias)
                if len(_tokens(part)) >= 1 and len(part) >= 4
            )
    return list(dict.fromkeys(forms))[:8]


def _answer_match_aliases(record: dict[str, Any]) -> list[str]:
    """Match full labels plus meaningful composite identity components."""

    originals = _aliases(record)
    components = [
        value
        for value in _answer_search_forms(record)
        if value not in originals
        and len(_tokens(value)) >= 2
        and not _COMMON_NUMERIC_ANSWER.fullmatch(value)
    ]
    return list(dict.fromkeys([*originals, *components]))


def redact_private_answer(text: str, record: dict[str, Any]) -> str:
    redacted = str(text)
    for alias in sorted(
        list(dict.fromkeys([*_aliases(record), *_answer_search_forms(record)])),
        key=len,
        reverse=True,
    ):
        prefix = r"(?<!\w)" if alias[:1].isalnum() else ""
        suffix = r"(?!\w)" if alias[-1:].isalnum() else ""
        redacted = re.sub(prefix + re.escape(alias) + suffix, "${candidate}", redacted, flags=re.I)
    return redacted


def private_source_queries(record: dict[str, Any], *, limit: int = 12) -> list[str]:
    """Build private source-location queries; callers must never add them to transcripts."""

    item = record.get("item") or {}
    question = _space(item.get("question_text"))
    if not question:
        return []
    profile = compile_question_discovery_profile(
        question,
        topic=str(item.get("topic") or "Other"),
        route_question_model=record.get("question_model") or {},
        route_queries=[],
    )
    anchors = [
        " ".join(_space(value).split()[:8])
        for value in [
            *profile.get("question_first_anchors", []),
            *profile.get("source_native_terms", []),
        ]
        if len(_tokens(value)) >= 2
    ]
    oracle_queries = [
        _space(value)
        for value in (record.get("oracle") or {}).get(
            "answer_conditioned_verification_queries", []
        )
        if len(_tokens(value)) >= 3
    ]
    queries: list[str] = []
    for form in _answer_search_forms(record):
        quoted = f'"{form.replace(chr(34), " ")}"'
        for anchor in anchors[:8]:
            queries.append(f"{quoted} {anchor}")
        if len(queries) >= limit:
            break
    queries.extend(oracle_queries)
    return list(dict.fromkeys(queries))[:limit]


def _contains_alias(text: str, aliases: list[str]) -> bool:
    folded = _space(text).casefold()
    return any(alias.casefold() in folded for alias in aliases if alias)


def _host_penalty(url: str) -> float:
    host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    if host in _LOW_VALUE_HOSTS or any(host.endswith("." + value) for value in _LOW_VALUE_HOSTS):
        return 25.0
    return 0.0


def _search_score(result: SearchResult, *, aliases: list[str], question_terms: set[str]) -> float:
    text = " ".join([result.title, result.snippet, *result.extra_snippets])
    overlap = len(_tokens(text) & question_terms)
    return (
        (35.0 if _contains_alias(text, aliases) else 0.0)
        + min(overlap, 12) * 3.0
        - min(max(result.rank - 1, 0), 20) * 0.35
        - _host_penalty(result.url)
    )


def _page_score(
    document: PageDocument,
    *,
    seed: SearchResult,
    aliases: list[str],
    question_terms: set[str],
) -> float:
    text = f"{document.title} {_answer_window(document, aliases)}"
    overlap = len(_tokens(text) & question_terms)
    alias_present = _contains_alias(text, aliases)
    return (
        _search_score(seed, aliases=aliases, question_terms=question_terms)
        + (80.0 if alias_present else 0.0)
        + min(overlap, 18) * 2.0
        + min(len(document.text), 30_000) / 10_000
    )


def _answer_window(document: PageDocument, aliases: list[str], *, radius: int = 1_200) -> str:
    folded = document.text.casefold()
    positions = [folded.find(alias.casefold()) for alias in aliases if alias]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - radius)
    end = min(len(document.text), center + radius)
    return document.text[start:end]


async def discover_redacted_public_sources(
    record: dict[str, Any],
    *,
    search: SearchProvider,
    browser: PageFetcher,
    max_queries: int = 12,
    max_sources: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Locate answer-bearing public pages without exposing the private query or answer."""

    aliases = _answer_match_aliases(record)
    queries = private_source_queries(record, limit=max_queries)
    question = _space((record.get("item") or {}).get("question_text"))
    if not aliases or not queries or not question:
        return [], {"query_count": 0, "candidate_count": 0, "fetched_count": 0}

    batches = await search.search_many(queries, count=10)
    raw_searches: list[dict[str, Any]] = []
    candidates: dict[str, SearchResult] = {}
    question_terms = _question_evidence_terms(question, aliases)
    search_error_count = 0
    search_error_types: dict[str, int] = {}
    empty_search_batches = 0
    for query, batch in zip(queries, batches, strict=True):
        if isinstance(batch, Exception):
            search_error_count += 1
            error_name = type(batch).__name__
            search_error_types[error_name] = search_error_types.get(error_name, 0) + 1
            raw_searches.append({"query": query, "error": error_name})
            continue
        if not batch:
            empty_search_batches += 1
        raw_searches.append(
            {"query": query, "results": [value.as_prompt_dict() for value in batch]}
        )
    filtered = AgentRunner._filter_query_mirror_search_results(
        question,
        {"ok": True, "searches": raw_searches},
    )
    for search_result in filtered.get("searches") or []:
        if not isinstance(search_result, dict):
            continue
        for row in search_result.get("results") or []:
            if not isinstance(row, dict) or not row.get("url"):
                continue
            candidate = SearchResult(
                title=str(row.get("title") or ""),
                url=str(row["url"]),
                snippet=str(row.get("snippet") or ""),
                rank=int(row.get("rank") or 0),
                source=str(row.get("source") or ""),
                extra_snippets=[str(value) for value in row.get("extra_snippets") or []],
            )
            prior = candidates.get(candidate.url)
            if prior is None or _search_score(
                candidate, aliases=aliases, question_terms=question_terms
            ) > _search_score(prior, aliases=aliases, question_terms=question_terms):
                candidates[candidate.url] = candidate

    ranked = sorted(
        candidates.values(),
        key=lambda value: _search_score(
            value,
            aliases=aliases,
            question_terms=question_terms,
        ),
        reverse=True,
    )[: max(max_sources * 4, 12)]
    fetched = await asyncio.gather(
        *(browser.fetch(value.url) for value in ranked),
        return_exceptions=True,
    )
    verified: list[tuple[float, SearchResult, PageDocument]] = []
    rejected_for_local_context = 0
    minimum_local_overlap = 3 if _is_common_numeric_answer(aliases) else 2
    for seed, document in zip(ranked, fetched, strict=True):
        if isinstance(document, Exception):
            continue
        public_text = f"{document.title} {document.text}"
        if not _contains_alias(public_text, aliases):
            continue
        if _host_penalty(document.final_url or document.requested_url) > 0:
            continue
        # Measure clues near the candidate mention, not over the whole page. A
        # long unrelated page otherwise passes by accumulating stray words.
        local_text = f"{document.title} {_answer_window(document, aliases)}"
        local_overlap = len(_tokens(local_text) & question_terms)
        if local_overlap < minimum_local_overlap:
            rejected_for_local_context += 1
            continue
        verified.append(
            (
                _page_score(
                    document,
                    seed=seed,
                    aliases=aliases,
                    question_terms=question_terms,
                ),
                seed,
                document,
            )
        )

    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_hosts: set[str] = set()
    for _, _, document in sorted(verified, key=lambda value: value[0], reverse=True):
        url = document.final_url or document.requested_url
        normalized = url.rstrip("/").casefold()
        if normalized in seen_urls:
            continue
        host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
        if host in seen_hosts and len(selected) >= 2:
            continue
        seen_urls.add(normalized)
        seen_hosts.add(host)
        source_id = "BOOT-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        selected.append(
            {
                "source_id": source_id,
                "role": "answer-redacted public source",
                "title": redact_private_answer(document.title, record)[:500],
                "url": url,
                "overview_excerpt": redact_private_answer(
                    _answer_window(document, aliases),
                    record,
                )[:2_400],
            }
        )
        if len(selected) >= max_sources:
            break
    return selected, {
        "query_count": len(queries),
        "search_error_count": search_error_count,
        "search_error_types": search_error_types,
        "empty_search_batches": empty_search_batches,
        "candidate_count": len(candidates),
        "fetched_count": sum(not isinstance(value, Exception) for value in fetched),
        "verified_count": len(verified),
        "minimum_local_clue_overlap": minimum_local_overlap,
        "rejected_for_local_context": rejected_for_local_context,
        "selected_count": len(selected),
        "filtered_query_mirror_results": int(
            filtered.get("filtered_query_mirror_results") or 0
        ),
    }


def apply_redacted_source_cache(
    records: list[dict[str, Any]],
    cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    enriched = deepcopy(records)
    rows = cache.get("rows") if isinstance(cache, dict) else None
    if not isinstance(rows, dict):
        return enriched, 0
    applied = 0
    for record in enriched:
        item_id = str((record.get("item") or {}).get("item_id") or "")
        entry = rows.get(item_id)
        sources = entry.get("sources") if isinstance(entry, dict) else None
        oracle = record.setdefault("oracle", {})
        if oracle.get("evidence_sources") or not isinstance(sources, list) or not sources:
            continue
        oracle["evidence_sources"] = deepcopy(sources)
        applied += 1
    return enriched, applied
