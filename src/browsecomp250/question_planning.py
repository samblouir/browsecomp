from __future__ import annotations

import itertools
import re
import unicodedata
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9]+(?:[.'’-][A-Za-z0-9]+)*")
_GENERIC_LEADS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "during",
    "each",
    "for",
    "from",
    "give",
    "has",
    "have",
    "identify",
    "in",
    "is",
    "it",
    "less",
    "name",
    "of",
    "on",
    "once",
    "provide",
    "represents",
    "restored",
    "that",
    "the",
    "there",
    "these",
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
_STOPWORDS = _GENERIC_LEADS | {
    "about",
    "after",
    "again",
    "against",
    "all",
    "also",
    "among",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "into",
    "its",
    "me",
    "more",
    "most",
    "my",
    "not",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "she",
    "some",
    "such",
    "than",
    "their",
    "them",
    "then",
    "they",
    "those",
    "through",
    "under",
    "until",
    "we",
    "while",
    "whose",
    "would",
    "you",
    "your",
}
_WEAK_CONTENT_WORDS = {
    "answer",
    "article",
    "city",
    "community",
    "country",
    "different",
    "first",
    "following",
    "found",
    "location",
    "month",
    "particular",
    "person",
    "place",
    "question",
    "report",
    "same",
    "somewhere",
    "time",
    "town",
    "year",
}
_QUESTION_SCAFFOLD = re.compile(
    r"(?:^|[.!?]\s*)(?:what|which|who|where|when|how|give|identify|name|provide)\b|"
    r"\b(?:what|which|who|where|when|how)\b|\b(?:give|identify|provide)\s+(?:the\s+)?"
    r"(?:name|title|date|time|number|percentage)\b|\bi wonder\b",
    re.I,
)


def _space(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def _words(text: str) -> list[str]:
    return _WORD.findall(_space(text))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = _space(raw).strip(" ,;:.?\"'()[]")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _tail_question(question: str) -> str:
    text = _space(question)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if sentences and sentences[-1].endswith("?"):
        return sentences[-1]
    for sentence in reversed(sentences):
        if _QUESTION_SCAFFOLD.search(sentence):
            return sentence
    return sentences[-1] if sentences else text


def infer_answer_contract(
    question: str,
    *,
    fallback_type: Any = None,
    fallback_cardinality: Any = None,
) -> dict[str, Any]:
    """Infer the requested output from the terminal ask, not incidental clue words."""
    text = _space(question)
    tail = _tail_question(text)
    q = tail.casefold()
    whole = text.casefold()
    answer_type = str(fallback_type or "other_short_string")
    reason = "guide fallback; no unambiguous terminal answer-form cue"

    rules: list[tuple[str, str]] = [
        (r"\b(?:12|24)[- ]hour clock\b|\bwhat time\b|\bwhich time\b", "time"),
        (r"\bwhat percentage\b|\bpercentage (?:of|was|is)\b|\bpercent\b", "percentage"),
        (r"\bhow many\b|\bnumber of\b|\bwhat number\b", "number"),
        (r"\bwhat year\b|\bwhich year\b|\byear (?:was|did|in which)\b", "year"),
        (r"\bwhat date\b|\bwhich date\b|\bwhat day\b|\bwhich day\b", "date"),
        (
            r"\bfirst and last name\b|\bfirst and surname\b|\bname and surname\b|"
            r"\bwho (?:was|is|were|are|wrote|won|founded|created|designed|presented)\b",
            "person",
        ),
        (
            r"\bname (?:and|&) surname\b.*\byear\b|\bname of .*\b(?:founder|person)\b.*\bborn\b",
            "person_and_birth_year",
        ),
        (
            r"\bname of (?:the )?(?:mystery )?(?:settlement|city|town|village|country|island|"
            r"river|mountain|location|place)\b|\bwhich (?:settlement|city|town|village|country|"
            r"island|river|mountain|location|place)\b",
            "place",
        ),
        (r"\bname the monument\b|\bname of (?:the )?monument\b", "monument_name"),
        (
            r"\b(?:title|name) of (?:the )?(?:book|film|movie|song|album|episode|paper|study|"
            r"article|report|painting|game|work)\b|\bwhat (?:book|film|movie|song|album|episode)\b",
            "work_title",
        ),
        (r"\bwhere (?:was|is|were|are|did)\b", "place"),
        (r"\bwho\b", "person"),
    ]
    # Compound person-and-year asks must win over the simpler person rule.
    compound = rules.pop(6)
    rules.insert(0, compound)
    for pattern, inferred in rules:
        if re.search(pattern, q, re.I):
            answer_type = inferred
            reason = f"terminal question matches {pattern}"
            break
    else:
        if "12-hour clock" in whole or "24-hour clock" in whole:
            answer_type = "time"
            reason = "explicit clock-format instruction"

    cardinality = fallback_cardinality
    if not isinstance(cardinality, dict):
        cardinality = {"minimum": 1, "maximum": 1, "ordered": False}
    else:
        cardinality = dict(cardinality)
    cardinality.setdefault("minimum", 1)
    cardinality.setdefault("maximum", 1)
    cardinality.setdefault("ordered", False)
    if re.search(r"\b(?:list|all|which) (?:names|people|countries|cities|works|items)\b", q):
        cardinality["maximum"] = None
    elif re.search(r"\b(?:both|pair|two names|two people)\b", q):
        cardinality.update(minimum=2, maximum=2)
    return {
        "answer_type": answer_type,
        "answer_cardinality": cardinality,
        "answer_type_inference": reason,
        "terminal_ask": tail,
    }


def _information_tokens(text: str) -> list[str]:
    return [
        word
        for word in _words(text)
        if word.casefold() not in _STOPWORDS and len(word) > 1
    ]


def _phrase_score(text: str) -> float:
    words = _information_tokens(text)
    if not words:
        return -100.0
    score = 0.0
    for word in words:
        lower = word.casefold()
        if any(char.isdigit() for char in word):
            score += 6.0
        elif lower in _WEAK_CONTENT_WORDS:
            score += 0.6
        elif len(word) >= 9:
            score += 3.2
        elif len(word) >= 6:
            score += 2.3
        else:
            score += 1.3
        if word[:1].isupper() and lower not in _GENERIC_LEADS:
            score += 0.8
    if len(words) >= 2:
        score += min(len(words), 6) * 0.8
    if len(words) == 1 and not any(char.isdigit() for char in words[0]):
        score -= 4.0
    return score


def _is_informative_phrase(text: str) -> bool:
    words = _information_tokens(text)
    if len(words) < 2:
        return False
    if all(word.casefold() in _WEAK_CONTENT_WORDS for word in words):
        return False
    first = _words(text)
    return not (len(first) == 1 and first[0].casefold() in _GENERIC_LEADS)


def _segments(question: str) -> list[str]:
    text = _space(question)
    parts = re.split(
        r"(?<=[.!?;])\s+|\s+-\s+(?=[A-Z])|\s+(?=\d+[.)]\s+)",
        text,
    )
    return [part.strip(" -") for part in parts if len(_information_tokens(part)) >= 2]


def _compressed_segment(segment: str) -> str:
    values = _information_tokens(segment)
    if len(values) > 9:
        ranked = sorted(
            enumerate(values),
            key=lambda cell: (_phrase_score(cell[1]), -cell[0]),
            reverse=True,
        )[:9]
        values = [value for _, value in sorted(ranked)]
    return " ".join(values)


def _proper_phrases(question: str) -> list[str]:
    # Include lowercase connectors inside names while requiring at least two real name words.
    connector = r"(?:of|the|and|for|in|on|at|to|&|de|la|van|von)"
    name = r"[A-Z][A-Za-z0-9'’&-]*"
    pattern = re.compile(rf"\b{name}(?:\s+(?:{connector}\s+)?{name}){{1,7}}\b")
    result = []
    for match in pattern.finditer(question):
        value = match.group(0)
        tokens = _words(value)
        if tokens and tokens[0].casefold() in _GENERIC_LEADS:
            tokens = tokens[1:]
            value = " ".join(tokens)
        if _is_informative_phrase(value):
            result.append(value)
    return _unique(result)


def _quote(value: str) -> str:
    escaped = value.replace('"', " ")
    return f'"{_space(escaped)}"'


def _query_key(value: str) -> str:
    return " ".join(word.casefold() for word in _information_tokens(value))


def _dedupe_queries(values: list[str], *, limit: int = 7) -> list[str]:
    result: list[str] = []
    keys: list[set[str]] = []
    for value in values:
        normalized = _space(value)
        key = set(_query_key(normalized).split())
        if len(key) < 2:
            continue
        if any(key == prior for prior in keys):
            continue
        result.append(normalized[:280])
        keys.append(key)
        if len(result) >= limit:
            break
    return result


def _source_profile(
    question: str,
    topic: str,
    route_targets: list[str],
    answer_type: str,
) -> tuple[list[str], list[str]]:
    q = question.casefold()
    targets: list[str] = []
    native_terms: list[str] = []

    rules = [
        (
            r"\b(?:interview|inquiry guide|article|journalist|opinion piece|foreword|introduction|"
            r"global report|published report|report (?:released|published|cover|credits))\b",
            ["original publisher pages and PDFs", "institutional publication archives", "web archives and contemporary reporting"],
            ["interview transcript inquiry guide archive", "publication PDF credits archive"],
        ),
        (
            r"\b(?:award|prize|medal|honou?r)\b",
            ["official award databases and recipient archives"],
            ["official award recipient archive"],
        ),
        (
            r"\b(?:study|co-author|journal|conference|section meeting|education sessions|proceedings|doi)\b",
            ["publisher and scholarly indexes", "conference programs and professional biographies"],
            ["conference program proceedings presenter", "study authors institutional profile"],
        ),
        (
            r"\b(?:students?\b.{0,80}\btour(?:ed)?|tour(?:ed)?\b.{0,80}\bstudents?|"
            r"first stop|itinerary)\b",
            ["university course and studio archives", "tour itineraries, event pages, and participant reports"],
            ["student tour itinerary first stop", "university course visit schedule"],
        ),
        (
            r"\b(?:university|college|degree|education|course)\b",
            ["university and institutional archives"],
            ["university institutional biography archive"],
        ),
        (
            r"\b(?:restaurant|hotel|museum|store|business)\b",
            ["official venue histories and business profiles", "local reporting and directories"],
            ["official history founder", "local business profile archive"],
        ),
        (
            r"\b(?:monument|memorial|heritage|restored|ironworks|world war|political dynasty)\b",
            ["municipal heritage registers", "war memorial and monument databases", "local-history and event archives"],
            ["heritage register monument restoration", "war memorial inscription local history"],
        ),
        (
            r"\b(?:distance|miles?|kilomet(?:er|re)|meters?|metres?|route|bicycle|train station|body of water|street)\b",
            ["maps, routing, and geospatial records", "municipal and national geographic records"],
            ["map route distance municipal record"],
        ),
        (
            r"\b(?:fire|arson|police|hospital|heliport|preserved area)\b",
            ["government incident records", "local news archives", "hospital and community records"],
            ["incident report local archive", "hospital community record"],
        ),
        (
            r"\b(?:cover designer|graphic design|publishing|worked at|portfolio)\b",
            ["designer portfolios and professional biographies", "report colophons and publisher credits"],
            ["designer portfolio biography", "report cover credits PDF"],
        ),
    ]
    for pattern, additions, terms in rules:
        if re.search(pattern, q):
            targets.extend(additions)
            native_terms.extend(terms)

    priority_patterns = {
        "monument_name": ("heritage", "monument", "memorial", "local-history"),
        "place": ("maps", "geographic", "municipal", "incident", "local news"),
        "person": ("biograph", "award", "conference", "publisher", "portfolio"),
        "person_and_birth_year": ("biograph", "official venue", "local reporting"),
        "time": ("itinerar", "event", "publication", "archive"),
        "work_title": ("publisher", "publication", "catalog"),
    }
    priorities = priority_patterns.get(answer_type, ())
    if priorities:
        targets.sort(
            key=lambda value: not any(
                pattern in value.casefold() for pattern in priorities
            )
        )

    if not targets:
        targets.extend(str(value) for value in route_targets if str(value).strip())
    if not targets:
        targets.extend(
            ["official or primary records", "specialist databases", "contemporary archives"]
        )
    if not native_terms:
        native_terms.append(f"{topic.casefold()} official archive record")
    return _unique(targets)[:10], _unique(native_terms)[:6]


def compile_question_discovery_profile(
    question: str,
    *,
    topic: str,
    route_question_model: dict[str, Any],
    route_queries: list[list[str]],
) -> dict[str, Any]:
    """Create answer-independent, high-information retrieval seeds from the question."""
    answer_contract = infer_answer_contract(
        question,
        fallback_type=route_question_model.get("answer_type"),
        fallback_cardinality=route_question_model.get("answer_cardinality"),
    )
    quoted = re.findall(r'["“]([^"”]{4,120})["”]', question)
    proper = _proper_phrases(question)
    compressed = [_compressed_segment(segment) for segment in _segments(question)]
    route_anchors = [str(value) for value in route_question_model.get("lexical_anchors") or []]
    normalized_question = _space(question).casefold()
    exact_anchors = [
        value
        for value in _unique(quoted + proper + route_anchors)
        if _is_informative_phrase(value)
        and _space(value).casefold() in normalized_question
    ]
    anchors = [
        value
        for value in _unique(exact_anchors + compressed)
        if _is_informative_phrase(value)
    ]
    anchors.sort(key=lambda value: (_phrase_score(value), len(value)), reverse=True)
    anchors = anchors[:14]

    targets, native_terms = _source_profile(
        question,
        topic,
        [str(value) for value in route_question_model.get("source_targets") or []],
        answer_contract["answer_type"],
    )

    exact_keys = {_space(value).casefold() for value in exact_anchors}

    def query_fragment(value: str) -> str:
        return _quote(value) if _space(value).casefold() in exact_keys else value

    exact_candidates = [_quote(value) for value in exact_anchors if len(_words(value)) <= 7]
    exact_candidates.extend(compressed)
    rung1 = _dedupe_queries(exact_candidates, limit=7)

    pair_candidates = []
    for left, right in itertools.combinations(anchors[:8], 2):
        if set(_query_key(left).split()) & set(_query_key(right).split()):
            continue
        pair_candidates.append(f"{query_fragment(left)} {query_fragment(right)}")
    rung2 = _dedupe_queries(pair_candidates, limit=7)

    source_candidates = [
        f"{query_fragment(anchor)} {native}"
        for anchor, native in itertools.product(anchors[:5], native_terms[:4])
    ]
    rung3 = _dedupe_queries(source_candidates, limit=7)

    # Preserve guide seeds only when they are independently information-bearing.
    generated = [rung1, rung2, rung3]
    for index, legacy_rung in enumerate(route_queries[:3]):
        additions = [
            str(value)
            for value in legacy_rung
            if _is_informative_phrase(re.sub(r"\bsite:\S+", "", str(value)))
            and _phrase_score(re.sub(r"\bsite:\S+", "", str(value))) >= 4.0
        ]
        generated[index] = _dedupe_queries(generated[index] + additions, limit=7)

    if not rung1:
        fallback = " ".join(_information_tokens(question)[:10])
        generated[0] = _dedupe_queries([fallback], limit=7)
    if not generated[1]:
        generated[1] = _dedupe_queries(generated[0][1:] + generated[0][:1], limit=7)
    if not generated[2]:
        generated[2] = _dedupe_queries(
            [f"{query} official archive" for query in generated[0]], limit=7
        )

    return {
        **answer_contract,
        "question_first_anchors": anchors,
        "source_targets": targets,
        "source_native_terms": native_terms,
        "query_rungs": generated,
        "route_metadata_role": "secondary_hint_only",
    }


__all__ = ["compile_question_discovery_profile", "infer_answer_contract"]
