from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(slots=True)
class ModelResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    raw_message: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    response_model: str | None = None
    latency_seconds: float = 0.0
    response_id: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    source: str = ""
    extra_snippets: list[str] = field(default_factory=list)

    def as_prompt_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not self.extra_snippets:
            data.pop("extra_snippets", None)
        return data


@dataclass(slots=True)
class PageDocument:
    requested_url: str
    final_url: str
    title: str
    text: str
    content_type: str
    status_code: int
    links: list[dict[str, str]] = field(default_factory=list)
    fetched_at: str = ""
    sha256: str = ""
    truncated: bool = False


@dataclass(slots=True)
class AgentAction:
    action: Literal[
        "search",
        "search_many",
        "open",
        "open_many",
        "find",
        "ask_external_model",
        "note",
        "final",
    ]
    payload: dict[str, Any]


@dataclass(slots=True)
class AgentOutcome:
    response_text: str
    exact_answer: str | None
    explanation: str
    confidence: float | None
    citations: list[str]
    status: str
    steps: int
    search_calls: int
    page_opens: int
    find_calls: int
    retrieved_chars: int
    duration_seconds: float
    usage: Usage
    external_model_calls: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GradeResult:
    correct: bool
    extracted_answer: str | None
    reasoning: str
    grader_response: str = ""
    grader_mode: str = ""
    usage: Usage = field(default_factory=Usage)
    parse_error: str | None = None


@dataclass(slots=True)
class BenchmarkItem:
    item_id: str
    subset_rank: int
    source_index: int
    encrypted_row_hash: str
    question: str
    answer: str
    canary: str


@dataclass(slots=True)
class TrialRecord:
    schema_version: str
    run_id: str
    item_id: str
    subset_rank: int
    source_index: int
    attempt: int
    model: str
    status: str
    started_at: str
    finished_at: str
    answer_response: str
    extracted_answer: str | None
    explanation: str
    confidence: float | None
    citations: list[str]
    correct: bool | None
    grading: dict[str, Any] | None
    metrics: dict[str, Any]
    error: str | None = None
