from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    DEFAULT_SUBSET_PATH,
    OFFICIAL_DATASET_ROWS,
    OFFICIAL_DATASET_URL,
    SUBSET_SIZE,
)
from .util import expand_env, expand_path, redact


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictConfigModel):
    name: str = "browsecomp250"
    output_dir: Path = Path("runs")
    seed: int = 0
    concurrency: int = Field(default=1, ge=1, le=256)
    attempts: int = Field(default=1, ge=1, le=64)
    shuffle: bool = True
    resume: bool = True
    fail_fast: bool = False
    task_timeout_seconds: float = Field(default=1800, gt=0)
    write_private_transcripts: bool = True


class DatasetConfig(StrictConfigModel):
    source_url: str = OFFICIAL_DATASET_URL
    cache_dir: Path = Path("~/.cache/browsecomp250")
    expected_rows: int = OFFICIAL_DATASET_ROWS
    expected_sha256: str = ""
    subset_indices_path: Path = DEFAULT_SUBSET_PATH
    subset_size: int = SUBSET_SIZE

    @model_validator(mode="after")
    def validate_fixed_subset(self) -> DatasetConfig:
        if self.expected_rows != OFFICIAL_DATASET_ROWS:
            raise ValueError(f"BrowseComp source must contain {OFFICIAL_DATASET_ROWS} rows")
        if self.subset_size != SUBSET_SIZE:
            raise ValueError(f"BrowseComp-250 requires subset_size={SUBSET_SIZE}")
        return self


class ModelConfig(StrictConfigModel):
    api_base: str
    api_key: str = ""
    allow_empty_api_key: bool = False
    model: str
    extra_headers_json: str | dict[str, str] = Field(default_factory=dict)
    protocol: Literal["json", "tools", "auto"] = "json"
    temperature: float | None = Field(default=0.3, ge=0.3)
    max_output_tokens: int = Field(default=16384, ge=16384)
    timeout_seconds: float = Field(default=300, gt=0)
    max_retries: int = Field(default=4, ge=0, le=20)
    response_chain: bool = False
    input_price_per_million: float = Field(default=0.0, ge=0)
    output_price_per_million: float = Field(default=0.0, ge=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_base")
    @classmethod
    def normalize_api_base(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value:
            raise ValueError("api_base is required")
        return value

    @field_validator("extra_headers_json", mode="before")
    @classmethod
    def parse_headers(cls, value: Any) -> dict[str, str]:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        parsed = json.loads(str(value))
        if not isinstance(parsed, dict):
            raise ValueError("extra_headers_json must decode to an object")
        return {str(k): str(v) for k, v in parsed.items()}


class SearchConfig(StrictConfigModel):
    provider: Literal[
        "brave",
        "google_chrome",
        "hybrid",
        "tavily",
        "serper",
        "searxng",
    ] = "brave"
    results_per_call: int = Field(default=10, ge=1, le=20)
    country: str = "US"
    language: str = "en"
    safe_search: Literal["off", "moderate", "strict"] = "moderate"
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=4, ge=0, le=20)
    cache_mode: Literal["off", "read", "write", "readwrite", "refresh"] = "readwrite"
    cache_path: Path = Path("~/.cache/browsecomp250/search-cache.sqlite3")
    brave_api_key: str = ""
    tavily_api_key: str = ""
    serper_api_key: str = ""
    searxng_base_url: str = "http://127.0.0.1:8080"
    google_chrome_host: str = ""
    google_chrome_ssh_bin: str = "ssh"
    google_chrome_scp_bin: str = "scp"
    google_chrome_python_bin: str = "python3"
    google_chrome_cua_driver: str = "~/.star-hermes/bin/cua-driver"
    google_chrome_bundle_id: str = "com.google.Chrome"
    google_chrome_max_fanout: int = Field(default=8, ge=1, le=20)
    google_chrome_timeout_seconds: float = Field(default=45, gt=0, le=300)
    google_chrome_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    google_chrome_max_retries: int = Field(default=0, ge=0, le=3)
    hybrid_mode: Literal["merge", "google_first", "brave_first"] = "merge"

    def selected_api_key(self) -> str:
        return {
            "brave": self.brave_api_key,
            "google_chrome": "",
            "hybrid": self.brave_api_key,
            "tavily": self.tavily_api_key,
            "serper": self.serper_api_key,
            "searxng": "",
        }[self.provider]


class ExternalModelConfig(StrictConfigModel):
    enabled: bool = False
    api_url: str = "http://127.0.0.1:8000/api/external-model-requests"
    admin_token: str = ""
    default_provider: Literal[
        "chatgpt",
        "openai",
        "anthropic",
        "gemini",
        "openrouter",
        "mock",
    ] = "chatgpt"
    allowed_providers: list[str] = Field(
        default_factory=lambda: [
            "chatgpt",
            "openai",
            "anthropic",
            "gemini",
            "openrouter",
        ]
    )
    default_model: str = ""
    temperature: float = Field(default=0.7, ge=0.3, le=1.0)
    top_p: float = Field(default=0.95, gt=0, le=1.0)
    max_output_tokens: int = Field(default=16384, ge=16384, le=32768)
    max_calls_per_task: int = Field(default=8, ge=0, le=64)
    max_batch_size: int = Field(default=4, ge=1, le=4)
    max_concurrency: int = Field(default=4, ge=1, le=16)
    timeout_seconds: float = Field(default=900, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("api_url")
    @classmethod
    def normalize_api_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value:
            raise ValueError("external_model.api_url is required")
        return value


class BrowserConfig(StrictConfigModel):
    backend: Literal["direct", "playwright", "auto"] = "direct"
    timeout_seconds: float = Field(default=30, gt=0)
    max_redirects: int = Field(default=5, ge=0, le=20)
    max_response_bytes: int = Field(default=8_000_000, ge=100_000)
    max_text_chars_per_open: int = Field(default=30_000, ge=1_000)
    max_links_per_page: int = Field(default=80, ge=0, le=1000)
    user_agent: str = "BrowseComp250-ResearchRunner/0.1"
    block_private_networks: bool = True
    allow_nonstandard_ports: bool = False
    cache_mode: Literal["off", "read", "write", "readwrite", "refresh"] = "readwrite"
    cache_path: Path = Path("~/.cache/browsecomp250/page-cache.sqlite3")


class AgentConfig(StrictConfigModel):
    max_steps: int = Field(default=80, ge=1)
    max_search_calls: int = Field(default=40, ge=0)
    max_page_opens: int = Field(default=100, ge=0)
    max_find_calls: int = Field(default=80, ge=0)
    max_retrieved_chars: int = Field(default=2_000_000, ge=1_000)
    max_history_chars: int = Field(default=500_000, ge=10_000)
    parse_retries: int = Field(default=2, ge=0, le=10)
    require_citations: bool = True
    enable_search_many: bool = True
    enable_open_many: bool = True
    max_batch_size: int = Field(default=5, ge=1, le=20)
    automatic_external_after_search_calls: int = Field(default=0, ge=0)
    automatic_external_requests: int = Field(default=3, ge=1, le=4)
    system_prompt_path: Path | None = None


class GraderConfig(StrictConfigModel):
    mode: Literal["official_llm", "deterministic", "both"] = "official_llm"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    allow_empty_api_key: bool = False
    model: str = "gpt-5.6"
    extra_headers_json: str | dict[str, str] = Field(default_factory=dict)
    temperature: float | None = Field(default=0.3, ge=0.3)
    max_output_tokens: int = Field(default=16384, ge=16384)
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=4, ge=0, le=20)
    input_price_per_million: float = Field(default=0.0, ge=0)
    output_price_per_million: float = Field(default=0.0, ge=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_base")
    @classmethod
    def normalize_api_base(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("extra_headers_json", mode="before")
    @classmethod
    def parse_headers(cls, value: Any) -> dict[str, str]:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        parsed = json.loads(str(value))
        if not isinstance(parsed, dict):
            raise ValueError("extra_headers_json must decode to an object")
        return {str(k): str(v) for k, v in parsed.items()}


class ReportConfig(StrictConfigModel):
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_samples: int = Field(default=10_000, ge=100)
    write_html: bool = True
    write_csv: bool = True
    public_report_omits_questions: bool = True
    public_report_omits_answers: bool = True


class AppConfig(StrictConfigModel):
    run: RunConfig
    dataset: DatasetConfig
    model: ModelConfig
    search: SearchConfig
    browser: BrowserConfig
    agent: AgentConfig
    external_model: ExternalModelConfig = Field(default_factory=ExternalModelConfig)
    grader: GraderConfig
    report: ReportConfig
    config_path: Path | None = Field(default=None, exclude=True)

    def resolved_paths(self, base: Path) -> AppConfig:
        copy = self.model_copy(deep=True)
        copy.run.output_dir = expand_path(copy.run.output_dir, base)
        copy.dataset.cache_dir = expand_path(copy.dataset.cache_dir)
        copy.dataset.subset_indices_path = expand_path(copy.dataset.subset_indices_path, base)
        copy.search.cache_path = expand_path(copy.search.cache_path)
        copy.browser.cache_path = expand_path(copy.browser.cache_path)
        if copy.agent.system_prompt_path is not None:
            copy.agent.system_prompt_path = expand_path(copy.agent.system_prompt_path, base)
        return copy

    def public_dict(self) -> dict[str, Any]:
        return redact(self.model_dump(mode="json", exclude={"config_path"}))


def load_config(path: Path) -> AppConfig:
    path = path.resolve()
    raw = yaml.safe_load(expand_env(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    config = AppConfig.model_validate(raw)
    config.config_path = path
    # Project-relative paths are resolved from the repository working directory when
    # available, then from the config's parent as a fallback.
    cwd = Path.cwd().resolve()
    base = cwd if (cwd / "data" / "subset_indices.json").exists() else path.parent.parent
    return config.resolved_paths(base)
