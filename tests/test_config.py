from pathlib import Path

import pytest

from browsecomp250.config import load_config
from browsecomp250.util import expand_env


def test_expand_env_accepts_json_object_default(monkeypatch) -> None:
    monkeypatch.delenv("BC250_TEST_HEADERS", raising=False)
    assert expand_env("${BC250_TEST_HEADERS:-{}}") == "{}"
    monkeypatch.setenv("BC250_TEST_HEADERS", '{"X-Test":"yes"}')
    assert expand_env("${BC250_TEST_HEADERS:-{}}") == '{"X-Test":"yes"}'


def test_env_expansion(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_MODEL", "unit-model")
    config = tmp_path / "config.yaml"
    subset = tmp_path / "indices.json"
    subset.write_text("[]")
    config.write_text(
        """
run: {name: test, output_dir: runs}
dataset:
  source_url: https://example.test/data.csv
  cache_dir: ~/.cache/test
  expected_rows: 1266
  subset_indices_path: data/subset_indices.json
  subset_size: 250
model:
  api_base: http://localhost:8000/v1
  api_key: key
  model: ${TEST_MODEL:-fallback}
search: {provider: searxng}
browser: {}
agent: {}
external_model:
  enabled: true
  mode: agent
  admin_token: external-secret
  agent_api_key: star-agent-secret
grader: {mode: deterministic}
report: {}
"""
    )
    parsed = load_config(config)
    assert parsed.model.model == "unit-model"
    assert parsed.model.api_base == "http://localhost:8000/v1"


def test_public_config_redacts_only_secrets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_MODEL", "unit-model")
    config = tmp_path / "config.yaml"
    config.write_text(
        """
run: {name: test, output_dir: runs}
dataset:
  source_url: https://example.test/data.csv
  cache_dir: ~/.cache/test
  expected_rows: 1266
  expected_sha256: ""
  subset_indices_path: data/subset_indices.json
  subset_size: 250
model:
  api_base: http://localhost:8000/v1
  api_key: top-secret
  allow_empty_api_key: false
  model: unit-model
  max_output_tokens: 16384
search: {provider: searxng}
browser: {}
agent: {}
external_model:
  enabled: true
  mode: agent
  admin_token: external-secret
  agent_api_key: star-agent-secret
grader: {mode: deterministic}
report: {}
"""
    )
    public = load_config(config).public_dict()
    assert public["model"]["api_key"] == "<redacted>"
    assert public["external_model"]["admin_token"] == "<redacted>"
    assert public["external_model"]["agent_api_key"] == "<redacted>"
    assert public["external_model"]["agent_model"] == "frontierrl/star-2"
    assert public["model"]["max_output_tokens"] == 16384
    assert public["model"]["allow_empty_api_key"] is False


@pytest.mark.parametrize(
    ("name", "strategy_recovery", "max_calls", "automatic_requests", "rescue_seconds"),
    [
        ("star-dev-baseline.yaml", False, 3, 1, 0),
        ("star-smoke.yaml", False, 3, 1, 0),
        ("star-headline.yaml", True, 4, 2, 900),
    ],
)
def test_star_profiles_use_selective_star2_help(
    name: str,
    strategy_recovery: bool,
    max_calls: int,
    automatic_requests: int,
    rescue_seconds: int,
) -> None:
    parsed = load_config(Path(__file__).parents[1] / "configs" / name)
    assert parsed.agent.automatic_external_requests == automatic_requests
    assert parsed.agent.automatic_external_strategy_recovery is strategy_recovery
    assert parsed.agent.automatic_finalization_rescue_after_seconds == rescue_seconds
    assert parsed.external_model.mode == "agent"
    assert parsed.external_model.agent_model == "frontierrl/star-2"
    assert parsed.external_model.max_calls_per_task == max_calls
    assert parsed.external_model.max_concurrency == 32
    assert parsed.external_model.max_output_tokens == 16384
    assert parsed.model.api_base == "http://127.0.0.1:8003/v1"
    assert parsed.model.response_chain is False
    assert parsed.external_model.agent_api_base == "http://127.0.0.1:8003/v1"
    assert parsed.external_model.agent_response_chain is False
    if name == "star-headline.yaml":
        assert parsed.agent.max_history_chars == 300000


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    import pytest
    from pydantic import ValidationError

    config = tmp_path / "config.yaml"
    config.write_text(
        """
run: {name: test, output_dir: runs}
dataset:
  expected_rows: 1266
  subset_indices_path: data/subset_indices.json
  subset_size: 250
model:
  api_base: http://localhost:8000/v1
  model: unit-model
  misspelled_temperature: 0.5
search: {provider: searxng}
browser: {}
agent: {}
grader: {mode: deterministic}
report: {}
"""
    )
    with pytest.raises(ValidationError, match="misspelled_temperature"):
        load_config(config)
