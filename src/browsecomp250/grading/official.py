from __future__ import annotations

import re

from ..config import GraderConfig
from ..llm.client import ClientSettings, OpenAICompatibleClient
from ..prompts import GRADER_TEMPLATE
from ..types import GradeResult

_CORRECT = re.compile(r"^\s*correct\s*:\s*(yes|no)\s*$", re.I | re.M)
_EXTRACTED = re.compile(r"^\s*extracted_final_answer\s*:\s*(.*?)\s*$", re.I | re.M)
_REASONING = re.compile(r"^\s*reasoning\s*:\s*(.*?)(?=^\s*correct\s*:|\Z)", re.I | re.M | re.S)


class OfficialLLMGrader:
    def __init__(self, config: GraderConfig, client: OpenAICompatibleClient | None = None):
        self.config = config
        self._owns_client = client is None
        self.client = client or OpenAICompatibleClient(
            ClientSettings(
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                extra_headers=dict(config.extra_headers_json or {}),
                extra_body=config.extra_body,
                input_price_per_million=config.input_price_per_million,
                output_price_per_million=config.output_price_per_million,
            )
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()

    async def grade(self, question: str, reference: str, response: str) -> GradeResult:
        prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=reference,
            response=response,
        )
        model_response = await self.client.chat([{"role": "user", "content": prompt}])
        text = model_response.content
        correct_match = _CORRECT.search(text)
        extracted_match = _EXTRACTED.search(text)
        reasoning_match = _REASONING.search(text)
        parse_error = None
        if not correct_match:
            correct = False
            parse_error = "Grader response omitted `correct: yes|no`; defaulted to incorrect"
        else:
            correct = correct_match.group(1).lower() == "yes"
        extracted = extracted_match.group(1).strip() if extracted_match else None
        if extracted and extracted.casefold() == "none":
            extracted = None
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return GradeResult(
            correct=correct,
            extracted_answer=extracted,
            reasoning=reasoning,
            grader_response=text,
            grader_mode=f"official_llm:{self.config.model}",
            usage=model_response.usage,
            parse_error=parse_error,
        )
