from __future__ import annotations

from ..config import GraderConfig
from ..types import GradeResult
from .deterministic import grade_deterministic
from .official import OfficialLLMGrader


class Grader:
    def __init__(self, config: GraderConfig, official: OfficialLLMGrader | None = None):
        self.config = config
        self.official = official or (
            OfficialLLMGrader(config) if config.mode in {"official_llm", "both"} else None
        )
        self._owns_official = official is None

    async def close(self) -> None:
        if self.official is not None and self._owns_official:
            await self.official.close()

    async def grade(self, question: str, reference: str, response: str) -> GradeResult:
        deterministic = grade_deterministic(response, reference)
        if self.config.mode == "deterministic":
            return deterministic
        assert self.official is not None
        official = await self.official.grade(question, reference, response)
        if self.config.mode == "both":
            official.reasoning = (
                official.reasoning + f"\n[Diagnostic deterministic grade: {deterministic.correct}; "
                f"extracted={deterministic.extracted_answer!r}]"
            ).strip()
        return official


__all__ = ["Grader", "GradeResult", "OfficialLLMGrader", "grade_deterministic"]
