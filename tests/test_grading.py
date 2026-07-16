import pytest

from browsecomp250.config import GraderConfig
from browsecomp250.grading.deterministic import equivalent, grade_deterministic
from browsecomp250.grading.official import OfficialLLMGrader
from browsecomp250.types import ModelResponse, Usage


class FakeClient:
    async def chat(self, messages):
        return ModelResponse(
            content=(
                "extracted_final_answer: Plastic Man\n"
                "reasoning: It matches the reference.\n"
                "correct: yes\nconfidence: 90"
            ),
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    async def close(self):
        return None


def test_deterministic_normalization() -> None:
    assert equivalent("The Plastic Man", "Plastic Man")
    assert equivalent("1,000", "1000")
    assert equivalent("1988-1996", "1988-96")
    assert equivalent("3:50 p.m.", "3:50 PM")
    assert equivalent("Republic of Ireland and Romania", "Ireland v Romania")
    assert not equivalent("Plastic Woman", "Plastic Man")
    assert not equivalent("Ireland and Romania", "Ireland v Bulgaria")


def test_deterministic_grade_extracts_line() -> None:
    result = grade_deterministic(
        "Explanation: evidence\nExact Answer: Plastic Man\nConfidence: 90%", "Plastic Man"
    )
    assert result.correct
    assert result.extracted_answer == "Plastic Man"


@pytest.mark.asyncio
async def test_official_grader_parses_yes() -> None:
    grader = OfficialLLMGrader(GraderConfig(mode="official_llm"), client=FakeClient())
    result = await grader.grade("question", "Plastic Man", "Exact Answer: Plastic Man")
    assert result.correct
    assert result.extracted_answer == "Plastic Man"
    assert result.usage.input_tokens == 10
