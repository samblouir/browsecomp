from __future__ import annotations

import math
import re
import unicodedata

from ..types import GradeResult

_EXACT_ANSWER = re.compile(r"^\s*Exact Answer\s*:\s*(.+?)\s*$", re.I | re.M)
_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_exact_answer(response: str) -> str | None:
    match = _EXACT_ANSWER.search(response)
    if match:
        return match.group(1).strip()
    stripped = response.strip()
    return stripped if stripped and "\n" not in stripped else None


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    # Remove thousands separators before punctuation normalization so 1,000 and
    # 1000 remain numerically equivalent rather than becoming "1 000".
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", value)
    value = re.sub(r"[^\w\s.+-]", " ", value)
    value = re.sub(r"\b(?:a|an|the)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _single_number(value: str) -> float | None:
    matches = _NUMBER.findall(value)
    if len(matches) != 1:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def equivalent(predicted: str, reference: str) -> bool:
    left = normalize_answer(predicted)
    right = normalize_answer(reference)
    if left == right:
        return True
    left_number = _single_number(left)
    right_number = _single_number(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=1e-4, abs_tol=1e-6)
    return False


def grade_deterministic(response: str, reference: str) -> GradeResult:
    extracted = extract_exact_answer(response)
    correct = extracted is not None and equivalent(extracted, reference)
    return GradeResult(
        correct=correct,
        extracted_answer=extracted,
        reasoning=(
            "Normalized exact/numeric match." if correct else "No strict normalized equivalence."
        ),
        grader_mode="deterministic",
    )
