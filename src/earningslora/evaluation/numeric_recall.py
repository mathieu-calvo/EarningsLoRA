"""Numeric-recall: a finance-specific factuality metric.

Computes the fraction of numbers in a generated summary that also appear in the
source transcript. Penalises hallucinated figures — the most consequential
failure mode for a summariser fed to investment workflows.

Numbers are normalised before matching so that
    "$1.2 billion", "1.2B", "1,200,000,000", "1.2 bn"
all resolve to the same numeric value.

Out of scope (deliberately):
- percentages relative-vs-absolute (4% vs 4 percentage points) — too domain-noisy
- date/year handling — tracked separately if needed
- ranges (10-15) are decomposed into both endpoints

This is a *recall* metric: of the numbers the model emitted, how many are grounded?
A number in the transcript that the summary omitted is fine — summaries are lossy
by design. We care about not making things up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Regex captures things like:
#   "$1.2 billion", "1,234.5", "12.5%", "1.2bn", "$5M", "10-15", "15 million"
_NUMBER_RE = re.compile(
    r"""
    \$?                          # optional currency
    (?P<num>
        \d{1,3}(?:,\d{3})+       # 1,234 or 1,234,567
        (?:\.\d+)?               # .56
      | \d+\.\d+                 # 12.5
      | \d+                      # 12
    )
    \s*
    (?P<unit>
        billion|million|thousand|trillion|
        bn|mn|m|k|b|t|
        %|percent
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_UNIT_MULTIPLIERS = {
    "trillion": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
    "%": 1.0, "percent": 1.0,
}

# Tolerance for floating-point comparison after normalisation.
_REL_TOL = 1e-3


@dataclass(frozen=True)
class NumericRecallResult:
    recall: float
    matched: int
    total: int
    unmatched_examples: list[float]

    def __str__(self) -> str:
        return f"numeric_recall={self.recall:.3f} ({self.matched}/{self.total})"


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text, normalised by their unit."""
    out: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover — regex shouldn't allow this
            continue

        unit = (match.group("unit") or "").lower()
        if unit:
            multiplier = _UNIT_MULTIPLIERS.get(unit, 1.0)
            value *= multiplier

        out.append(value)
    return out


def _approx_in(value: float, haystack: list[float]) -> bool:
    """Membership with relative tolerance for float wobble."""
    for candidate in haystack:
        if value == 0.0 and candidate == 0.0:
            return True
        if value == 0.0 or candidate == 0.0:
            continue
        if abs(value - candidate) / max(abs(value), abs(candidate)) <= _REL_TOL:
            return True
    return False


def numeric_recall(transcript: str, summary: str) -> NumericRecallResult:
    """Fraction of numbers in `summary` that also appear in `transcript`.

    Returns 1.0 if the summary contains no numbers (vacuously grounded).
    """
    summary_nums = _extract_numbers(summary)
    if not summary_nums:
        return NumericRecallResult(recall=1.0, matched=0, total=0, unmatched_examples=[])

    transcript_nums = _extract_numbers(transcript)

    matched = 0
    unmatched: list[float] = []
    for value in summary_nums:
        if _approx_in(value, transcript_nums):
            matched += 1
        else:
            unmatched.append(value)

    total = len(summary_nums)
    return NumericRecallResult(
        recall=matched / total,
        matched=matched,
        total=total,
        unmatched_examples=unmatched[:5],
    )
