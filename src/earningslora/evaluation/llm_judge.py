"""LLM-as-judge using Gemini 2.5 Flash via Google AI Studio (free tier).

Compares two candidate summaries on faithfulness / completeness / conciseness /
style and returns structured per-rubric scores plus a winner. Calls are cached
to a SQLite blob keyed by (transcript, summary_a, summary_b, rubric_version) so
reruns don't re-spend the daily quota.

Stub — implemented in Weekend 2 (harness) and Weekend 4 (bench wiring).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeVerdict:
    winner: str  # "a" | "b" | "tie"
    scores_a: dict[str, int]
    scores_b: dict[str, int]
    rationale: str


def judge(transcript: str, summary_a: str, summary_b: str) -> JudgeVerdict:
    """Score two summaries against the rubric defined in `utils.prompts`."""
    raise NotImplementedError("Implemented in Weekend 2 — judge harness.")
