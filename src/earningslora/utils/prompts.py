"""Prompt templates used across SFT, frontier baseline, and LLM-as-judge.

Kept in one module so any change to the task definition (system prompt, output
contract, judge rubric) flips through a single file. All prompts are deterministic
strings; rendering uses `.format(...)` rather than f-strings so they remain
serialisable and diffable.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a buy-side equity analyst. Given an earnings call transcript, "
    "produce a concise bullet summary that captures the most decision-relevant "
    "information for an investor. Use 5-10 bullets. Be faithful to the transcript: "
    "do not invent figures, products, or guidance. Preserve numbers exactly as stated."
)

USER_PROMPT_TEMPLATE = (
    "Earnings call transcript:\n\n"
    "---\n"
    "{transcript}\n"
    "---\n\n"
    "Write the analyst-style bullet summary now."
)


JUDGE_RUBRIC = """\
You are evaluating two analyst summaries of the same earnings call transcript.
Score each on four rubrics on a 1-5 scale, then declare a winner.

Rubrics:
- faithfulness: every claim is supported by the transcript; no invented figures.
- completeness: covers the most decision-relevant items (revenue, guidance, risks, surprises).
- conciseness: information density per token; no padding.
- style: matches buy-side analyst register; bullet structure clean.

Return strict JSON:
{{
  "summary_a": {{"faithfulness": int, "completeness": int, "conciseness": int, "style": int}},
  "summary_b": {{"faithfulness": int, "completeness": int, "conciseness": int, "style": int}},
  "winner": "a" | "b" | "tie",
  "rationale": str
}}

Transcript (truncated for context):
---
{transcript}
---

Summary A:
---
{summary_a}
---

Summary B:
---
{summary_b}
---
"""


def render_user_prompt(transcript: str) -> str:
    """Render the SFT/inference user prompt for a transcript."""
    return USER_PROMPT_TEMPLATE.format(transcript=transcript)


def render_judge_prompt(transcript: str, summary_a: str, summary_b: str) -> str:
    """Render the LLM-as-judge prompt for two candidate summaries."""
    return JUDGE_RUBRIC.format(
        transcript=transcript,
        summary_a=summary_a,
        summary_b=summary_b,
    )
