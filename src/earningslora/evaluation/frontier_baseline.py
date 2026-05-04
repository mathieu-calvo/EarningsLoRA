"""Zero-shot frontier baseline: Gemini 2.5 Flash on the same hold-out.

Same prompt template as SFT (see `utils.prompts`) so the comparison is apples to
apples. Sets the upper bound the fine-tuned 3B model is trying to approach.

Stub — implemented in Weekend 2.
"""

from __future__ import annotations


def frontier_summary(transcript: str) -> str:
    """Generate a zero-shot summary with the frontier model."""
    raise NotImplementedError("Implemented in Weekend 2 — frontier baseline.")


def frontier_batch(transcripts: list[str]) -> list[str]:
    """Batched zero-shot summaries with caching."""
    raise NotImplementedError("Implemented in Weekend 2 — batched frontier.")
