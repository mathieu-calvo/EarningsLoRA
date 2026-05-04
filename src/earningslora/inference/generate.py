"""Single + batched generation with deterministic seed and stop tokens.

Stub — implemented in Weekend 3.
"""

from __future__ import annotations


def generate_summary(model, tokenizer, transcript: str, max_new_tokens: int = 512) -> str:
    """Generate a bullet summary for one transcript."""
    raise NotImplementedError("Implemented in Weekend 3.")


def generate_batch(model, tokenizer, transcripts: list[str], max_new_tokens: int = 512) -> list[str]:
    """Batched generation; pads on the left and uses a deterministic seed."""
    raise NotImplementedError("Implemented in Weekend 3.")
