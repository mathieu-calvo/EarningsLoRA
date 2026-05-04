"""Long-transcript handling for the 4096-token training context.

Earnings calls vary from ~3k to ~15k tokens. When the transcript exceeds the
configured `max_seq_len`, we truncate by section priority:
  1. Prepared remarks (kept in full where possible)
  2. Q&A (truncated last)
  3. Operator boilerplate (dropped first)

Stub — implemented in Weekend 1. Will use a regex section detector and a
tokenizer-aware truncator (no naive char slicing).
"""

from __future__ import annotations


def split_sections(transcript: str) -> dict[str, str]:
    """Detect prepared-remarks vs Q&A vs boilerplate sections."""
    raise NotImplementedError("Implemented in Weekend 1 — section detection.")


def fit_to_budget(transcript: str, max_tokens: int, tokenizer) -> str:
    """Truncate by section priority to fit `max_tokens`."""
    raise NotImplementedError("Implemented in Weekend 1 — token-aware truncation.")
