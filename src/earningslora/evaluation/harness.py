"""Run a model on the hold-out and persist predictions to disk.

The harness is configuration-agnostic: it accepts any callable mapping
`transcript -> summary`, so the same code path runs the base model, the
fine-tuned model, and the frontier baseline.

Stub — implemented in Weekend 2.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def run_holdout(
    name: str,
    summarise: Callable[[str], str],
    holdout,
    output_dir: Path,
) -> Path:
    """Run `summarise` over the hold-out and save predictions to JSONL."""
    raise NotImplementedError("Implemented in Weekend 2.")
