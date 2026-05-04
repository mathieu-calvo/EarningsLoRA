"""ECTSum dataset loader.

Loads the upstream ECTSum dataset from HuggingFace and returns a
`datasets.DatasetDict` with `train` / `validation` / `test` splits plus a frozen
50-row hold-out for the eval bench.

Stub — implemented in Weekend 1. Will:
- pin the upstream revision via `revision=` for reproducibility
- materialise the 50-row hold-out with `Settings.eval_seed` for determinism
- save the chat-template formatted variant locally for fast reuse
"""

from __future__ import annotations


def load_ectsum():
    """Load the ECTSum dataset with pinned revision and split sizes."""
    raise NotImplementedError("Implemented in Weekend 1 — ECTSum loader.")


def make_holdout(seed: int = 42, size: int = 50):
    """Carve a frozen N-row hold-out from the test split."""
    raise NotImplementedError("Implemented in Weekend 1 — frozen hold-out.")
