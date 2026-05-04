"""Load base model + LoRA adapter (4-bit) or a merged model.

Stub — implemented in Weekend 3.
"""

from __future__ import annotations

from pathlib import Path


def load_with_adapter(base_model: str, adapter_dir: Path):
    """Load base in 4-bit and attach the LoRA adapter."""
    raise NotImplementedError("Implemented in Weekend 3.")


def load_merged(merged_dir: Path):
    """Load a merged adapter+base model from disk."""
    raise NotImplementedError("Implemented in Weekend 4.")
