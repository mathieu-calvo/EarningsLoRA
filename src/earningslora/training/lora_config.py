"""PEFT LoRA configuration.

Default hyperparameters target the QLoRA recipe on a T4 16 GB:
  - r=16, alpha=32, dropout=0.05
  - all attention + MLP projections as target modules
  - ~50 MB adapter footprint

Stub — implemented in Weekend 3.
"""

from __future__ import annotations


def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Return a `peft.LoraConfig` with project defaults."""
    raise NotImplementedError("Implemented in Weekend 3 — LoRA config.")
