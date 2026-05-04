"""TRL SFTTrainer wrapper.

Stub — implemented in Weekend 3. Will:
- load base model in 4-bit (`bitsandbytes` NF4) with gradient checkpointing
- attach LoRA adapter from `lora_config.build_lora_config()`
- wrap dataset with the base tokenizer's chat template
- train for `Settings.num_train_epochs` with W&B logging
- save adapter to `runs/<timestamp>/`
"""

from __future__ import annotations

from pathlib import Path


def train(
    base_model: str,
    dataset_dir: Path,
    output_dir: Path,
    max_seq_len: int = 4096,
) -> Path:
    """Run SFT and return the path to the saved adapter."""
    raise NotImplementedError("Implemented in Weekend 3 — SFTTrainer.")
