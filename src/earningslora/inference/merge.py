"""Merge a LoRA adapter into the base model and persist the merged weights.

The merged model is bf16 (no quantisation) so it loads cleanly on Spaces /
ZeroGPU and on any inference runtime that doesn't ship bitsandbytes. Heavy
imports stay inside the function so importing this module on CPU CI / Windows
remains cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_and_save(
    base_model: str,
    adapter_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    """Merge `adapter_dir` into `base_model` and save to `output_dir` (bf16)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base %s in bf16 for merge", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Attaching adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    logger.info("Merging adapter into base")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Merged model saved to %s", output_dir)
    return output_dir
