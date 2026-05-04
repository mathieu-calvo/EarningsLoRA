"""Load base model + LoRA adapter (4-bit) or a merged model.

All heavy imports are inside the functions so module import stays cheap on
Windows / CPU CI without `[train]` extras.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_4bit(base_model: str):
    """Load base in 4-bit NF4 with a tokenizer prepared for batched left-padded gen."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched causal generation

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = True
    return model, tokenizer


def load_base(base_model: str):
    """Load just the base model in 4-bit (zero-shot baseline path)."""
    logger.info("Loading base model %s in 4-bit", base_model)
    model, tokenizer = _load_4bit(base_model)
    model.eval()
    return model, tokenizer


def load_with_adapter(base_model: str, adapter_dir: Path | str):
    """Load the base in 4-bit and attach a LoRA adapter for inference."""
    from peft import PeftModel

    logger.info("Loading base + adapter from %s", adapter_dir)
    model, tokenizer = _load_4bit(base_model)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def load_merged(merged_dir: Path | str):
    """Load a merged adapter+base model from disk (post-`merge.py`)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading merged model from %s", merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer
