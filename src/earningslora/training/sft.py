"""TRL SFTTrainer wrapper.

Drives the QLoRA recipe: 4-bit NF4 base + LoRA adapter + chat-template SFT.
Designed for a single T4 16 GB box (Kaggle / Colab). All heavy deps
(`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`) are lazy-imported
inside `train()` so importing `earningslora.training.sft` on Windows / CPU CI
still works.
"""

from __future__ import annotations

import logging
from pathlib import Path

from earningslora.config import get_settings
from earningslora.training.lora_config import build_lora_config

logger = logging.getLogger(__name__)


def _build_bnb_config():
    """4-bit NF4 quantisation config (bitsandbytes)."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_base_for_training(base_model: str):
    """Load 4-bit base + tokenizer prepared for kbit training."""
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_build_bnb_config(),
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def train(
    base_model: str | None = None,
    dataset_dir: Path | str | None = None,
    output_dir: Path | str = "runs/latest",
    *,
    max_seq_len: int | None = None,
    num_train_epochs: int | None = None,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    resume_from: str | None = None,
) -> Path:
    """Run SFT and return the path to the saved adapter.

    `None` arguments fall back to `Settings` defaults so notebooks and the CLI
    stay declarative.
    """
    from datasets import load_from_disk
    from peft import get_peft_model
    from trl import SFTConfig, SFTTrainer

    settings = get_settings()
    base_model = base_model or settings.base_model
    max_seq_len = max_seq_len or settings.max_seq_len
    num_train_epochs = num_train_epochs or settings.num_train_epochs
    per_device_batch_size = per_device_batch_size or settings.per_device_batch_size
    gradient_accumulation_steps = (
        gradient_accumulation_steps or settings.gradient_accumulation_steps
    )
    learning_rate = learning_rate or settings.learning_rate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_dir is None:
        dataset_dir = "data/processed/ectsum-chat"
    dataset_dir = Path(dataset_dir)

    logger.info("Loading dataset from %s", dataset_dir)
    dataset = load_from_disk(str(dataset_dir))

    logger.info("Loading base model %s in 4-bit", base_model)
    model, tokenizer = _load_base_for_training(base_model)

    logger.info(
        "Attaching LoRA adapter (r=%d, alpha=%d, dropout=%.2f)",
        settings.lora_r,
        settings.lora_alpha,
        settings.lora_dropout,
    )
    lora_cfg = build_lora_config(
        r=settings.lora_r,
        alpha=settings.lora_alpha,
        dropout=settings.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    has_validation = "validation" in dataset
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if has_validation else "no",
        bf16=True,
        report_to="wandb",
        seed=42,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") if has_validation else None,
        tokenizer=tokenizer,
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=resume_from)

    adapter_path = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Saved adapter to %s", adapter_path)
    return adapter_path
