"""Centralised settings via pydantic-settings.

All env vars are prefixed `EARNINGSLORA_`; values can also be supplied via a `.env`
file at the repo root. Secret-style keys (GOOGLE_API_KEY, HF_TOKEN, WANDB_API_KEY)
keep their canonical names since downstream libraries read them directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="EARNINGSLORA_",
        extra="ignore",
    )

    # Models
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    judge_model: str = "gemini-2.5-flash"
    frontier_model: str = "gemini-2.5-flash"

    # HF Hub
    adapter_repo: str = "mathieu-calvo/llama-3.2-3b-earningslora"
    dataset_repo: str = "mathieu-calvo/ectsum-chat"
    upstream_dataset: str = "mrm8488/ectsum"

    # Training shape
    max_seq_len: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Eval
    eval_holdout_size: int = 50
    eval_seed: int = 42

    # Cache
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".earningslora")


def get_settings() -> Settings:
    """Lazy accessor so `Settings()` isn't constructed at import time."""
    return Settings()
