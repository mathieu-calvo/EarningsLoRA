"""PEFT LoRA configuration.

Default hyperparameters target the QLoRA recipe on a T4 16 GB:
  - r=16, alpha=32, dropout=0.05
  - all attention + MLP projections as target modules
  - ~50 MB adapter footprint
"""

from __future__ import annotations

# Llama-family + Qwen-family share these projection names. If a future base model
# uses a different naming scheme, override `target_modules` at the call site.
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
):
    """Return a `peft.LoraConfig` with project defaults.

    `peft` is imported lazily so the rest of the package stays importable
    without the `[train]` extras (and on Windows where bitsandbytes can't install).
    """
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
