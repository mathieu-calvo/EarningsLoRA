"""HuggingFace Hub uploaders for adapters, datasets, and model cards.

Stub — fills in during Weekend 4. Will wrap `huggingface_hub.HfApi` with
project-specific repo conventions and a model-card template that includes the
eval table from `reports/bench.json`.
"""

from __future__ import annotations

from pathlib import Path


def push_adapter(adapter_dir: Path, repo_id: str) -> str:
    """Push a trained LoRA adapter to the Hub. Returns the repo URL."""
    raise NotImplementedError("Implemented in Weekend 4 — adapter publish.")


def push_dataset(dataset_dir: Path, repo_id: str) -> str:
    """Push a chat-template-formatted dataset to the Hub."""
    raise NotImplementedError("Implemented in Weekend 1 — dataset publish.")


def render_model_card(bench_json_path: Path) -> str:
    """Render the model card markdown from the latest bench JSON."""
    raise NotImplementedError("Implemented in Weekend 4 — model card.")
