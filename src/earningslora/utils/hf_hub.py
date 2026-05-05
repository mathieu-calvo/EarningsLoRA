"""HuggingFace Hub uploaders for adapters, datasets, and the model card.

Thin wrappers around `huggingface_hub.HfApi` with project-specific repo
conventions and a model-card template that reads from `reports/bench.json` so
the published card never drifts from the latest eval table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from earningslora.config import get_settings
from earningslora.evaluation.bench import render_headline_table

logger = logging.getLogger(__name__)


_MODEL_CARD_TEMPLATE = """\
---
language: en
license: mit
tags:
  - llama
  - lora
  - peft
  - finance
  - earnings-call
  - summarization
base_model: {base_model}
datasets:
  - {dataset_repo}
library_name: peft
pipeline_tag: text-generation
---

# {repo_id}

QLoRA adapter for **{base_model}** fine-tuned on earnings call transcripts to
produce analyst-style bullet summaries. Trained on a free Kaggle/Colab T4 with
4-bit NF4 quantisation; ~50 MB adapter, 2 epochs, ~2k examples.

Source code, training recipe, and the bench harness:
[github.com/mathieu-calvo/EarningsLoRA](https://github.com/mathieu-calvo/EarningsLoRA).

## Headline eval

Frozen 50-row hold-out (seed={eval_seed}) from the ECTSum test split. Same
hold-out across all three configurations.

{headline_table}

`numeric_recall` is a project-specific metric: the fraction of numbers in the
generated summary that appear (after unit normalisation) in the source
transcript. The most consequential failure mode for finance summarisation is
hallucinated figures; this metric measures it directly.

LLM-as-judge: Gemini 2.5 Flash with a 4-rubric scale (faithfulness,
completeness, conciseness, style), order-randomised to remove position bias.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "{base_model}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```

The chat template, system prompt, and inference helpers live in the source
repo at `src/earningslora/inference/`.

## Reproducibility

Hold-out, prompt versions, and the bench JSON are version-pinned in the source
repo. Re-running `python scripts/evaluate.py` regenerates this card's table
from `reports/bench.json` (judge calls cached on disk, no quota burn).
"""


def push_adapter(
    adapter_dir: Path | str,
    repo_id: str | None = None,
    *,
    bench_path: Path | str | None = None,
    private: bool = False,
    commit_message: str = "Update adapter + model card",
) -> str:
    """Push a trained LoRA adapter (and a freshly rendered card) to the Hub.

    Returns the repo URL. Raises if `adapter_dir` doesn't exist — callers that
    want a graceful no-op should check first.
    """
    from huggingface_hub import HfApi

    settings = get_settings()
    repo_id = repo_id or settings.adapter_repo
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_dir {adapter_dir} does not exist")

    card_text = render_model_card(
        bench_path=bench_path or settings.bench_path,
        repo_id=repo_id,
    )
    card_path = adapter_dir / "README.md"
    card_path.write_text(card_text, encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Pushed adapter to %s", url)
    return url


def push_dataset(
    dataset_dir: Path | str,
    repo_id: str | None = None,
    *,
    private: bool = False,
    commit_message: str = "Update chat-formatted dataset",
) -> str:
    """Push a chat-template-formatted DatasetDict (saved via `save_to_disk`) to the Hub."""
    from datasets import load_from_disk

    settings = get_settings()
    repo_id = repo_id or settings.dataset_repo
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir {dataset_dir} does not exist")

    dataset = load_from_disk(str(dataset_dir))
    dataset.push_to_hub(repo_id, private=private, commit_message=commit_message)
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Pushed dataset to %s", url)
    return url


def push_space(
    space_dir: Path | str,
    repo_id: str | None = None,
    *,
    commit_message: str = "Update Space",
) -> str:
    """Push the Space code (Gradio app + requirements + README) to the Hub."""
    from huggingface_hub import HfApi

    settings = get_settings()
    repo_id = repo_id or settings.space_repo
    space_dir = Path(space_dir)
    if not space_dir.exists():
        raise FileNotFoundError(f"space_dir {space_dir} does not exist")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/spaces/{repo_id}"
    logger.info("Pushed Space to %s", url)
    return url


def render_model_card(
    bench_path: Path | str,
    repo_id: str,
    base_model: str | None = None,
    dataset_repo: str | None = None,
) -> str:
    """Render the model-card markdown from a bench.json snapshot."""
    settings = get_settings()
    base_model = base_model or settings.base_model
    dataset_repo = dataset_repo or settings.dataset_repo

    bench_path = Path(bench_path)
    if bench_path.exists():
        bench: dict[str, Any] = json.loads(bench_path.read_text(encoding="utf-8"))
        headline_table = render_headline_table(bench)
        eval_seed = bench.get("metadata", {}).get("holdout", {}).get("seed", settings.eval_seed)
    else:
        logger.warning("bench.json not found at %s — model card will ship without numbers.", bench_path)
        headline_table = "_Eval numbers will populate once `scripts/evaluate.py` has been run._"
        eval_seed = settings.eval_seed

    return _MODEL_CARD_TEMPLATE.format(
        base_model=base_model,
        dataset_repo=dataset_repo,
        repo_id=repo_id,
        headline_table=headline_table,
        eval_seed=eval_seed,
    )
