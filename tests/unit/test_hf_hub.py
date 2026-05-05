"""Tests for the HF Hub helpers — model-card rendering only.

Push paths are not exercised here (they require network + Hub credentials);
tests focus on the deterministic templating that ships with the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

from earningslora.utils.hf_hub import render_model_card


def _bench_blob(tmp_path: Path) -> Path:
    bench = {
        "metadata": {
            "holdout": {"size": 50, "seed": 42},
        },
        "configs": {
            "ft": {
                "name": "FT",
                "rouge": {"rougeL": 0.30},
                "numeric_recall": 0.91,
                "latency_ms_p50": 1900.0,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            }
        },
        "judge_winrates": {
            "ft_vs_base": {"ft": 0.62, "base": 0.32, "tie": 0.06, "n": 50},
        },
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(bench))
    return path


def test_render_model_card_includes_table(tmp_path):
    card = render_model_card(
        bench_path=_bench_blob(tmp_path),
        repo_id="me/adapter",
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        dataset_repo="me/dataset",
    )
    assert "me/adapter" in card
    assert "meta-llama/Llama-3.2-3B-Instruct" in card
    assert "**FT**" in card  # rendered headline table
    assert "numeric_recall" in card  # narrative explainer
    assert card.startswith("---")  # YAML front-matter for the Hub


def test_render_model_card_no_bench(tmp_path):
    card = render_model_card(
        bench_path=tmp_path / "missing.json",
        repo_id="me/adapter",
    )
    assert "Eval numbers will populate" in card
    assert "me/adapter" in card
