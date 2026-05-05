"""Tests for the bench orchestrator that don't need any model or network.

We inject summariser factories so `run_bench` runs against trivial Python
callables, then verify the bench.json shape, the FT-skip behaviour, and the
README headline-table regenerator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from earningslora.config import Settings
from earningslora.evaluation import bench as bench_mod
from earningslora.evaluation.bench import (
    HEADLINE_BEGIN,
    HEADLINE_END,
    regenerate_readme_table,
    render_headline_table,
    run_bench,
)


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        eval_dir=tmp_path / "eval",
        bench_path=tmp_path / "bench.json",
        holdout_manifest=tmp_path / "holdout.json",
    )


def _holdout() -> list[dict]:
    return [
        {"transcript": "Q3 revenue was 1.2 billion.", "summary": "- Revenue: $1.2B"},
        {"transcript": "Q4 revenue was 1.5 billion.", "summary": "- Revenue: $1.5B"},
    ]


def _stub_factory(label: str):
    def _factory():
        def _summarise(transcript: str) -> str:
            # Deterministic + grounded — uses the same number that's in the transcript.
            number = "1.2 billion" if "1.2" in transcript else "1.5 billion"
            return f"- {label}: {number} reported"

        return _summarise

    return _factory


def test_run_bench_writes_expected_shape(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        bench_mod,
        "_judge_pair",
        lambda *a, **kw: {"a": 0.8, "b": 0.2, "tie": 0.0, "n": 2},
    )

    out = run_bench(
        holdout=_holdout(),
        configs=("base", "frontier"),
        settings=settings,
        summariser_factories={
            "base": _stub_factory("base"),
            "frontier": _stub_factory("frontier"),
        },
    )

    assert out == settings.bench_path
    payload = json.loads(out.read_text())
    assert set(payload["configs"].keys()) == {"base", "frontier"}
    base_cfg = payload["configs"]["base"]
    assert base_cfg["n"] == 2
    assert base_cfg["n_errors"] == 0
    assert base_cfg["numeric_recall"] == pytest.approx(1.0)
    assert "rouge1" in base_cfg["rouge"]
    assert base_cfg["cost_per_1m_input"] == 0.0
    assert payload["judge_winrates"]["frontier_vs_base"]["frontier"] == 0.8
    assert payload["metadata"]["holdout"]["size"] == 2


def test_run_bench_skips_ft_without_adapter(tmp_path):
    settings = _make_settings(tmp_path)
    out = run_bench(
        holdout=_holdout(),
        configs=("base", "ft"),
        settings=settings,
        adapter_dir=tmp_path / "does-not-exist",
        skip_judge=True,
        summariser_factories={"base": _stub_factory("base")},
    )
    payload = json.loads(out.read_text())
    assert "ft" not in payload["configs"]
    assert payload["metadata"]["skipped"] == [{"config": "ft", "reason": "adapter_dir missing"}]


def test_run_bench_handles_summariser_errors(tmp_path):
    settings = _make_settings(tmp_path)

    def _bad_factory():
        def _summarise(transcript: str) -> str:
            raise RuntimeError("boom")

        return _summarise

    out = run_bench(
        holdout=_holdout(),
        configs=("base",),
        settings=settings,
        skip_judge=True,
        summariser_factories={"base": _bad_factory},
    )
    payload = json.loads(out.read_text())
    base_cfg = payload["configs"]["base"]
    assert base_cfg["n"] == 2
    assert base_cfg["n_errors"] == 2
    assert base_cfg["numeric_recall"] == 0.0


def test_render_headline_table_marks_ft_bold():
    bench = {
        "configs": {
            "base": {
                "name": "Base",
                "rouge": {"rougeL": 0.21},
                "numeric_recall": 0.81,
                "latency_ms_p50": 1850.0,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            },
            "ft": {
                "name": "FT",
                "rouge": {"rougeL": 0.30},
                "numeric_recall": 0.91,
                "latency_ms_p50": 1900.0,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            },
            "frontier": {
                "name": "Frontier",
                "rouge": {"rougeL": 0.33},
                "numeric_recall": 0.95,
                "latency_ms_p50": 600.0,
                "cost_per_1m_input": 0.075,
                "cost_per_1m_output": 0.30,
            },
        },
        "judge_winrates": {
            "ft_vs_base": {"ft": 0.62, "base": 0.32, "tie": 0.06, "n": 50},
            "frontier_vs_base": {"frontier": 0.74, "base": 0.20, "tie": 0.06, "n": 50},
        },
    }
    table = render_headline_table(bench)
    assert "**FT**" in table
    assert "**0.300**" in table
    assert "$0.075 / $0.3" in table  # cost line
    assert "62%" in table  # FT vs base winrate


def test_regenerate_readme_table_round_trip(tmp_path):
    bench_path = tmp_path / "bench.json"
    bench_path.write_text(
        json.dumps(
            {
                "configs": {
                    "base": {
                        "name": "Base",
                        "rouge": {"rougeL": 0.2},
                        "numeric_recall": 0.8,
                        "latency_ms_p50": 1000.0,
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                    }
                },
                "judge_winrates": {},
            }
        )
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{HEADLINE_BEGIN}\nold table\n{HEADLINE_END}\noutro\n")

    changed = regenerate_readme_table(readme_path=readme, bench_path=bench_path)
    assert changed is True
    text = readme.read_text()
    assert "old table" not in text
    assert "Base" in text
    assert "intro" in text and "outro" in text


def test_regenerate_readme_requires_markers(tmp_path):
    bench_path = tmp_path / "bench.json"
    bench_path.write_text('{"configs": {}, "judge_winrates": {}}')
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")
    with pytest.raises(ValueError):
        regenerate_readme_table(readme_path=readme, bench_path=bench_path)


def test_regenerate_readme_missing_bench(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"{HEADLINE_BEGIN}\n{HEADLINE_END}")
    with pytest.raises(FileNotFoundError):
        regenerate_readme_table(readme_path=readme, bench_path=tmp_path / "missing.json")
