"""Bench orchestrator: run all configurations and write `reports/bench.json`.

Configurations:
  - `base`     : Llama 3.2 3B Instruct, zero-shot
  - `ft`       : same base + the LoRA adapter from `adapter_dir` (skipped if missing)
  - `frontier` : Gemini 2.5 Flash, zero-shot

Per-configuration metrics:
  - rouge_1 / rouge_2 / rouge_l
  - numeric_recall (mean across hold-out)
  - latency_ms_p50
  - cost_per_1m_input / cost_per_1m_output  (label, from Settings)
  - n / n_errors

Pairwise (LLM-as-judge):
  - judge_winrates: {"ft_vs_base": {...}, "frontier_vs_base": {...}, "ft_vs_frontier": {...}}

The output JSON is the source of truth for the README headline table; the table
is regenerated from this file (no manual edits) so it never drifts. Heavy deps
(`torch`, `transformers`, `peft`) are imported lazily inside the per-config
runners so this module stays importable on CPU CI / Windows.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from earningslora.config import Settings, get_settings
from earningslora.evaluation.frontier_baseline import frontier_summary
from earningslora.evaluation.harness import load_predictions, run_holdout
from earningslora.evaluation.llm_judge import JudgeVerdict, judge, winrate
from earningslora.evaluation.numeric_recall import numeric_recall
from earningslora.evaluation.rouge import rouge_scores

logger = logging.getLogger(__name__)

ALL_CONFIGS = ("base", "ft", "frontier")
_JUDGE_PAIRS = (
    ("ft", "base"),
    ("frontier", "base"),
    ("ft", "frontier"),
)


# --- holdout loading ---------------------------------------------------------


def load_holdout_manifest(path: Path | str) -> list[dict]:
    """Load the frozen hold-out written by `scripts/prepare_dataset.py`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["rows"])


# --- summariser factories ----------------------------------------------------


def _make_local_summariser(base_model: str, adapter_dir: Path | None):
    """Build a `summarise(transcript)->str` for the base or fine-tuned model.

    Heavy imports are inside this function so calling code that doesn't run
    `base`/`ft` configs (e.g. only `frontier`) doesn't need `[train]` extras.
    """
    from earningslora.inference.generate import generate_summary
    from earningslora.inference.load import load_base, load_with_adapter

    if adapter_dir is None:
        model, tokenizer = load_base(base_model)
    else:
        model, tokenizer = load_with_adapter(base_model, adapter_dir)

    def _summarise(transcript: str) -> str:
        return generate_summary(model, tokenizer, transcript)

    return _summarise


def _make_frontier_summariser():
    def _summarise(transcript: str) -> str:
        return frontier_summary(transcript)

    return _summarise


# --- per-config metric rollup ------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def _summarise_predictions(
    predictions_path: Path,
    cost_per_1m_input: float,
    cost_per_1m_output: float,
) -> dict[str, Any]:
    """Roll up rouge + numeric_recall + latency for one config."""
    rows = load_predictions(predictions_path)
    n = len(rows)
    n_errors = sum(1 for r in rows if r.get("error"))
    successful = [r for r in rows if not r.get("error") and r.get("prediction")]

    if not successful:
        return {
            "predictions_path": str(predictions_path),
            "n": n,
            "n_errors": n_errors,
            "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0},
            "numeric_recall": 0.0,
            "latency_ms_p50": 0.0,
            "cost_per_1m_input": cost_per_1m_input,
            "cost_per_1m_output": cost_per_1m_output,
        }

    predictions = [r["prediction"] for r in successful]
    references = [r["reference"] for r in successful]
    rouge = rouge_scores(predictions, references)

    nr_values = [
        numeric_recall(r["transcript"], r["prediction"]).recall for r in successful
    ]
    latencies = [float(r["latency_ms"]) for r in successful if r.get("latency_ms") is not None]

    return {
        "predictions_path": str(predictions_path),
        "n": n,
        "n_errors": n_errors,
        "rouge": rouge,
        "numeric_recall": round(statistics.mean(nr_values), 4) if nr_values else 0.0,
        "latency_ms_p50": round(_percentile(latencies, 0.5), 1),
        "cost_per_1m_input": cost_per_1m_input,
        "cost_per_1m_output": cost_per_1m_output,
    }


# --- pairwise judge ----------------------------------------------------------


def _judge_pair(
    holdout: Iterable[dict],
    a_preds: list[dict],
    b_preds: list[dict],
) -> dict[str, Any]:
    """Run head-to-head judge for two prediction sets, return winrate summary."""
    verdicts: list[JudgeVerdict] = []
    for transcript_row, row_a, row_b in zip(holdout, a_preds, b_preds, strict=False):
        if row_a.get("error") or row_b.get("error"):
            continue
        if not row_a.get("prediction") or not row_b.get("prediction"):
            continue
        verdicts.append(
            judge(
                transcript=transcript_row["transcript"],
                summary_a=row_a["prediction"],
                summary_b=row_b["prediction"],
            )
        )
    if not verdicts:
        return {"a": 0.0, "b": 0.0, "tie": 0.0, "n": 0}
    n = len(verdicts)
    return {
        "a": round(winrate(verdicts, "a"), 4),
        "b": round(winrate(verdicts, "b"), 4),
        "tie": round(sum(1 for v in verdicts if v.winner == "tie") / n, 4),
        "n": n,
    }


# --- main entrypoint ---------------------------------------------------------


def run_bench(
    output_path: Path | str | None = None,
    *,
    holdout: list[dict] | None = None,
    holdout_path: Path | str | None = None,
    adapter_dir: Path | str | None = None,
    configs: Sequence[str] = ALL_CONFIGS,
    eval_dir: Path | str | None = None,
    skip_judge: bool = False,
    settings: Settings | None = None,
    summariser_factories: dict[str, Callable[[], Callable[[str], str]]] | None = None,
) -> Path:
    """Run all selected configurations and persist `bench.json`.

    Parameters that matter:
      - `holdout` / `holdout_path`: pass either; `holdout_path` defaults to
        `Settings.holdout_manifest` (`data/eval/holdout.json`).
      - `adapter_dir`: required for the `ft` config. If `None` or missing, `ft`
        is skipped with a warning — lets the bench run before training lands.
      - `configs`: subset of `("base", "ft", "frontier")`.
      - `summariser_factories`: dependency-injection seam for tests; maps a
        config name to a no-arg callable that returns the summariser.
    """
    settings = settings or get_settings()
    output_path = Path(output_path) if output_path else settings.bench_path
    eval_dir = Path(eval_dir) if eval_dir else settings.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if holdout is None:
        holdout_path = Path(holdout_path) if holdout_path else settings.holdout_manifest
        if not holdout_path.exists():
            raise FileNotFoundError(
                f"Hold-out manifest not found at {holdout_path}. "
                f"Run `python scripts/prepare_dataset.py` first."
            )
        holdout = load_holdout_manifest(holdout_path)

    requested = list(configs)
    skipped: list[dict[str, str]] = []

    # Skip FT if no adapter; warn loudly.
    if "ft" in requested:
        adapter_path = Path(adapter_dir) if adapter_dir else None
        if adapter_path is None or not adapter_path.exists():
            logger.warning(
                "ft config requested but adapter_dir=%r does not exist — skipping. "
                "Run training first, then `scripts/evaluate.py --adapter-dir <path>`.",
                adapter_dir,
            )
            skipped.append({"config": "ft", "reason": "adapter_dir missing"})
            requested = [c for c in requested if c != "ft"]

    factories = dict(summariser_factories or {})

    def _factory_for(name: str) -> Callable[[], Callable[[str], str]]:
        if name in factories:
            return factories[name]
        if name == "base":
            return lambda: _make_local_summariser(settings.base_model, None)
        if name == "ft":
            return lambda: _make_local_summariser(settings.base_model, Path(adapter_dir))
        if name == "frontier":
            return _make_frontier_summariser
        raise ValueError(f"Unknown config: {name}")

    cost_for = {
        "base": (settings.base_cost_per_1m_input, settings.base_cost_per_1m_output),
        "ft": (settings.ft_cost_per_1m_input, settings.ft_cost_per_1m_output),
        "frontier": (
            settings.frontier_cost_per_1m_input,
            settings.frontier_cost_per_1m_output,
        ),
    }
    pretty_name = {
        "base": "Llama 3.2 3B Instruct (zero-shot)",
        "ft": "+ QLoRA (this repo)",
        "frontier": "Gemini 2.5 Flash (zero-shot)",
    }

    config_summaries: dict[str, dict[str, Any]] = {}
    config_predictions: dict[str, list[dict]] = {}
    for name in requested:
        logger.info("Running config %s", name)
        summarise = _factory_for(name)()
        preds_path = run_holdout(name, summarise, holdout, eval_dir)
        ci, co = cost_for[name]
        config_summaries[name] = {
            "name": pretty_name.get(name, name),
            **_summarise_predictions(preds_path, ci, co),
        }
        config_predictions[name] = load_predictions(preds_path)

    judge_winrates: dict[str, dict[str, Any]] = {}
    if not skip_judge:
        for a_name, b_name in _JUDGE_PAIRS:
            if a_name not in config_predictions or b_name not in config_predictions:
                continue
            logger.info("Judging %s vs %s", a_name, b_name)
            wr = _judge_pair(holdout, config_predictions[a_name], config_predictions[b_name])
            judge_winrates[f"{a_name}_vs_{b_name}"] = {
                a_name: wr["a"],
                b_name: wr["b"],
                "tie": wr["tie"],
                "n": wr["n"],
            }

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "holdout": {
                "size": len(holdout),
                "seed": settings.eval_seed,
            },
            "settings": {
                "base_model": settings.base_model,
                "frontier_model": settings.frontier_model,
                "judge_model": settings.judge_model,
                "adapter_repo": settings.adapter_repo,
            },
            "configs_run": list(config_summaries.keys()),
            "skipped": skipped,
        },
        "configs": config_summaries,
        "judge_winrates": judge_winrates,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return output_path


# --- README headline-table regeneration --------------------------------------


HEADLINE_BEGIN = "<!-- BENCH:HEADLINE:BEGIN -->"
HEADLINE_END = "<!-- BENCH:HEADLINE:END -->"


def _format_cost(cost_in: float, cost_out: float) -> str:
    if cost_in == 0 and cost_out == 0:
        return "$0"
    if cost_in == cost_out:
        return f"${cost_in:g}"
    return f"${cost_in:g} / ${cost_out:g}"


def _format_winrate(judge_winrates: dict[str, Any], config: str) -> str:
    """FT config: vs base. Frontier config: vs base. Base: '—'."""
    if config == "base":
        return "—"
    pair_key = f"{config}_vs_base"
    pair = judge_winrates.get(pair_key)
    if not pair:
        return "n/a"
    val = pair.get(config)
    if val is None:
        return "n/a"
    return f"{val * 100:.0f}%"


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if value == 0:
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def render_headline_table(bench: dict[str, Any]) -> str:
    """Render the markdown headline table from a parsed bench.json payload."""
    configs = bench.get("configs", {})
    judge = bench.get("judge_winrates", {})
    order = ["base", "ft", "frontier"]

    lines = [
        "| Configuration | ROUGE-L | Numeric recall | LLM-judge win-rate vs base | $ / 1M tok | ms / req (p50) |",
        "|---|---|---|---|---|---|",
    ]
    for name in order:
        cfg = configs.get(name)
        if cfg is None:
            continue
        rouge_l = cfg.get("rouge", {}).get("rougeL")
        nr = cfg.get("numeric_recall")
        cost = _format_cost(
            cfg.get("cost_per_1m_input", 0.0),
            cfg.get("cost_per_1m_output", 0.0),
        )
        latency = cfg.get("latency_ms_p50")
        latency_str = f"{latency:.0f}" if latency else "n/a"
        winrate_str = _format_winrate(judge, name)
        label = cfg.get("name", name)
        bold = name == "ft"
        cells = [
            f"**{label}**" if bold else label,
            f"**{_format_metric(rouge_l)}**" if bold else _format_metric(rouge_l),
            f"**{_format_metric(nr)}**" if bold else _format_metric(nr),
            f"**{winrate_str}**" if bold else winrate_str,
            f"**{cost}**" if bold else cost,
            f"**{latency_str}**" if bold else latency_str,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def regenerate_readme_table(
    readme_path: Path | str = "README.md",
    bench_path: Path | str | None = None,
) -> bool:
    """Replace the headline-table block in README between BEGIN/END markers.

    Returns True if the README was changed, False otherwise. The README must
    contain both markers; if missing, raises ValueError so callers know to add
    them once.
    """
    readme_path = Path(readme_path)
    bench_path = Path(bench_path) if bench_path else get_settings().bench_path
    if not bench_path.exists():
        raise FileNotFoundError(f"bench.json not found at {bench_path}")

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    table_md = render_headline_table(bench)
    block = f"{HEADLINE_BEGIN}\n{table_md}\n{HEADLINE_END}"

    text = readme_path.read_text(encoding="utf-8")
    if HEADLINE_BEGIN not in text or HEADLINE_END not in text:
        raise ValueError(
            f"{readme_path} is missing the headline-table markers "
            f"({HEADLINE_BEGIN} / {HEADLINE_END}). Add them around the table once."
        )
    pre, _, rest = text.partition(HEADLINE_BEGIN)
    _, _, post = rest.partition(HEADLINE_END)
    new_text = pre + block + post
    if new_text == text:
        return False
    readme_path.write_text(new_text, encoding="utf-8")
    return True
