# Evaluation

## The frozen hold-out

A single 50-row hold-out, materialised once with `Settings.eval_seed=42` from the
ECTSum test split. Same set used across all three configurations (base / FT /
frontier). Materialised by `scripts/prepare_dataset.py` to
`data/eval/holdout.json` and (optionally) mirrored on HF Hub at
`mathieu-calvo/ectsum-chat:holdout`.

## Metrics

### ROUGE-1 / ROUGE-2 / ROUGE-L

Mechanical baseline. Useful as a smoke test, weak as an absolute quality signal —
ROUGE rewards lexical overlap, which mostly tracks "did the model copy the right
phrases" rather than "is this a good summary."

### Numeric recall (project-specific)

Fraction of numbers in the generated summary that also appear in the source
transcript. Implemented in `evaluation/numeric_recall.py`. Penalises hallucinated
figures — the most consequential failure mode for finance summarisation. Numbers
are normalised across "$1.2 billion" / "1.2B" / "1,200,000,000" / "1.2 bn" so unit
formatting doesn't drive the score.

This metric is the project's most differentiated piece. Off-the-shelf eval libs
ship ROUGE / BLEU / BERTScore but not finance-specific factuality.

### LLM-as-judge (Gemini 2.5 Flash)

Four rubrics on a 1-5 scale: faithfulness, completeness, conciseness, style.
Each pair (FT vs base, FT vs frontier, frontier vs base) is judged head-to-head
with deterministic prompts (temperature=0). Calls cached to SQLite so reruns
are free. Rubric prompt version-pinned in `utils/prompts.py:JUDGE_RUBRIC` —
bumping the rubric version invalidates the cache.

Order-randomised (`shuffle=True` by default) to remove the position bias that
LLM-as-judge models tend to show.

### Cost / latency

- Cost: `$ / 1M input` / `$ / 1M output` from `Settings.{base,ft,frontier}_cost_per_1m_*`. Static labels — base and FT are $0; frontier carries the public list price even though we use the AI Studio free tier, since the comparison story is at-scale economics, not our quota.
- Latency: p50 ms/request from `runs/eval/<config>.jsonl`. Local models on T4; Gemini on the API.

## Bench output

`scripts/evaluate.py` writes `reports/bench.json`. Schema:

```jsonc
{
  "metadata": {
    "generated_at": "2026-...",
    "holdout": {"size": 50, "seed": 42},
    "settings": {"base_model": "...", "frontier_model": "...", "judge_model": "...", "adapter_repo": "..."},
    "configs_run": ["base", "ft", "frontier"],
    "skipped": []                        // e.g. [{"config": "ft", "reason": "adapter_dir missing"}]
  },
  "configs": {
    "base": {
      "name": "Llama 3.2 3B Instruct (zero-shot)",
      "predictions_path": "runs/eval/base.jsonl",
      "n": 50, "n_errors": 0,
      "rouge": {"rouge1": 0.34, "rouge2": 0.13, "rougeL": 0.21, "n": 50},
      "numeric_recall": 0.81,
      "latency_ms_p50": 1850.0,
      "cost_per_1m_input": 0.0,
      "cost_per_1m_output": 0.0
    },
    "ft":      { /* same shape */ },
    "frontier":{ /* same shape */ }
  },
  "judge_winrates": {
    "ft_vs_base":       {"ft": 0.62,       "base": 0.32, "tie": 0.06, "n": 50},
    "frontier_vs_base": {"frontier": 0.74, "base": 0.20, "tie": 0.06, "n": 50},
    "ft_vs_frontier":   {"ft": 0.41,       "frontier": 0.49, "tie": 0.10, "n": 50}
  }
}
```

The README headline table is regenerated from this file by
`scripts/evaluate.py` (or `--update-readme-only`). Never hand-edit the table —
edits between the `<!-- BENCH:HEADLINE:BEGIN -->` markers will be overwritten
on the next run.

## CLI

```bash
# Full bench (after training has produced an adapter)
python scripts/evaluate.py --adapter-dir runs/latest/adapter

# Pre-training: only base + frontier
python scripts/evaluate.py --configs base,frontier

# Skip the LLM judge (saves Gemini quota during dev)
python scripts/evaluate.py --configs base,ft --no-judge --adapter-dir runs/latest/adapter

# Just refresh the README from existing bench.json
python scripts/evaluate.py --update-readme-only
```

## Idempotency

Per-config predictions land in `runs/eval/<config>.jsonl`; the harness will
overwrite them on each run. Frontier + judge calls are SQLite-cached on disk
(see `~/.earningslora/{frontier,judge}_cache.db`), so reruns over the same
hold-out cost zero quota. Bumping `JUDGE_RUBRIC` text invalidates the judge
cache automatically since the rubric is part of the cache key.
