"""Bench orchestrator: run all three configurations and write `reports/bench.json`.

Configurations:
  - `base`     : Llama 3.2 3B Instruct, zero-shot
  - `ft`       : same base + the LoRA adapter from `Settings.adapter_repo`
  - `frontier` : Gemini 2.5 Flash, zero-shot

Metrics rolled up per configuration:
  - rouge_1 / rouge_2 / rouge_l
  - numeric_recall (mean across hold-out)
  - llm_judge_winrate (head-to-head vs base)
  - cost_per_1m_tokens
  - latency_ms_p50

The output JSON is the source of truth for the README headline table; the
README is regenerated from it (no manual edits) so the table never drifts from
the numbers.

Stub — implemented in Weekend 4.
"""

from __future__ import annotations

from pathlib import Path


def run_bench(output_path: Path = Path("reports/bench.json")) -> Path:
    """Run all three configurations and persist the bench JSON."""
    raise NotImplementedError("Implemented in Weekend 4 — bench orchestration.")
