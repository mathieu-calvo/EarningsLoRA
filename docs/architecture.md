# Architecture

## Layers

- **`data/`** — ECTSum loader, chat-template formatter, section-aware chunker, EDA stats. Owns the source-of-truth of "what an SFT example looks like."
- **`training/`** — TRL `SFTTrainer` wrapper + PEFT LoRA config + W&B callbacks. Single entrypoint: `training.sft.train(...)`.
- **`inference/`** — Loads base + adapter (4-bit) or merged model; `merge_and_save` produces the bf16 weights served by the Space. Single-call and batched generation with deterministic seeds.
- **`evaluation/`** — Per-metric modules (`rouge`, `numeric_recall`, `llm_judge`, `frontier_baseline`) composed by `bench.run_bench()`. Outputs `reports/bench.json` and regenerates the README headline table.
- **`demo/`** — Framework-agnostic helpers (`time_call`, `hf_inference_summarise`, `numeric_recall` for the UI). Both the Streamlit app and the Gradio Space import from here.
- **`utils/`** — Prompts (single source of truth across SFT / baseline / judge), HF Hub uploaders, SQLite cache.

## Provider abstraction

Frontier baseline and judge LLMs are interchangeable: both are wrapped behind a
small `Protocol` so swapping Gemini → Groq → Anthropic is a one-file change.
Same pattern as `RAG-knowledge-engine/src/rag_engine/llm/provider.py`.

## Cache strategy

Two-tier: in-memory LRU + SQLite blob, keyed by deterministic input hashes:

- Frontier-baseline calls keyed by `hash(prompt_template_version, transcript)`.
- LLM-judge calls keyed by `hash(rubric_version, transcript, summary_a, summary_b)`.

Reruns of `scripts/evaluate.py` cost zero quota.

## Bench orchestration

`evaluation/bench.py` is the conductor:

1. Loads the frozen 50-row hold-out from `data/eval/holdout.json`.
2. For each requested config (`base` / `ft` / `frontier`), constructs a `summarise(transcript) -> str` callable and feeds it through `evaluation/harness.run_holdout`. Predictions land in `runs/eval/<config>.jsonl`.
3. Rolls up ROUGE, numeric-recall, p50 latency, cost-per-1M (label) per config.
4. Runs head-to-head LLM-as-judge for each pair (`ft_vs_base`, `frontier_vs_base`, `ft_vs_frontier`). Order-randomised to remove position bias; cached.
5. Writes `reports/bench.json` and regenerates the headline table in `README.md` between the `<!-- BENCH:HEADLINE:BEGIN -->` markers.

The FT path is **adapter-optional**: if `--adapter-dir` is missing, FT is skipped with a warning and the run still produces a bench JSON for the other configs. This lets the harness ship before the first training run.

## Why this layering

- `data/` and `evaluation/` are the only modules that need to be production-quality on day one — they're the load-bearing pieces of the bench.
- `training/` and `inference/` lean on TRL/PEFT/transformers heavily; the wrappers stay thin so the upstream libraries handle correctness.
- `utils/prompts.py` is deliberately one file: changing the task definition flips through a single diff and a single commit message.
- `demo/` is in the package (not under `app/`) so it's installable and tested. The `app/*` files are launchers that import from it.
