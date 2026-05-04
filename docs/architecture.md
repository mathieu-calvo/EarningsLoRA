# Architecture

> Skeleton — fills in across the build.

## Layers

- **`data/`** — ECTSum loader, chat-template formatter, section-aware chunker, EDA stats. Owns the source-of-truth of "what an SFT example looks like."
- **`training/`** — TRL `SFTTrainer` wrapper + PEFT LoRA config + W&B callbacks. Single entrypoint: `training.sft.train(...)`.
- **`inference/`** — Loads base + adapter (4-bit) or merged model. Single-call and batched generation with deterministic seeds.
- **`evaluation/`** — Per-metric modules (rouge, numeric_recall, llm_judge, frontier_baseline) composed by `bench.run_bench()`. Outputs `reports/bench.json`.
- **`utils/`** — Prompts (single source of truth across SFT / baseline / judge), HF Hub uploaders.

## Provider abstraction

Frontier baseline and judge LLMs are interchangeable: both are wrapped behind a
small `Protocol` so swapping Gemini → Groq → Anthropic is a one-file change.
Same pattern as `RAG-knowledge-engine/src/rag_engine/llm/provider.py`.

## Cache strategy

Two-tier: in-memory LRU + SQLite blob, keyed by deterministic input hashes:

- Frontier-baseline calls keyed by `hash(prompt_template_version, transcript)`.
- LLM-judge calls keyed by `hash(rubric_version, transcript, summary_a, summary_b)`.

Reruns of `scripts/evaluate.py` cost zero quota.

## Why this layering

- `data/` and `evaluation/` are the only modules that need to be production-quality
  on day one — they're the load-bearing pieces of the bench.
- `training/` and `inference/` lean on TRL/PEFT/transformers heavily; the wrappers
  stay thin so the upstream libraries handle correctness.
- `utils/prompts.py` is deliberately one file: changing the task definition flips
  through a single diff and a single commit message.
