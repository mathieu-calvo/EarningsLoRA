# EarningsLoRA

**Fine-tuning a 3B open model on earnings call transcripts for analyst-style bullet summaries — entirely free, end-to-end.**

A portfolio project that closes the inference-only gap most LLM portfolios have: an honest QLoRA fine-tuning pipeline (TRL + PEFT + bitsandbytes), an eval harness with a finance-specific factuality metric, and a head-to-head benchmark against zero-shot Gemini 2.5 Flash. Compute, hosting, datasets, and demo all run on free tiers.

> **Status:** 🚧 Skeleton in place. Roadmap below.

---

## Why fine-tune (and why not just RAG?)

Fine-tuning is the right tool when the goal is **behavioural**, not knowledge retrieval. For earnings call summarisation:

1. **Style/format learning** — bullet structure, tone, granularity are properties best learned from examples, not retrieved.
2. **Latency / cost at scale** — once trained, a 3B local adapter beats Gemini Flash on $/transcript and on cold-start latency. RAG addresses a different failure mode.
3. **Schema compliance** — predictable output structure for downstream pipelines.
4. **The transcript is fully in-context** — there is no external knowledge to retrieve. A retrieval step would just add noise.

See [`docs/why-fine-tune.md`](docs/why-fine-tune.md) for the full case.

---

## Headline eval table

_To be populated once Weekend 4 is complete. Anchored on the same frozen 50-transcript hold-out for all three configurations._

| Configuration | ROUGE-L | Numeric recall | LLM-judge win-rate | $ / 1M tok | ms / req (T4) |
|---|---|---|---|---|---|
| Llama 3.2 3B (zero-shot) | TBD | TBD | TBD | $0 | TBD |
| **+ QLoRA (this repo)** | **TBD** | **TBD** | **TBD** | **$0** | **TBD** |
| Gemini 2.5 Flash (zero-shot) | TBD | TBD | TBD | $0.075 | TBD |

The story this table will tell: small fine-tuned model approaches frontier quality at zero marginal cost.

---

## Architecture

```
                  ┌──────────────────┐
                  │  ECTSum (HF)     │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ data/ formatter  │  chat-template SFT pairs
                  └────────┬─────────┘
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐   ┌──────────────────┐
│ base 3B      │  │ QLoRA SFT    │   │ Gemini 2.5 Flash │
│ (zero-shot)  │  │ (T4, ~2 hr)  │   │ (zero-shot)      │
└──────┬───────┘  └──────┬───────┘   └────────┬─────────┘
       └────────┬────────┴────────────┬───────┘
                ▼                     ▼
       ┌─────────────────┐    ┌──────────────────┐
       │ Eval harness    │    │ Cost / latency   │
       │ ROUGE + numeric │    │ analysis         │
       │ recall + judge  │    │                  │
       └────────┬────────┘    └────────┬─────────┘
                └──────────┬───────────┘
                           ▼
                  ┌────────────────────┐
                  │ reports/bench.json │  → README headline table
                  └────────────────────┘
```

---

## Free-stack matrix

| Layer | Service | Free tier |
|---|---|---|
| Compute (training, primary) | Kaggle Notebooks | T4 16 GB · 30 hr/week |
| Compute (training, fallback) | Google Colab | T4 16 GB · 12 hr session |
| Compute (demo) | HuggingFace Spaces ZeroGPU | Ephemeral H200 · public Spaces |
| Demo fallback | Streamlit Community Cloud | Calls HF Inference API |
| Base model | Llama 3.2 3B Instruct | Open weights |
| Frontier baseline / judge | Gemini 2.5 Flash (Google AI Studio) | 1500 req/day |
| Dataset | ECTSum on HF Hub | Public |
| Adapter / dataset / model card hosting | HuggingFace Hub | Unlimited public |
| Experiment tracking | Weights & Biases | Free personal tier |
| CI | GitHub Actions | 2000 min/month for public repos |

**Total monthly cost: $0.**

---

## Quickstart (skeleton — full quickstart lands at the end of Weekend 1)

```bash
git clone https://github.com/mathieu-calvo/EarningsLoRA.git
cd EarningsLoRA
python -m venv .venv && source .venv/Scripts/activate   # Windows
pip install -e ".[dev,eval]"

cp .env.example .env  # optional — only needed for eval / publish

pytest -m "not slow"
ruff check src tests
```

For the training path, you'll additionally want `pip install -e ".[train]"` on a GPU host (Kaggle / Colab T4).

---

## Repository layout

```
src/earningslora/
├── config.py                 pydantic-settings (EARNINGSLORA_* env vars)
├── data/                     ECTSum loader, chat-template formatter, chunking, EDA stats
├── training/                 LoRA config, SFTTrainer wrapper, callbacks
├── inference/                Load base+adapter, generate, merge+push
├── evaluation/               ROUGE, numeric-recall, LLM-as-judge, frontier baseline, bench orchestrator
└── utils/                    Hub uploaders, prompts
notebooks/                    6 educational walkthroughs (01 → 06)
scripts/                      prepare_dataset / train / evaluate / publish CLIs
app/                          Streamlit + HF Space demo
tests/                        pytest unit + integration (CPU-only)
data/eval/                    Frozen 50-row hold-out (pointers, not raw)
reports/                      bench.json + side-by-side examples
docs/                         Architecture, training, evaluation, deployment, why-fine-tune
```

---

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 1 | Scaffold + ECTSum loader + chat-template formatter + length-distribution EDA (notebook 01) | Done |
| 2 | Zero-shot baselines: ROUGE + numeric-recall + judge + harness + notebook 02 (frontier code-complete; numbers populate on first run with `GOOGLE_API_KEY`) | Done |
| 3 | QLoRA SFT on Kaggle T4 (notebook 03 + `scripts/train.py`) | Pending |
| 4 | Bench orchestration — `scripts/evaluate.py` + `reports/bench.json` regenerator | Pending |
| 5 | HF Hub publish + HF Space demo + Streamlit Cloud fallback + README polish | Pending |

---

## Design principles

- **Eval-first.** The bench harness is in place before training begins, anchored on the same 50-row frozen hold-out across all three configurations. Every change is measured against it.
- **Free or it doesn't ship.** No paid LLM, no rented GPU, no commercial dataset.
- **Provider-agnostic at the seams.** Judge LLM and frontier LLM are interchangeable behind a `Protocol` so swapping Gemini for Groq is a one-file change.
- **CPU-runnable tests.** Unit + integration tests use a tiny stub model so CI never touches a GPU.
- **Reproducible bench.** `scripts/evaluate.py` regenerates `reports/bench.json` end-to-end with no manual edits.

---

## License

Code: MIT. ECTSum dataset retains its upstream licence — see [`docs/dataset.md`](docs/dataset.md) (TODO).
