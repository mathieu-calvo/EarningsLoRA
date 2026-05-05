# Deployment

## HuggingFace Hub

Three artefacts published per release, all by `scripts/publish.py`:

1. **Adapter** — `mathieu-calvo/llama-3.2-3b-earningslora`
   - LoRA adapter weights (~50 MB) + tokenizer config.
   - Model card auto-rendered from `reports/bench.json` so it always matches the latest eval table.
2. **Dataset** — `mathieu-calvo/ectsum-chat`
   - Chat-template-formatted ECTSum splits + the 50-row frozen hold-out.
3. **Space** — `mathieu-calvo/EarningsLoRA`
   - Gradio demo on ZeroGPU. Code lives in `app/space/` and is uploaded as-is.

`scripts/publish.py` is idempotent and selectable per artefact:

```bash
# Everything (skips parts whose source dir is missing)
python scripts/publish.py --all --adapter-dir runs/latest/adapter

# Just the adapter (also re-renders the model card from bench.json)
python scripts/publish.py --adapter --adapter-dir runs/latest/adapter

# Dry run — render artefacts to disk without uploading
python scripts/publish.py --all --adapter-dir runs/latest/adapter --dry-run
```

Required env vars: `HF_TOKEN` (with write scope on the target repos).

## HuggingFace Spaces (ZeroGPU)

ZeroGPU gives a free ephemeral H200 slot per request. The `@spaces.GPU` decorator
on `_generate_local` (`app/space/app.py`) acquires the GPU only during
`model.generate()`, so idle Spaces don't burn quota.

Cold-start: ~15-30s while the Space spins up and loads weights.
Warm: < 5s for a typical transcript.

The Space loads:
- Base: `Settings.base_model` (Llama 3.2 3B Instruct, bf16).
- FT: `Settings.adapter_repo` — expected to contain *merged* bf16 weights (run `merge_and_save` after training; see `docs/training.md`).
- Frontier: Gemini 2.5 Flash via `GOOGLE_API_KEY` (set as a Space secret).

`app/space/requirements.txt` pulls the package from GitHub so the Space code
shares prompts, numeric-recall, and `generate_summary` with the rest of the
repo (single source of truth for the chat template).

## Streamlit Community Cloud (fallback)

Free tier has 1 GB RAM — too small to load the model in-process. The Streamlit
demo (`app/streamlit_app.py`) calls:

- HF Inference API for both base and FT (using `HF_TOKEN`).
- Google AI Studio for Gemini (using `GOOGLE_API_KEY`).

Slower than the ZeroGPU Space but works under HF API free-tier rate limits for
a portfolio demo.

## Secrets

| Where | Secret | Purpose |
|---|---|---|
| HF Spaces | `HF_TOKEN` | Pulls private/gated base model if needed; fetches from the source GitHub repo on cold-start. |
| HF Spaces | `GOOGLE_API_KEY` | Frontier-baseline column in the demo. |
| Streamlit Cloud | `HF_TOKEN`, `GOOGLE_API_KEY` | HF Inference API + Gemini side-by-side. |
| GitHub Actions | none | CI is CPU-only and doesn't hit any API. |
| Local dev | `.env` (see `.env.example`) | Loaded by `pydantic-settings`; same names as Spaces. |
