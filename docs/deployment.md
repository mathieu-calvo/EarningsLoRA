# Deployment

> Skeleton — fills in during Weekend 5.

## HuggingFace Hub

Three artefacts published per release:

1. **Adapter** — `mathieu-calvo/llama-3.2-3b-earningslora`
   - LoRA adapter weights (~50 MB) + tokenizer config.
   - Model card includes the eval table from `reports/bench.json`.
2. **Dataset** — `mathieu-calvo/ectsum-chat`
   - Chat-template-formatted ECTSum splits + the 50-row frozen hold-out.
3. **Space** — `mathieu-calvo/EarningsLoRA`
   - Gradio demo on ZeroGPU. Code lives in `app/space/`.

`scripts/publish.py` orchestrates all three.

## HuggingFace Spaces (ZeroGPU)

ZeroGPU gives a free ephemeral H200 slot per request. The `@spaces.GPU` decorator
on the inference function acquires the GPU only during `model.generate()`, so
idle Spaces don't burn quota.

Cold-start: ~15-30s while the Space spins up and loads the merged model.
Warm: < 5s for a typical transcript.

## Streamlit Community Cloud (fallback)

Free tier has 1 GB RAM — too small to load the model in-process. The Streamlit
demo calls the HuggingFace Inference API instead. Slower than the ZeroGPU Space
but works under HF API free-tier rate limits for a portfolio demo.

## Secrets

| Where | Secret | Purpose |
|---|---|---|
| HF Spaces | `HF_TOKEN` | Pulls the private/gated base model if needed. |
| Streamlit Cloud | `HF_TOKEN`, `GOOGLE_API_KEY` | HF Inference API + Gemini side-by-side. |
| GitHub Actions | none | CI is CPU-only and doesn't hit any API. |
