# Training — Kaggle / Colab how-to

> First time setting up accounts + keys + the Kaggle environment? Follow
> [`docs/setup.md`](setup.md) end-to-end. This page is the reference for the
> training recipe itself.

## Hyperparameters (default)

| Parameter | Value | Notes |
|---|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` | Qwen 2.5 3B Instruct as drop-in alt. |
| Quantisation | 4-bit NF4 (bitsandbytes) | Required to fit T4 16 GB. |
| LoRA r / α / dropout | 16 / 32 / 0.05 | ~50 MB adapter. |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full attention + MLP coverage. |
| Max seq len (train) | 4096 | Section-truncated by `data/chunk.py`. |
| Max seq len (eval) | 8192 | Allowed; eval is offline. |
| Per-device batch | 1 | + gradient accumulation 8 → effective batch 8. |
| Learning rate | 2e-4 | Cosine, warmup ratio 0.03. |
| Epochs | 2 | Small adapter + ~2k examples. |
| Gradient checkpointing | on | Required for VRAM headroom. |

All defaults live in `Settings` (`src/earningslora/config.py`); CLI flags on
`scripts/train.py` override per-run.

## Expected wall-clock

- Kaggle T4 16 GB: ~90-120 min for 2 epochs over ~2k examples.
- Colab T4 12 hr session: same; leaves headroom for eval in the same session.

## Kaggle path

1. New Notebook → Accelerator: GPU T4 / P100 → Internet ON.
2. `git clone https://github.com/mathieu-calvo/EarningsLoRA.git`
3. `pip install -e ".[train,eval]"`
4. Set `WANDB_API_KEY` and `HF_TOKEN` as Kaggle secrets; expose as env vars.
5. Run `notebooks/03_qlora_training.ipynb` (or `python scripts/prepare_dataset.py && python scripts/train.py`).

## Colab path

1. Runtime → Change runtime type → T4 GPU.
2. Mount Drive (optional) for checkpoint persistence across sessions.
3. Same `pip install` + same notebook.
4. `scripts/train.py --resume-from <checkpoint>` if the 12-hour session expires.

## Checkpointing

`SFTTrainer` saves every `eval_steps` to `runs/<timestamp>/checkpoint-*`.
Resume is supported; W&B run id is preserved across resumes.

## After training

Once the run finishes and `runs/latest/adapter/` exists, the rest of the
pipeline runs locally on CPU (no GPU needed, since the harness uses cached
predictions and the judge calls Gemini):

```bash
python scripts/evaluate.py --adapter-dir runs/latest/adapter   # writes reports/bench.json + updates README
python scripts/publish.py --all --adapter-dir runs/latest/adapter   # adapter + dataset + Space to the Hub
```

For the Space's merged-bf16 model upload, run the merge inside the same GPU
session (it needs to load the bf16 base):

```python
from earningslora.inference.merge import merge_and_save
merge_and_save("meta-llama/Llama-3.2-3B-Instruct", "runs/latest/adapter", "runs/latest/merged")
```

then push the merged dir as a separate model repo (or replace the adapter repo
contents with the merged weights).
