# Training — Kaggle / Colab how-to

> Skeleton — fills in during Weekend 3.

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

## Expected wall-clock

- Kaggle T4 16 GB: ~90-120 min for 2 epochs over ~2k examples.
- Colab T4 12 hr session: same; leaves headroom for eval in the same session.

## Kaggle path

1. New Notebook → Accelerator: GPU T4 / P100 → Internet ON.
2. `git clone https://github.com/mathieu-calvo/EarningsLoRA.git`
3. `pip install -e ".[train,eval]"`
4. Set `WANDB_API_KEY` and `HF_TOKEN` as Kaggle secrets; expose as env vars.
5. Run `notebooks/03_qlora_training.ipynb`.

## Colab path

1. Runtime → Change runtime type → T4 GPU.
2. Mount Drive (optional) for checkpoint persistence across sessions.
3. Same `pip install` + same notebook.
4. `scripts/train.py --resume-from <checkpoint>` if the 12-hour session expires.

## Checkpointing

`SFTTrainer` saves every `eval_steps` to `runs/<timestamp>/checkpoint-*`.
Resume is supported; W&B run id is preserved across resumes.
