# Notebooks

Educational walkthroughs that import from `src/earningslora/` rather than
re-implementing the logic. Each one is runnable end-to-end.

| # | Notebook | Lands in | What it shows |
|---|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | Weekend 1 | ECTSum EDA: token-length distributions, summary-style stats, train/val/test splits, frozen 50-row hold-out construction. |
| 02 | `02_baseline_zero_shot.ipynb` | Weekend 2 | Zero-shot Llama 3.2 3B + Gemini 2.5 Flash on the hold-out; first ROUGE / numeric-recall numbers. Establishes the bar before training. |
| 03 | `03_qlora_training.ipynb` | Weekend 3 | The training notebook. Kaggle/Colab-runnable, parameterised at the top. Saves adapter to `runs/`. |
| 04 | `04_inference_and_merge.ipynb` | Weekend 4 | Load adapter, sample outputs, merge into base, push to Hub with model card. |
| 05 | `05_evaluation.ipynb` | Weekend 4 | Full bench: ROUGE + numeric-recall + LLM-as-judge across all three configurations. Regenerates `reports/bench.json`. |
| 06 | `06_cost_latency_analysis.ipynb` | Weekend 5 | $/1M tokens + ms/req, base vs FT vs frontier. The "production fine-tuning crossover" chart. |

Notebooks are kept light — narration + plots — and import the heavy lifting
from the package. This keeps `nbdiff` reviews tractable and the package the
single source of truth.
