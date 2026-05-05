# Notebooks

Educational walkthroughs that import from `src/earningslora/` rather than
re-implementing the logic. Each one is runnable end-to-end.

| # | Notebook | Status | What it shows |
|---|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | Done | ECTSum EDA: token-length distributions, summary-style stats, train/val/test splits, frozen 50-row hold-out construction. |
| 02 | `02_baseline_zero_shot.ipynb` | Done | Zero-shot Llama 3.2 3B + Gemini 2.5 Flash on the hold-out; first ROUGE / numeric-recall numbers. Establishes the bar before training. |
| 03 | `03_qlora_training.ipynb` | Done — runs on Kaggle/Colab T4 | The training notebook. Parameterised at the top. Saves adapter to `runs/`. |

The CLI entrypoints (`scripts/prepare_dataset.py`, `scripts/train.py`,
`scripts/evaluate.py`, `scripts/publish.py`) cover the full pipeline without
needing further notebooks. Additional walkthroughs (inference & merge, full
bench, cost / latency analysis) may be added once a real training run has
produced a populated `bench.json` worth narrating.

Notebooks are kept light — narration + plots — and import the heavy lifting
from the package. This keeps `nbdiff` reviews tractable and the package the
single source of truth.
