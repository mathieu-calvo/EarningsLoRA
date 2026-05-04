# Data

This directory holds **pointers**, not raw text. The ECTSum dataset is large and
licence-restricted; the actual transcripts and summaries live on HF Hub at
`mathieu-calvo/ectsum-chat` (built from the upstream dataset by
`scripts/prepare_dataset.py`).

| Subdir | Contents |
|---|---|
| `eval/` | The 50-row frozen hold-out manifest (split id + indices). Materialise via `data.ectsum.make_holdout()`. |

Anything matching `data/raw/` or `data/processed/` is `.gitignore`-d.
