# Reports

| File | Purpose |
|---|---|
| `bench.json` | Latest eval table — base vs FT vs frontier across all metrics. Regenerated end-to-end by `scripts/evaluate.py`. The README headline table reads from this file. |
| `examples/` | 5 cherry-picked + 5 random side-by-side outputs (markdown). Useful for the README and the model card. |

Both are committed (small) so the repo is self-contained — no need to run a
training job to read the bench numbers.
