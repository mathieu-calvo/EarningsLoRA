# Reports

| File | Purpose |
|---|---|
| `bench.json` | Latest eval table — base vs FT vs frontier across all metrics. Regenerated end-to-end by `scripts/evaluate.py`. The README headline table reads from this file. Schema: see `docs/evaluation.md`. |
| `examples/` | 5 cherry-picked + 5 random side-by-side outputs (markdown). Useful for the README and the model card. Materialised by hand from `runs/eval/<config>.jsonl` after the first bench run. |

Both are committed (small) so the repo is self-contained — no need to run a
training job to read the bench numbers.

## How `bench.json` is regenerated

```bash
# After a training run produces runs/latest/adapter/
python scripts/evaluate.py --adapter-dir runs/latest/adapter
```

This:

1. Generates predictions for each requested config to `runs/eval/<config>.jsonl`.
2. Computes ROUGE / numeric-recall / latency per config.
3. Runs the LLM judge head-to-head for each pair (cached).
4. Writes `reports/bench.json` and updates the README headline table between the `<!-- BENCH:HEADLINE:* -->` markers.
