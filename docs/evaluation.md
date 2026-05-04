# Evaluation

> Skeleton — fills in during Weekend 2 and Weekend 4.

## The frozen hold-out

A single 50-row hold-out, materialised once with `Settings.eval_seed=42` from the
ECTSum test split. Same set used across all three configurations (base / FT /
frontier). Stored on HF Hub at `mathieu-calvo/ectsum-chat:holdout`.

## Metrics

### ROUGE-1 / ROUGE-2 / ROUGE-L

Mechanical baseline. Useful as a smoke test, weak as an absolute quality signal —
ROUGE rewards lexical overlap, which mostly tracks "did the model copy the right
phrases" rather than "is this a good summary."

### Numeric recall (project-specific)

Fraction of numbers in the generated summary that also appear in the source
transcript. Implemented in `evaluation/numeric_recall.py`. Penalises hallucinated
figures — the most consequential failure mode for finance summarisation. Numbers
are normalised across "$1.2 billion" / "1.2B" / "1,200,000,000" / "1.2 bn" so unit
formatting doesn't drive the score.

This metric is the project's most differentiated piece. Off-the-shelf eval libs
ship ROUGE / BLEU / BERTScore but not finance-specific factuality.

### LLM-as-judge (Gemini 2.5 Flash)

Four rubrics on a 1-5 scale: faithfulness, completeness, conciseness, style.
Each pair (FT vs base, FT vs frontier) is judged head-to-head with deterministic
prompts (temperature=0). Calls cached to SQLite so reruns are free. Rubric prompt
version-pinned in `utils/prompts.py:JUDGE_RUBRIC` — bumping the rubric version
invalidates the cache.

### Cost / latency

- Cost: $/1M input tokens (training) + $/1M output tokens (inference).
- Latency: median ms/request on a T4 for the local model; same on the API for
  Gemini Flash.

These two columns are why "fine-tuning small open model" wins — quality alone
isn't the story.

## Bench output

`scripts/evaluate.py` writes `reports/bench.json`. The README headline table is
regenerated from this file. Never hand-edit the table.
