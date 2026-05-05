"""ROUGE-1 / ROUGE-2 / ROUGE-L wrapper around `rouge-score`.

Mean F1 across the prediction/reference list. Reference impl rather than a
hand-rolled scorer so the numbers are directly comparable to other papers.

Implemented in Weekend 2.
"""

from __future__ import annotations

import statistics

from rouge_score import rouge_scorer


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Return mean ROUGE-1 / ROUGE-2 / ROUGE-L F1 across the inputs."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length "
            f"(got {len(predictions)} vs {len(references)})"
        )
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references, strict=True):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(statistics.mean(r1), 4),
        "rouge2": round(statistics.mean(r2), 4),
        "rougeL": round(statistics.mean(rl), 4),
        "n": len(predictions),
    }
