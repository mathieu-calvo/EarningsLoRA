"""Length-distribution and summary-style statistics for the dataset.

Used in `notebooks/01_dataset_exploration.ipynb` and as part of the
`scripts/prepare_dataset.py` summary that gets printed before pushing to Hub.

Implemented in Weekend 1.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable

from earningslora.data.chunk import approx_token_count


def _percentiles(values: list[int], qs: tuple[int, ...] = (50, 90, 99)) -> dict[str, int]:
    """Discrete-percentile helper. Empty list returns zeros."""
    if not values:
        return {f"p{q}": 0 for q in qs}
    sorted_vals = sorted(values)
    out: dict[str, int] = {}
    last_idx = len(sorted_vals) - 1
    for q in qs:
        idx = int(round(q / 100.0 * last_idx))
        idx = max(0, min(last_idx, idx))
        out[f"p{q}"] = sorted_vals[idx]
    return out


def _bullet_count(summary: str) -> int:
    """Count bullets (lines starting with '-', '*', '•', or '–') — 0 if none."""
    return sum(
        1
        for line in summary.splitlines()
        if line.lstrip().startswith(("-", "*", "•", "–"))
    )


def transcript_length_stats(
    rows: Iterable[dict],
    count_tokens: Callable[[str], int] | None = None,
) -> dict:
    """Token-length distribution of transcripts (n / mean / p50 / p90 / p99 / max)."""
    counter = count_tokens or approx_token_count
    lengths = [counter(row["transcript"]) for row in rows]
    if not lengths:
        return {"n": 0}
    return {
        "n": len(lengths),
        "mean": int(statistics.mean(lengths)),
        "max": max(lengths),
        **_percentiles(lengths),
    }


def summary_length_stats(
    rows: Iterable[dict],
    count_tokens: Callable[[str], int] | None = None,
) -> dict:
    """Bullet-count and summary-token-length distribution."""
    counter = count_tokens or approx_token_count
    rows = list(rows)
    token_lengths = [counter(row["summary"]) for row in rows]
    bullet_counts = [_bullet_count(row["summary"]) for row in rows]

    if not token_lengths:
        return {"n": 0}

    pct = _percentiles(token_lengths)
    return {
        "n": len(token_lengths),
        "tokens_mean": int(statistics.mean(token_lengths)),
        "tokens_max": max(token_lengths),
        "bullets_mean": round(statistics.mean(bullet_counts), 1),
        "bullets_max": max(bullet_counts),
        **{f"tokens_{k}": v for k, v in pct.items()},
    }
