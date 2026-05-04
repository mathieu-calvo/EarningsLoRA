"""Run a model on the hold-out and persist predictions to disk.

Configuration-agnostic: accepts any callable mapping `transcript -> summary`,
so the same code path runs the base model, the fine-tuned model, and the
frontier baseline. Output schema is a JSONL file at `<output_dir>/<name>.jsonl`
with one row per hold-out example:
    {"transcript": ..., "reference": ..., "prediction": ..., "latency_ms": ...}

Implemented in Weekend 2.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


def run_holdout(
    name: str,
    summarise: Callable[[str], str],
    holdout: Iterable[dict],
    output_dir: Path | str,
) -> Path:
    """Run `summarise` over the hold-out and save predictions to JSONL.

    `holdout` is any iterable of dicts with `transcript` + `summary` keys
    (the canonical schema after `data.ectsum.load_ectsum`).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"

    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in holdout:
            transcript = row["transcript"]
            reference = row["summary"]

            t0 = time.perf_counter()
            try:
                prediction = summarise(transcript)
                error = None
            except Exception as exc:  # noqa: BLE001
                logger.exception("summarise failed for row %d", n)
                prediction = ""
                error = str(exc)
            latency_ms = (time.perf_counter() - t0) * 1000

            f.write(
                json.dumps(
                    {
                        "transcript": transcript,
                        "reference": reference,
                        "prediction": prediction,
                        "latency_ms": round(latency_ms, 1),
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1

    logger.info("Wrote %d rows to %s", n, output_path)
    return output_path


def load_predictions(predictions_path: Path | str) -> list[dict]:
    """Load a JSONL produced by `run_holdout`."""
    out = []
    with Path(predictions_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
