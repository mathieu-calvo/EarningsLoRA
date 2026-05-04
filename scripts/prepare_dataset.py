"""Materialise ECTSum into a chat-template HF dataset and (optionally) push it.

Pipeline:
    1. Load upstream ECTSum (configurable via EARNINGSLORA_UPSTREAM_DATASET).
    2. Normalise columns to (transcript, summary).
    3. Truncate transcripts to fit `Settings.max_seq_len` via section-priority chunking.
    4. Format as chat-template messages.
    5. Print length-distribution stats.
    6. Save to local dir; optionally push to HF Hub.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --push --output-dir ./data/processed/ectsum-chat
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

from earningslora.config import get_settings
from earningslora.data.chunk import fit_to_budget
from earningslora.data.ectsum import (
    load_ectsum,
    make_holdout,
    materialise_local,
    to_chat_format,
)
from earningslora.data.stats import summary_length_stats, transcript_length_stats

logger = logging.getLogger("prepare_dataset")


def _truncate_to_budget(dataset_dict: DatasetDict, max_tokens: int) -> DatasetDict:
    """Apply section-aware truncation to every transcript."""

    def _fit(row):
        return {"transcript": fit_to_budget(row["transcript"], max_tokens=max_tokens)}

    out: dict[str, Dataset] = {}
    for split, ds in dataset_dict.items():
        out[split] = ds.map(_fit)
    return DatasetDict(out)


def _print_stats(dataset_dict: DatasetDict) -> dict:
    """Print + return per-split stats."""
    summary = {}
    for split, ds in dataset_dict.items():
        rows = list(ds)
        summary[split] = {
            "rows": len(rows),
            "transcript": transcript_length_stats(rows),
            "summary": summary_length_stats(rows),
        }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/processed/ectsum-chat",
        help="Local directory to materialise the chat-formatted dataset to.",
    )
    parser.add_argument(
        "--holdout-output",
        default="data/eval/holdout.json",
        help="Path to write the frozen hold-out manifest as JSON (transcripts + gold summaries).",
    )
    parser.add_argument("--push", action="store_true", help="Push to HF Hub after building.")
    parser.add_argument("--upstream", default=None, help="Override EARNINGSLORA_UPSTREAM_DATASET.")
    args = parser.parse_args()

    settings = get_settings()

    logger.info("Loading upstream dataset…")
    raw = load_ectsum(dataset_id=args.upstream)

    logger.info("Stats before truncation:")
    _print_stats(raw)

    logger.info("Truncating transcripts to fit max_seq_len=%d…", settings.max_seq_len)
    truncated = _truncate_to_budget(raw, max_tokens=settings.max_seq_len - 256)

    logger.info("Carving frozen hold-out: size=%d, seed=%d", settings.eval_holdout_size, settings.eval_seed)
    holdout = make_holdout(truncated, seed=settings.eval_seed, size=settings.eval_holdout_size)

    holdout_path = Path(args.holdout_output)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_payload = {
        "seed": settings.eval_seed,
        "size": len(holdout),
        "rows": [
            {"transcript": r["transcript"], "summary": r["summary"]}
            for r in holdout
        ],
    }
    holdout_path.write_text(json.dumps(holdout_payload, indent=2), encoding="utf-8")
    logger.info("Hold-out written to %s (%d rows)", holdout_path, len(holdout))

    logger.info("Formatting as chat templates…")
    chat = to_chat_format(truncated)

    logger.info("Saving to %s", args.output_dir)
    materialise_local(chat, args.output_dir)

    if args.push:
        try:
            from huggingface_hub import HfApi  # noqa: F401  # pragma: no cover

            chat.push_to_hub(settings.dataset_repo)
            logger.info("Pushed to %s", settings.dataset_repo)
        except Exception as exc:  # noqa: BLE001
            logger.error("Push failed: %s", exc)
            return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
