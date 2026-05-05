"""Run the bench (base / FT / frontier) and write `reports/bench.json`.

Examples:
    # Full bench (after training has produced an adapter)
    python scripts/evaluate.py --adapter-dir runs/latest/adapter

    # Only base + frontier (e.g. before training is run)
    python scripts/evaluate.py --configs base,frontier

    # Skip the LLM judge (saves the daily Gemini quota during dev)
    python scripts/evaluate.py --configs base --no-judge

    # Regenerate the README headline table from existing bench.json
    python scripts/evaluate.py --update-readme-only

The script is idempotent: harness predictions land in `runs/eval/<config>.jsonl`
and Gemini calls are SQLite-cached, so reruns cost zero quota.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from earningslora.config import get_settings
from earningslora.evaluation.bench import (
    ALL_CONFIGS,
    regenerate_readme_table,
    run_bench,
)


def _parse_configs(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL_CONFIGS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown configs: {bad}. Choose from {list(ALL_CONFIGS)}."
        )
    return parts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=_parse_configs,
        default=list(ALL_CONFIGS),
        help="Comma-separated subset of base,ft,frontier (default: all).",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Path to a trained adapter directory. Required for --configs ft.",
    )
    parser.add_argument(
        "--holdout-path",
        default=None,
        help="Override the hold-out manifest path (default: Settings.holdout_manifest).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the bench.json output path (default: Settings.bench_path).",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Where per-config prediction JSONL files land (default: Settings.eval_dir).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge step.",
    )
    parser.add_argument(
        "--no-readme-update",
        action="store_true",
        help="Skip regenerating the README headline table.",
    )
    parser.add_argument(
        "--update-readme-only",
        action="store_true",
        help="Regenerate the README from existing bench.json and exit.",
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.update_readme_only:
        bench_path = Path(args.output) if args.output else settings.bench_path
        changed = regenerate_readme_table(bench_path=bench_path)
        print(f"README table {'updated' if changed else 'unchanged'} from {bench_path}.")
        return 0

    output_path = run_bench(
        output_path=args.output,
        holdout_path=args.holdout_path,
        adapter_dir=args.adapter_dir,
        configs=args.configs,
        eval_dir=args.eval_dir,
        skip_judge=args.no_judge,
        settings=settings,
    )
    print(f"Bench written to {output_path}")

    if not args.no_readme_update:
        try:
            changed = regenerate_readme_table(bench_path=output_path)
            print(f"README table {'updated' if changed else 'unchanged'}.")
        except (FileNotFoundError, ValueError) as exc:
            print(f"README table not updated: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
