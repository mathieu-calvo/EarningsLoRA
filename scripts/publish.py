"""Publish artefacts to the HuggingFace Hub.

Three artefact types — adapter, dataset, Space — selectable independently:

    # Publish everything (skips parts whose source dir is missing)
    python scripts/publish.py --all --adapter-dir runs/latest/adapter

    # Just the adapter (also merges + writes a fresh model card from bench.json)
    python scripts/publish.py --adapter --adapter-dir runs/latest/adapter

    # Just the dataset
    python scripts/publish.py --dataset --dataset-dir data/processed/ectsum-chat

    # Just the Space (Gradio app)
    python scripts/publish.py --space --space-dir app/space

    # Dry run — render artefacts to disk without uploading
    python scripts/publish.py --all --adapter-dir runs/latest/adapter --dry-run

Adapter publish step is idempotent and gracefully no-ops with a warning if
`--adapter-dir` doesn't exist (so the script can be wired into CI before the
first training run).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from earningslora.config import get_settings
from earningslora.utils.hf_hub import (
    push_adapter,
    push_dataset,
    push_space,
    render_model_card,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Publish adapter + dataset + Space.")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--space", action="store_true")

    parser.add_argument("--adapter-dir", default="runs/latest/adapter")
    parser.add_argument("--dataset-dir", default="data/processed/ectsum-chat")
    parser.add_argument("--space-dir", default="app/space")

    parser.add_argument("--bench", default=None, help="Override Settings.bench_path.")
    parser.add_argument(
        "--adapter-repo",
        default=None,
        help="Override Settings.adapter_repo.",
    )
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--space-repo", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the model card to the adapter dir but skip Hub uploads.",
    )
    args = parser.parse_args()

    if not (args.all or args.adapter or args.dataset or args.space):
        parser.error("Pick at least one of --all / --adapter / --dataset / --space.")

    settings = get_settings()
    do_adapter = args.all or args.adapter
    do_dataset = args.all or args.dataset
    do_space = args.all or args.space

    bench_path = Path(args.bench) if args.bench else settings.bench_path

    rc = 0

    if do_adapter:
        adapter_dir = Path(args.adapter_dir)
        repo_id = args.adapter_repo or settings.adapter_repo
        if not adapter_dir.exists():
            print(
                f"[adapter] {adapter_dir} does not exist — skipping. "
                f"Train first (`python scripts/train.py`).",
                file=sys.stderr,
            )
        else:
            card = render_model_card(bench_path=bench_path, repo_id=repo_id)
            (adapter_dir / "README.md").write_text(card, encoding="utf-8")
            print(f"[adapter] Wrote model card to {adapter_dir / 'README.md'}.")
            if args.dry_run:
                print(f"[adapter] dry-run: would upload {adapter_dir} -> {repo_id}.")
            else:
                try:
                    url = push_adapter(adapter_dir, repo_id=repo_id, bench_path=bench_path, private=args.private)
                    print(f"[adapter] Pushed -> {url}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[adapter] Push failed: {exc}", file=sys.stderr)
                    rc = 1

    if do_dataset:
        dataset_dir = Path(args.dataset_dir)
        repo_id = args.dataset_repo or settings.dataset_repo
        if not dataset_dir.exists():
            print(
                f"[dataset] {dataset_dir} does not exist — skipping. "
                f"Build first (`python scripts/prepare_dataset.py`).",
                file=sys.stderr,
            )
        elif args.dry_run:
            print(f"[dataset] dry-run: would push {dataset_dir} -> {repo_id}.")
        else:
            try:
                url = push_dataset(dataset_dir, repo_id=repo_id, private=args.private)
                print(f"[dataset] Pushed -> {url}")
            except Exception as exc:  # noqa: BLE001
                print(f"[dataset] Push failed: {exc}", file=sys.stderr)
                rc = 1

    if do_space:
        space_dir = Path(args.space_dir)
        repo_id = args.space_repo or settings.space_repo
        if not space_dir.exists():
            print(f"[space] {space_dir} does not exist — skipping.", file=sys.stderr)
        elif args.dry_run:
            print(f"[space] dry-run: would push {space_dir} -> {repo_id}.")
        else:
            try:
                url = push_space(space_dir, repo_id=repo_id)
                print(f"[space] Pushed -> {url}")
            except Exception as exc:  # noqa: BLE001
                print(f"[space] Push failed: {exc}", file=sys.stderr)
                rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
