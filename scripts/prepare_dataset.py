"""Materialise ECTSum into a chat-template HF dataset and (optionally) push it.

Usage:
    python scripts/prepare_dataset.py [--push]

Stub — implemented in Weekend 1.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Push to HF Hub after building.")
    args = parser.parse_args()
    raise NotImplementedError(f"Implemented in Weekend 1. (args={args!r})")


if __name__ == "__main__":
    main()
