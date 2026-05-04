"""Merge the LoRA adapter and publish to HuggingFace Hub with a model card.

Usage:
    python scripts/publish.py --adapter-dir runs/latest --repo mathieu-calvo/llama-3.2-3b-earningslora

Stub — implemented in Weekend 4.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=False, default=None)
    parser.add_argument("--repo", required=False, default=None)
    parser.add_argument("--bench", default="reports/bench.json")
    args = parser.parse_args()
    raise NotImplementedError(f"Implemented in Weekend 4. (args={args!r})")


if __name__ == "__main__":
    main()
