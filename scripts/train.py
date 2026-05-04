"""Train the LoRA adapter via SFTTrainer.

Usage:
    python scripts/train.py \
        --base-model meta-llama/Llama-3.2-3B-Instruct \
        --output-dir runs/exp1 \
        --max-steps 800

Stub — implemented in Weekend 3.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", default="runs/latest")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    raise NotImplementedError(f"Implemented in Weekend 3. (args={args!r})")


if __name__ == "__main__":
    main()
