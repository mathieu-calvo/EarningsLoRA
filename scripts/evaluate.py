"""Run the bench (base vs FT vs frontier) and write reports/bench.json.

Usage:
    python scripts/evaluate.py [--configs base,ft,frontier] [--output reports/bench.json]

Stub — implemented in Weekend 4.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="base,ft,frontier")
    parser.add_argument("--output", default="reports/bench.json")
    args = parser.parse_args()
    raise NotImplementedError(f"Implemented in Weekend 4. (args={args!r})")


if __name__ == "__main__":
    main()
