"""Train the LoRA adapter via SFTTrainer.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 3 --output-dir runs/exp2
    python scripts/train.py --resume-from runs/latest/checkpoint-200

Defaults pull from `Settings` so the no-arg invocation matches the documented
recipe (Llama 3.2 3B Instruct, 2 epochs, batch=1+grad-accum=8, max_seq_len=4096).
"""

from __future__ import annotations

import argparse
import logging
import sys

from earningslora.training.callbacks import configure_wandb
from earningslora.training.sft import train as run_train


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None, help="Override Settings.base_model.")
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/ectsum-chat",
        help="Path to the chat-formatted dataset (output of scripts/prepare_dataset.py).",
    )
    parser.add_argument("--output-dir", default="runs/latest")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    configure_wandb(run_name=args.wandb_run_name)

    adapter_path = run_train(
        base_model=args.base_model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        resume_from=args.resume_from,
    )
    print(f"Adapter saved to {adapter_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
