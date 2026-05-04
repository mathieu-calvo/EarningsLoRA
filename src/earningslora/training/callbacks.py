"""Training callbacks: W&B project setup and tiny helpers.

`SFTTrainer` already handles W&B integration when `report_to="wandb"` is set;
this module only owns the project-name / run-name plumbing so the SFT path
stays declarative.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_wandb(
    project: str | None = None,
    run_name: str | None = None,
) -> bool:
    """Configure W&B if installed. Returns True iff W&B is reachable.

    - If `wandb` isn't installed → returns False, training continues without it.
    - If `WANDB_API_KEY` isn't set → W&B will run in offline mode (anonymous local
      logs) which is fine for portfolio runs the user later promotes.
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.info("wandb not installed; skipping W&B integration.")
        return False

    project = project or os.environ.get("WANDB_PROJECT", "earningslora")
    os.environ["WANDB_PROJECT"] = project
    if run_name:
        os.environ["WANDB_NAME"] = run_name

    if not os.environ.get("WANDB_API_KEY"):
        logger.info(
            "WANDB_API_KEY not set; runs will be offline. "
            "Set the key (or `wandb login`) to push runs to your W&B project."
        )
    return True
