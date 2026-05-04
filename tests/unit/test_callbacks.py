"""Tests for `training/callbacks.py`.

Doesn't require torch/peft/trl, runs cleanly on the CPU CI runner.
"""

from __future__ import annotations

import sys


def test_configure_wandb_returns_false_without_wandb(monkeypatch):
    # Force the import to fail by hiding the module from sys.modules.
    monkeypatch.setitem(sys.modules, "wandb", None)

    from earningslora.training.callbacks import configure_wandb

    assert configure_wandb() is False


def test_configure_wandb_sets_project(monkeypatch):
    """When wandb is importable, project + run name should land in the env."""

    class _FakeWandb:
        pass

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_NAME", raising=False)

    from earningslora.training.callbacks import configure_wandb

    assert configure_wandb(project="my-proj", run_name="my-run") is True
    import os

    assert os.environ["WANDB_PROJECT"] == "my-proj"
    assert os.environ["WANDB_NAME"] == "my-run"


def test_configure_wandb_uses_env_project_default(monkeypatch):
    class _FakeWandb:
        pass

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.setenv("WANDB_PROJECT", "from-env")

    from earningslora.training.callbacks import configure_wandb

    configure_wandb()
    import os

    assert os.environ["WANDB_PROJECT"] == "from-env"
