"""Tiny-model CPU inference sanity test.

Loads `sshleifer/tiny-gpt2` (≈10 MB), runs a single generation, asserts shape and
non-empty output. Marked `slow` so it stays out of the default CI run; CI calls
`pytest -m "not slow"`. Lifts as a regression smoke test for the inference path
once Weekend 3 lands.
"""

import pytest

pytestmark = pytest.mark.slow


def test_tiny_model_generates_non_empty():
    pytest.skip("Implemented alongside inference/generate.py in Weekend 3.")
