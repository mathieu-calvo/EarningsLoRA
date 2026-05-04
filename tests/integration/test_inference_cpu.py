"""GPU-only smoke test for the inference path.

Marked `slow` so CI (CPU-only) skips it. Run locally with a T4/H100 via:
    pytest tests/integration/test_inference_cpu.py -m slow

Loads a tiny CausalLM, runs a single chat-template generation, asserts the
output is a non-empty string. Not a quality test — just a wiring smoke test
for `inference.load` + `inference.generate`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_generate_summary_smoke():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from earningslora.inference.generate import generate_summary

    # Tiny instruction-tuned model with a chat template.
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    out = generate_summary(
        model,
        tokenizer,
        "Q3 revenue was 1.2 billion. EPS came in at 2.45.",
        max_new_tokens=64,
    )
    assert isinstance(out, str)
    assert len(out.strip()) > 0
    del transformers
