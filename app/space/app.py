"""HuggingFace Spaces (ZeroGPU) entrypoint — Gradio UI.

Loads the merged fine-tuned model from `Settings.adapter_repo` (a model repo
with merged bf16 weights, produced by `scripts/publish.py`). Inference happens
inside `@spaces.GPU`-decorated functions so the GPU slot is only acquired for
the actual generate call — idle Spaces don't burn quota.

Three side-by-side columns: zero-shot base, fine-tuned, Gemini 2.5 Flash.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import gradio as gr

try:
    import spaces  # ZeroGPU runtime; only present on HF Spaces

    GPU_DECORATOR = spaces.GPU(duration=90)
except ImportError:  # local dev fallback — no decorator
    def GPU_DECORATOR(fn):
        return fn

from earningslora.config import get_settings
from earningslora.evaluation.frontier_baseline import frontier_summary

logger = logging.getLogger(__name__)


SAMPLE_TRANSCRIPT = (
    "Operator: Welcome to the Q3 earnings call. I will now turn it over to the CEO.\n"
    "CEO: Thank you. We had a strong quarter — revenue came in at $3.2 billion, up 18% year-over-year. "
    "Gross margin expanded 220 basis points to 64.5%. We are raising full-year guidance to $12.8 billion at the midpoint. "
    "Operating cash flow was $920 million. We returned $400 million to shareholders via buybacks. "
    "Looking forward, we see continued strength in our enterprise segment, with bookings up 32% sequentially. "
    "We are guiding Q4 revenue to $3.4 to $3.5 billion."
)


@lru_cache(maxsize=1)
def _load_models():
    """Load base + fine-tuned merged model + tokenizer once per worker."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = get_settings()
    base_id = settings.base_model
    ft_id = os.environ.get("FT_MODEL_ID", settings.adapter_repo)

    logger.info("Loading base %s", base_id)
    base_tok = AutoTokenizer.from_pretrained(base_id)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    logger.info("Loading fine-tuned %s", ft_id)
    ft_tok = AutoTokenizer.from_pretrained(ft_id)
    if ft_tok.pad_token is None:
        ft_tok.pad_token = ft_tok.eos_token
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return (base_model, base_tok), (ft_model, ft_tok)


@GPU_DECORATOR
def _generate_local(model, tokenizer, transcript: str) -> str:
    """One generate() call. Inside @spaces.GPU so the GPU slot is acquired."""
    from earningslora.inference.generate import generate_summary

    return generate_summary(model, tokenizer, transcript)


def summarise_all(transcript: str) -> tuple[str, str, str]:
    """Run all three configurations and return (base, ft, frontier) summaries."""
    if not transcript or not transcript.strip():
        return ("Paste a transcript first.",) * 3

    (base_model, base_tok), (ft_model, ft_tok) = _load_models()

    try:
        base_out = _generate_local(base_model, base_tok, transcript)
    except Exception as exc:  # noqa: BLE001
        base_out = f"[base] error: {exc}"

    try:
        ft_out = _generate_local(ft_model, ft_tok, transcript)
    except Exception as exc:  # noqa: BLE001
        ft_out = f"[ft] error: {exc}"

    try:
        frontier_out = frontier_summary(transcript)
    except Exception as exc:  # noqa: BLE001
        frontier_out = f"[frontier] error: {exc} (set GOOGLE_API_KEY)"

    return base_out, ft_out, frontier_out


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EarningsLoRA", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# EarningsLoRA\n"
            "QLoRA-fine-tuned **Llama 3.2 3B** vs zero-shot base vs **Gemini 2.5 Flash** on earnings calls. "
            "Source + bench harness: [github.com/mathieu-calvo/EarningsLoRA](https://github.com/mathieu-calvo/EarningsLoRA)."
        )

        transcript = gr.Textbox(
            label="Earnings call transcript",
            value=SAMPLE_TRANSCRIPT,
            lines=14,
            placeholder="Paste an earnings call transcript here…",
        )
        run = gr.Button("Summarise", variant="primary")

        with gr.Row():
            base_out = gr.Textbox(label="Base (zero-shot)", lines=12, show_copy_button=True)
            ft_out = gr.Textbox(label="Fine-tuned (this repo)", lines=12, show_copy_button=True)
            frontier_out = gr.Textbox(label="Gemini 2.5 Flash", lines=12, show_copy_button=True)

        gr.Markdown(
            "_Cold-start ~15-30s while the Space loads weights; warm < 5s. "
            "The fine-tuned column comes from a merged bf16 build of the LoRA adapter._"
        )

        run.click(
            summarise_all,
            inputs=[transcript],
            outputs=[base_out, ft_out, frontier_out],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
