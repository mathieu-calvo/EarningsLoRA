"""Streamlit Cloud demo: paste a transcript, see base / FT / Gemini side-by-side.

Streamlit Cloud only ships 1 GB of RAM, so the model is *not* loaded in-process.
We call:
  - Fine-tuned model via the HuggingFace Inference API (`HF_TOKEN` required)
  - Gemini 2.5 Flash via Google AI Studio (`GOOGLE_API_KEY` required)
  - The "base" column reuses the HF Inference API against `Settings.base_model`

If a key is missing, that column gracefully renders an instructive error.
"""

from __future__ import annotations

import streamlit as st

from earningslora.config import get_settings
from earningslora.demo import (
    GenerationResult,
    hf_inference_summarise,
    time_call,
    truncate_for_display,
)
from earningslora.evaluation.frontier_baseline import frontier_summary

st.set_page_config(page_title="EarningsLoRA — demo", layout="wide")

settings = get_settings()

st.title("EarningsLoRA")
st.caption(
    "Fine-tuned Llama 3.2 3B vs zero-shot base vs Gemini 2.5 Flash on earnings call transcripts. "
    "[Source](https://github.com/mathieu-calvo/EarningsLoRA)."
)

DEFAULT = (
    "Operator: Welcome to the Q3 earnings call. I will now turn it over to the CEO.\n"
    "CEO: Thank you. We had a strong quarter — revenue came in at $3.2 billion, up 18% year-over-year. "
    "Gross margin expanded 220 basis points to 64.5%. We are raising full-year guidance to $12.8 billion at the midpoint. "
    "Operating cash flow was $920 million. We returned $400 million to shareholders via buybacks. "
    "Looking forward, we see continued strength in our enterprise segment, with bookings up 32% sequentially. "
    "We are guiding Q4 revenue to $3.4 to $3.5 billion."
)

transcript = st.text_area(
    "Earnings call transcript",
    value=DEFAULT,
    height=320,
)

run = st.button("Summarise", type="primary", use_container_width=True)


def _render_column(title: str, result: GenerationResult) -> None:
    st.subheader(title)
    if result.error:
        st.error(result.error)
        return
    st.markdown(result.text)
    st.caption(
        f"latency: {result.latency_ms:.0f} ms · "
        f"numeric_recall: {result.numeric_recall:.2f}"
    )


if run and transcript.strip():
    with st.spinner("Running base, FT, and Gemini in parallel… (sequential under the hood)"):
        base_result = time_call(lambda t: hf_inference_summarise(t, settings.base_model), transcript)
        ft_result = time_call(lambda t: hf_inference_summarise(t, settings.adapter_repo), transcript)
        frontier_result = time_call(frontier_summary, transcript)

    col_base, col_ft, col_front = st.columns(3)
    with col_base:
        _render_column(f"Base — {settings.base_model.split('/')[-1]}", base_result)
    with col_ft:
        _render_column("Fine-tuned (this repo)", ft_result)
    with col_front:
        _render_column(f"Frontier — {settings.frontier_model}", frontier_result)
elif run:
    st.warning("Paste a transcript first.")

with st.expander("Transcript preview (truncated for display)"):
    st.code(truncate_for_display(transcript))

st.divider()
st.caption(
    "Set `HF_TOKEN` and `GOOGLE_API_KEY` as Streamlit Cloud secrets. "
    "For lower latency and live cost/latency display, see the ZeroGPU Space at "
    "[huggingface.co/spaces/" + settings.space_repo + "](https://huggingface.co/spaces/" + settings.space_repo + ")."
)
