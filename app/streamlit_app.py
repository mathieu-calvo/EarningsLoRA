"""Streamlit Cloud demo: paste a transcript, see base vs FT vs Gemini side-by-side.

Streamlit Cloud has 1 GB RAM, so we don't load the model in-process. We call the
fine-tuned model via the HF Inference API and call Gemini via Google AI Studio.

Stub — implemented in Weekend 5.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="EarningsLoRA — demo", layout="wide")
st.title("EarningsLoRA")
st.caption("Fine-tuned 3B Llama vs zero-shot base vs Gemini 2.5 Flash on earnings call transcripts.")

st.info("App lands in Weekend 5. Skeleton only.")
