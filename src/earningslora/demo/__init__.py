"""Framework-agnostic helpers shared by the Streamlit + Gradio demo apps."""

from earningslora.demo.shared import (
    GenerationResult,
    hf_inference_summarise,
    time_call,
    truncate_for_display,
)

__all__ = [
    "GenerationResult",
    "hf_inference_summarise",
    "time_call",
    "truncate_for_display",
]
