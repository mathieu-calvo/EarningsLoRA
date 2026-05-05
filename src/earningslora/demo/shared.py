"""Framework-agnostic helpers used by both demo entrypoints.

Keeps no Streamlit/Gradio imports so either app can pull from here without
dragging the other framework in.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from earningslora.config import get_settings
from earningslora.evaluation.numeric_recall import numeric_recall
from earningslora.utils.prompts import SYSTEM_PROMPT, render_user_prompt


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_ms: float
    numeric_recall: float
    error: str | None = None


def time_call(fn: Callable[[str], str], transcript: str) -> GenerationResult:
    """Run a `transcript -> summary` callable, capture latency + numeric_recall."""
    t0 = time.perf_counter()
    try:
        text = fn(transcript)
        error: str | None = None
    except Exception as exc:  # noqa: BLE001
        text = ""
        error = str(exc)
    latency_ms = (time.perf_counter() - t0) * 1000
    nr = numeric_recall(transcript, text).recall if text else 0.0
    return GenerationResult(
        text=text,
        latency_ms=round(latency_ms, 1),
        numeric_recall=round(nr, 3),
        error=error,
    )


def hf_inference_summarise(transcript: str, model_id: str | None = None) -> str:
    """Call the HF Inference API for a summary. Used by the Streamlit fallback.

    Requires `HF_TOKEN` in the environment. Returns the assistant's reply.
    """
    import requests

    settings = get_settings()
    model_id = model_id or settings.adapter_repo
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var is not set; cannot call HF Inference API.")

    url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(transcript)},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def truncate_for_display(text: str, max_chars: int = 4000) -> str:
    """Truncate a transcript for the UI text-area preview without surprising the user."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n…[{len(text) - max_chars} more chars truncated for display]…"
