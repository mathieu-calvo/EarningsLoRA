"""Zero-shot frontier baseline: Gemini 2.5 Flash on the same hold-out.

Same prompt template as SFT (see `utils.prompts`) so the comparison is apples
to apples. Sets the upper bound the fine-tuned 3B model is trying to approach.
Calls are cached to SQLite — reruns of `scripts/evaluate.py` don't re-spend the
daily free-tier quota.

Implemented in Weekend 2.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from earningslora.config import get_settings
from earningslora.utils.cache import JsonCache
from earningslora.utils.prompts import SYSTEM_PROMPT, render_user_prompt

logger = logging.getLogger(__name__)

_NAMESPACE = "frontier_baseline_v1"


def _cache(cache_dir: Path | None = None) -> JsonCache:
    settings = get_settings()
    base = Path(cache_dir) if cache_dir else settings.cache_dir
    return JsonCache(base / "frontier_cache.db")


def _generate_with_gemini(transcript: str, model: str) -> str:
    """One Gemini call. Imported lazily so the rest of the package doesn't pull `google-generativeai`."""
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set; cannot call Gemini.")
    genai.configure(api_key=api_key)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{render_user_prompt(transcript)}"
    response = genai.GenerativeModel(model).generate_content(
        full_prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 512},
    )
    return response.text or ""


def frontier_summary(transcript: str, model: str | None = None) -> str:
    """Generate a zero-shot summary with the frontier model. Cached on disk."""
    settings = get_settings()
    model = model or settings.frontier_model
    cache = _cache()

    payload = {"model": model, "transcript": transcript}
    cached = cache.get(_NAMESPACE, payload)
    if cached is not None:
        return cached

    out = _generate_with_gemini(transcript, model)
    cache.set(_NAMESPACE, payload, out)
    return out


def frontier_batch(transcripts: list[str], model: str | None = None) -> list[str]:
    """Batched zero-shot summaries. Sequential under the hood (free tier RPS is the bottleneck)."""
    return [frontier_summary(t, model=model) for t in transcripts]
