"""LLM-as-judge using Gemini 2.5 Flash via Google AI Studio (free tier).

Compares two candidate summaries on faithfulness / completeness / conciseness /
style and returns structured per-rubric scores plus a winner. Calls are cached
to SQLite keyed by (transcript, summary_a, summary_b, rubric_version) so reruns
don't re-spend the daily quota.

The judge is intentionally biased-blind via order-randomisation: pass
`shuffle=True` to `judge()` and the function will randomly swap A and B before
calling, then unswap the result. Use this when comparing two summaries from
different configurations to avoid position bias.

Implemented in Weekend 2.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from earningslora.config import get_settings
from earningslora.utils.cache import JsonCache
from earningslora.utils.prompts import render_judge_prompt

logger = logging.getLogger(__name__)

_NAMESPACE = "judge_v1"


@dataclass(frozen=True)
class JudgeVerdict:
    winner: str  # "a" | "b" | "tie"
    scores_a: dict[str, int]
    scores_b: dict[str, int]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "scores_a": self.scores_a,
            "scores_b": self.scores_b,
            "rationale": self.rationale,
        }


def _cache(cache_dir: Path | None = None) -> JsonCache:
    settings = get_settings()
    base = Path(cache_dir) if cache_dir else settings.cache_dir
    return JsonCache(base / "judge_cache.db")


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a Gemini response.

    The judge prompt asks for strict JSON, but Gemini sometimes wraps it in
    markdown fences or adds a preamble. This finds the first balanced object.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))
    # Fall back: greedily find the first `{...}` substring that parses.
    start = text.find("{")
    while start != -1:
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
        start = text.find("{", start + 1)
    raise ValueError(f"No JSON object found in judge response: {text[:300]}")


def _call_judge(transcript: str, summary_a: str, summary_b: str, model: str) -> dict:
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set; cannot call judge.")
    genai.configure(api_key=api_key)

    prompt = render_judge_prompt(transcript=transcript, summary_a=summary_a, summary_b=summary_b)
    response = genai.GenerativeModel(model).generate_content(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 512},
    )
    return _extract_json(response.text or "")


def judge(
    transcript: str,
    summary_a: str,
    summary_b: str,
    model: str | None = None,
    shuffle: bool = True,
    seed: int | None = None,
) -> JudgeVerdict:
    """Score two summaries against the rubric defined in `utils.prompts`.

    With `shuffle=True` (default), A and B are randomly swapped before the
    model sees them and the verdict is unswapped before return. This removes
    position bias that LLM-as-judge models tend to exhibit.
    """
    settings = get_settings()
    model = model or settings.judge_model

    rng = random.Random(seed if seed is not None else hash((summary_a, summary_b)) & 0xFFFFFFFF)
    swapped = bool(shuffle and rng.random() < 0.5)
    a_in, b_in = (summary_b, summary_a) if swapped else (summary_a, summary_b)

    cache = _cache()
    payload = {"model": model, "transcript": transcript, "a": a_in, "b": b_in}
    cached = cache.get(_NAMESPACE, payload)
    if cached is None:
        cached = _call_judge(transcript, a_in, b_in, model)
        cache.set(_NAMESPACE, payload, cached)

    raw_winner = cached.get("winner", "tie")
    raw_scores_a = cached.get("summary_a", {}) or cached.get("scores_a", {})
    raw_scores_b = cached.get("summary_b", {}) or cached.get("scores_b", {})

    if swapped:
        scores_a, scores_b = raw_scores_b, raw_scores_a
        if raw_winner == "a":
            winner = "b"
        elif raw_winner == "b":
            winner = "a"
        else:
            winner = raw_winner
    else:
        scores_a, scores_b = raw_scores_a, raw_scores_b
        winner = raw_winner

    return JudgeVerdict(
        winner=winner if winner in ("a", "b", "tie") else "tie",
        scores_a={k: int(v) for k, v in scores_a.items()},
        scores_b={k: int(v) for k, v in scores_b.items()},
        rationale=str(cached.get("rationale", "")),
    )


def winrate(verdicts: list[JudgeVerdict], who: str) -> float:
    """Fraction of verdicts where `who` (a or b) won. Ties count half."""
    if not verdicts:
        return 0.0
    wins = sum(1 for v in verdicts if v.winner == who)
    ties = sum(1 for v in verdicts if v.winner == "tie")
    return (wins + 0.5 * ties) / len(verdicts)
