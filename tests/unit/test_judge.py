"""Tests for the judge that don't hit Gemini.

The actual API call is mocked. We verify:
  - JSON extraction from raw / fenced / preamble responses
  - Position-debiasing (shuffle) round-trips correctly
  - winrate aggregation
"""

from __future__ import annotations

import pytest

from earningslora.evaluation import llm_judge
from earningslora.evaluation.llm_judge import JudgeVerdict, _extract_json, judge, winrate


def test_extract_json_plain():
    raw = '{"winner": "a", "summary_a": {"faithfulness": 5}, "summary_b": {"faithfulness": 4}, "rationale": "ok"}'
    out = _extract_json(raw)
    assert out["winner"] == "a"


def test_extract_json_fenced():
    raw = 'Here is my verdict:\n```json\n{"winner": "tie", "summary_a": {}, "summary_b": {}, "rationale": "x"}\n```'
    out = _extract_json(raw)
    assert out["winner"] == "tie"


def test_extract_json_with_preamble():
    raw = "I am an AI. My answer:\n{\"winner\": \"b\", \"summary_a\": {}, \"summary_b\": {}, \"rationale\": \"\"}"
    out = _extract_json(raw)
    assert out["winner"] == "b"


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_judge_unswapped(monkeypatch, tmp_path):
    """When shuffle=False the verdict is returned as-is."""
    monkeypatch.setattr(
        llm_judge,
        "_call_judge",
        lambda *args, **kw: {
            "winner": "a",
            "summary_a": {"faithfulness": 5, "completeness": 5, "conciseness": 5, "style": 5},
            "summary_b": {"faithfulness": 3, "completeness": 3, "conciseness": 3, "style": 3},
            "rationale": "A wins.",
        },
    )
    monkeypatch.setattr(llm_judge, "_cache", lambda *a, **kw: _InMemoryCache())

    v = judge("transcript", "summary_a_text", "summary_b_text", shuffle=False)
    assert v.winner == "a"
    assert v.scores_a["faithfulness"] == 5
    assert v.scores_b["faithfulness"] == 3


def test_judge_shuffle_unswaps(monkeypatch):
    """With shuffle=True and a forced swap, an 'a' verdict from the model becomes 'b'."""
    monkeypatch.setattr(
        llm_judge,
        "_call_judge",
        lambda *args, **kw: {
            "winner": "a",  # model thinks A won — but A and B were swapped before the model saw them
            "summary_a": {"faithfulness": 5, "completeness": 5, "conciseness": 5, "style": 5},
            "summary_b": {"faithfulness": 3, "completeness": 3, "conciseness": 3, "style": 3},
            "rationale": "irrelevant",
        },
    )
    monkeypatch.setattr(llm_judge, "_cache", lambda *a, **kw: _InMemoryCache())

    # Force a swap by patching random.Random.random to always return < 0.5.
    monkeypatch.setattr(llm_judge.random.Random, "random", lambda self: 0.0)

    v = judge("transcript", "real_a", "real_b", shuffle=True)
    # Model said A won; A was actually real_b in the prompt, so real_b won → "b" in unswapped frame.
    assert v.winner == "b"
    # Scores also swap back.
    assert v.scores_a["faithfulness"] == 3
    assert v.scores_b["faithfulness"] == 5


def test_judge_uses_cache_on_second_call(monkeypatch):
    calls = {"n": 0}

    def fake_call(*args, **kw):
        calls["n"] += 1
        return {
            "winner": "a",
            "summary_a": {},
            "summary_b": {},
            "rationale": "",
        }

    cache = _InMemoryCache()
    monkeypatch.setattr(llm_judge, "_call_judge", fake_call)
    monkeypatch.setattr(llm_judge, "_cache", lambda *a, **kw: cache)

    judge("t", "a", "b", shuffle=False)
    judge("t", "a", "b", shuffle=False)
    assert calls["n"] == 1, "Second call should be served from cache."


def test_winrate_basics():
    verdicts = [
        JudgeVerdict("a", {}, {}, ""),
        JudgeVerdict("b", {}, {}, ""),
        JudgeVerdict("tie", {}, {}, ""),
        JudgeVerdict("a", {}, {}, ""),
    ]
    # 2 wins for a, 1 tie counted as 0.5 → 2.5/4 = 0.625
    assert winrate(verdicts, "a") == 0.625
    assert winrate(verdicts, "b") == 0.375


def test_winrate_empty():
    assert winrate([], "a") == 0.0


# --- helpers ---


class _InMemoryCache:
    def __init__(self):
        self._store = {}

    @staticmethod
    def _k(ns, payload):
        import json

        return ns + "::" + json.dumps(payload, sort_keys=True)

    def get(self, ns, payload):
        return self._store.get(self._k(ns, payload))

    def set(self, ns, payload, value):
        self._store[self._k(ns, payload)] = value
