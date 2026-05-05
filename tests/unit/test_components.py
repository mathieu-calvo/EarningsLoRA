"""Tests for the framework-agnostic demo helpers in `earningslora.demo`.

UI frameworks are not imported here — the helpers themselves are framework-free.
"""

from __future__ import annotations

import pytest

from earningslora.demo import GenerationResult, time_call, truncate_for_display


def test_time_call_captures_text_and_recall():
    transcript = "Q3 revenue was 1.2 billion."
    result = time_call(lambda t: "Revenue: 1.2 billion.", transcript)
    assert result.text == "Revenue: 1.2 billion."
    assert result.error is None
    assert result.numeric_recall == pytest.approx(1.0)
    assert result.latency_ms >= 0


def test_time_call_captures_error():
    def crashes(_: str) -> str:
        raise RuntimeError("nope")

    result = time_call(crashes, "anything")
    assert result.text == ""
    assert "nope" in result.error
    assert result.numeric_recall == 0.0


def test_time_call_handles_unfaithful_summary():
    transcript = "Q3 revenue was 1.2 billion."
    result = time_call(lambda t: "Revenue: $99 billion.", transcript)  # made up
    assert result.numeric_recall < 1.0


def test_truncate_for_display_short_text_unchanged():
    assert truncate_for_display("short") == "short"


def test_truncate_for_display_long_text_marked():
    text = "x" * 5000
    out = truncate_for_display(text, max_chars=100)
    assert out.startswith("x" * 100)
    assert "truncated for display" in out


def test_generation_result_is_immutable():
    r = GenerationResult(text="x", latency_ms=10.0, numeric_recall=0.5)
    with pytest.raises((AttributeError, Exception)):
        r.text = "y"  # frozen dataclass
