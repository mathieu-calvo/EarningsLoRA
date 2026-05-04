import pytest

from earningslora.data.format import build_inference_record, build_record
from earningslora.utils.prompts import SYSTEM_PROMPT


def test_build_record_has_three_turns():
    record = build_record("Q3 revenue was $1.2B.", "- Revenue: $1.2B")
    assert len(record.messages) == 3
    assert record.messages[0]["role"] == "system"
    assert record.messages[1]["role"] == "user"
    assert record.messages[2]["role"] == "assistant"


def test_build_record_uses_canonical_system_prompt():
    record = build_record("transcript", "summary")
    assert record.messages[0]["content"] == SYSTEM_PROMPT


def test_build_record_strips_whitespace():
    record = build_record("  transcript  ", "  summary  ")
    assert "transcript" in record.messages[1]["content"]
    assert record.messages[2]["content"] == "summary"


def test_build_inference_record_has_two_turns():
    record = build_inference_record("transcript")
    assert len(record.messages) == 2
    assert {m["role"] for m in record.messages} == {"system", "user"}


def test_build_record_rejects_empty():
    with pytest.raises(ValueError):
        build_record("", "summary")
    with pytest.raises(ValueError):
        build_record("transcript", "   ")


def test_to_dict_round_trip():
    record = build_record("transcript", "summary")
    d = record.to_dict()
    assert d["messages"][0]["role"] == "system"
    assert len(d["messages"]) == 3
