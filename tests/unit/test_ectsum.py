"""Unit tests for the ECTSum loader.

Avoids network by mocking `datasets.load_dataset` and exercising the
column-normalisation, validation-filtering, hold-out carving, and
chat-formatting logic directly on synthetic in-memory `Dataset` objects.
"""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from earningslora.data import ectsum as ectsum_module
from earningslora.data.ectsum import (
    _filter_invalid,
    _normalize_columns,
    load_ectsum,
    make_holdout,
    to_chat_format,
)


def _fake_dict_with_alias_columns() -> DatasetDict:
    """Mimic an upstream that uses `text` + `summary` instead of `transcript` + `summary`."""
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": [f"transcript number {i}." for i in range(20)],
                    "summary": [f"- bullet for row {i}" for i in range(20)],
                    "extra_metadata": list(range(20)),  # should be dropped
                }
            ),
            "test": Dataset.from_dict(
                {
                    "text": [f"test transcript {i}." for i in range(10)],
                    "summary": [f"- test bullet {i}" for i in range(10)],
                    "extra_metadata": list(range(10)),
                }
            ),
        }
    )


def test_normalize_columns_renames_aliases():
    ds = Dataset.from_dict({"text": ["t"], "summary": ["s"], "noise": [1]})
    out = _normalize_columns(ds)
    assert set(out.column_names) == {"transcript", "summary"}
    assert out["transcript"][0] == "t"
    assert out["summary"][0] == "s"


def test_normalize_columns_already_canonical():
    ds = Dataset.from_dict({"transcript": ["t"], "summary": ["s"]})
    out = _normalize_columns(ds)
    assert set(out.column_names) == {"transcript", "summary"}


def test_normalize_columns_raises_on_unknown_schema():
    ds = Dataset.from_dict({"foo": ["x"], "bar": ["y"]})
    with pytest.raises(ValueError, match="Could not find transcript/summary"):
        _normalize_columns(ds)


def test_filter_invalid_drops_empty():
    ds = Dataset.from_dict(
        {
            "transcript": ["good transcript", "", "   ", "another good one"],
            "summary": ["good summary", "x", "y", "    "],
        }
    )
    out = _filter_invalid(ds)
    assert len(out) == 1
    assert out[0]["transcript"] == "good transcript"


def test_load_ectsum_uses_aliases_and_synthesises_validation(monkeypatch):
    fake = _fake_dict_with_alias_columns()
    monkeypatch.setattr(ectsum_module, "load_dataset", lambda *a, **kw: fake)
    out = load_ectsum(dataset_id="fake/ectsum")
    assert set(out.keys()) == {"train", "validation", "test"}
    assert set(out["train"].column_names) == {"transcript", "summary"}
    assert len(out["validation"]) >= 1


def test_load_ectsum_preserves_existing_validation(monkeypatch):
    fake = DatasetDict(
        {
            "train": Dataset.from_dict({"transcript": ["a"] * 10, "summary": ["b"] * 10}),
            "validation": Dataset.from_dict({"transcript": ["c"] * 4, "summary": ["d"] * 4}),
            "test": Dataset.from_dict({"transcript": ["e"] * 5, "summary": ["f"] * 5}),
        }
    )
    monkeypatch.setattr(ectsum_module, "load_dataset", lambda *a, **kw: fake)
    out = load_ectsum(dataset_id="fake/ectsum")
    assert len(out["validation"]) == 4


def test_make_holdout_deterministic():
    test = Dataset.from_dict(
        {"transcript": [f"t{i}" for i in range(100)], "summary": [f"s{i}" for i in range(100)]}
    )
    dd = DatasetDict({"train": test, "test": test})
    a = make_holdout(dd, seed=42, size=10)
    b = make_holdout(dd, seed=42, size=10)
    assert a["transcript"] == b["transcript"]
    assert len(a) == 10


def test_make_holdout_returns_full_test_when_smaller_than_size():
    small_test = Dataset.from_dict({"transcript": ["t1", "t2"], "summary": ["s1", "s2"]})
    dd = DatasetDict({"train": small_test, "test": small_test})
    out = make_holdout(dd, seed=0, size=50)
    assert len(out) == 2


def test_make_holdout_requires_test_split():
    dd = DatasetDict({"train": Dataset.from_dict({"transcript": ["t"], "summary": ["s"]})})
    with pytest.raises(ValueError, match="Expected a 'test' split"):
        make_holdout(dd, seed=0, size=10)


def test_to_chat_format_produces_messages():
    dd = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"transcript": ["Q3 revenue was $1B."], "summary": ["- Revenue: $1B"]}
            ),
        }
    )
    chat = to_chat_format(dd)
    assert chat["train"].column_names == ["messages"]
    msgs = chat["train"][0]["messages"]
    assert {m["role"] for m in msgs} == {"system", "user", "assistant"}
    assert msgs[2]["content"] == "- Revenue: $1B"
