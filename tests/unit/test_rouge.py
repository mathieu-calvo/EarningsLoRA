import pytest

from earningslora.evaluation.rouge import rouge_scores


def test_perfect_overlap_scores_high():
    pred = ["Revenue grew 12% to one billion dollars."]
    ref = ["Revenue grew 12% to one billion dollars."]
    s = rouge_scores(pred, ref)
    assert s["rouge1"] > 0.9
    assert s["rougeL"] > 0.9
    assert s["n"] == 1


def test_disjoint_scores_zero():
    s = rouge_scores(["alpha bravo charlie"], ["xray yankee zulu"])
    assert s["rouge1"] == 0.0


def test_empty_inputs():
    s = rouge_scores([], [])
    assert s == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0}


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        rouge_scores(["a"], ["a", "b"])


def test_partial_overlap_in_between():
    s = rouge_scores(
        ["Revenue grew this quarter."],
        ["Revenue increased this quarter."],
    )
    assert 0.4 < s["rouge1"] < 1.0
