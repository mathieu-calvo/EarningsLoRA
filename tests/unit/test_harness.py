from earningslora.evaluation.harness import load_predictions, run_holdout


def test_run_holdout_writes_jsonl(tmp_path):
    holdout = [
        {"transcript": "Q3 revenue 1B.", "summary": "- Rev: 1B"},
        {"transcript": "Q4 revenue 2B.", "summary": "- Rev: 2B"},
    ]

    def stub_summarise(transcript: str) -> str:
        return f"summary of {len(transcript)} chars"

    out_path = run_holdout("base", stub_summarise, holdout, tmp_path)
    assert out_path.exists()
    rows = load_predictions(out_path)
    assert len(rows) == 2
    assert rows[0]["transcript"] == "Q3 revenue 1B."
    assert rows[0]["reference"] == "- Rev: 1B"
    assert rows[0]["prediction"].startswith("summary of")
    assert rows[0]["error"] is None
    assert rows[0]["latency_ms"] >= 0


def test_run_holdout_captures_errors(tmp_path):
    def crashes(transcript):
        raise RuntimeError("boom")

    holdout = [{"transcript": "t", "summary": "s"}]
    out_path = run_holdout("frontier", crashes, holdout, tmp_path)
    rows = load_predictions(out_path)
    assert len(rows) == 1
    assert rows[0]["prediction"] == ""
    assert "boom" in rows[0]["error"]


def test_run_holdout_handles_empty(tmp_path):
    out_path = run_holdout("ft", lambda t: "x", [], tmp_path)
    assert out_path.exists()
    assert load_predictions(out_path) == []
