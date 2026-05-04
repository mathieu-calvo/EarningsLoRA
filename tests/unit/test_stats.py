from earningslora.data.stats import (
    summary_length_stats,
    transcript_length_stats,
)


def test_transcript_length_stats_empty():
    assert transcript_length_stats([]) == {"n": 0}


def test_transcript_length_stats_basic():
    rows = [{"transcript": "a" * 400}, {"transcript": "b" * 800}, {"transcript": "c" * 1200}]
    stats = transcript_length_stats(rows)
    assert stats["n"] == 3
    assert stats["max"] >= stats["p99"] >= stats["p90"] >= stats["p50"]
    assert stats["mean"] > 0


def test_transcript_length_stats_with_word_counter():
    def word_count(text):
        return max(1, len(text.split()))

    rows = [{"transcript": " ".join(["w"] * 100)}, {"transcript": " ".join(["w"] * 200)}]
    stats = transcript_length_stats(rows, count_tokens=word_count)
    assert stats["max"] == 200
    assert stats["mean"] == 150


def test_summary_length_stats_counts_bullets():
    rows = [
        {"summary": "- one\n- two\n- three"},
        {"summary": "* alpha\n* beta"},
    ]
    stats = summary_length_stats(rows)
    assert stats["n"] == 2
    assert stats["bullets_max"] == 3
    assert stats["bullets_mean"] == 2.5


def test_summary_length_stats_no_bullets():
    rows = [{"summary": "Just a paragraph with no bullets."}]
    stats = summary_length_stats(rows)
    assert stats["bullets_max"] == 0
    assert stats["bullets_mean"] == 0


def test_summary_length_stats_empty():
    assert summary_length_stats([]) == {"n": 0}


def test_percentiles_monotonic():
    rows = [{"transcript": "x" * (i * 100)} for i in range(1, 21)]
    stats = transcript_length_stats(rows)
    assert stats["p50"] <= stats["p90"] <= stats["p99"] <= stats["max"]
