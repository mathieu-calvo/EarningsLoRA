from earningslora.data.chunk import (
    approx_token_count,
    fit_to_budget,
    split_sections,
)


def test_approx_token_count_nonzero():
    assert approx_token_count("hello") >= 1
    assert approx_token_count("a" * 400) > approx_token_count("a" * 100)


def test_split_sections_finds_qna_header():
    transcript = (
        "Good morning everyone. Operator: please go ahead.\n"
        "[prepared remarks]\n\n"
        "Question-and-Answer Session\n\n"
        "Analyst: thanks for taking my question."
    )
    sections = split_sections(transcript)
    assert "prepared remarks" in sections["prepared"]
    assert "Analyst" in sections["qna"]


def test_split_sections_handles_qa_alternative_marker():
    transcript = (
        "Prepared remarks here.\n\n"
        "Q&A\n\n"
        "Analyst question."
    )
    sections = split_sections(transcript)
    assert "Prepared remarks" in sections["prepared"]
    assert "Analyst question" in sections["qna"]


def test_split_sections_no_qna_returns_all_prepared():
    transcript = "Just prepared remarks, no Q&A in this transcript."
    sections = split_sections(transcript)
    assert sections["prepared"] == transcript
    assert sections["qna"] == ""


def test_split_sections_extracts_boilerplate():
    transcript = (
        "Forward-looking statements: this call may contain forward-looking "
        "statements within the meaning of the Private Securities Litigation "
        "Reform Act. Please refer to the safe harbor disclaimer.\n\n"
        "Operator: welcome.\n\n"
        "[prepared remarks]"
    )
    sections = split_sections(transcript)
    assert "forward-looking" in sections["boilerplate"].lower()
    assert "[prepared remarks]" in sections["prepared"]


def test_split_sections_empty_transcript():
    sections = split_sections("")
    assert sections == {"prepared": "", "qna": "", "boilerplate": ""}


def test_fit_to_budget_short_transcript_unchanged():
    transcript = "Q3 revenue was 1.2 billion."
    assert fit_to_budget(transcript, max_tokens=1000) == transcript


def test_fit_to_budget_drops_qna_when_prepared_fits():
    prepared = "P" * 800       # ~200 tokens
    qna_marker = "\nQ&A\n"
    qna = "Q" * 800            # ~200 tokens
    transcript = prepared + qna_marker + qna
    out = fit_to_budget(transcript, max_tokens=300)
    # 200 (prepared) + 200 (qna) = 400 > 300, so qna should be partially trimmed.
    # The prepared section must remain in full (or near-full).
    assert out.startswith("P" * 100)  # prepared text preserved
    # Output stays within budget plus some slack from approximation rounding.
    assert approx_token_count(out) <= 320


def test_fit_to_budget_truncates_prepared_when_too_long():
    prepared = "P" * 4000  # ~1000 tokens
    out = fit_to_budget(prepared, max_tokens=200)
    assert approx_token_count(out) <= 220
    assert out.startswith("P")  # head retained


def test_fit_to_budget_rejects_zero_budget():
    import pytest

    with pytest.raises(ValueError):
        fit_to_budget("anything", max_tokens=0)


def test_fit_to_budget_idempotent_under_budget():
    transcript = "Short transcript that easily fits the budget."
    once = fit_to_budget(transcript, max_tokens=1000)
    twice = fit_to_budget(once, max_tokens=1000)
    assert once == twice


def test_fit_to_budget_with_real_token_counter():
    """Inject a deterministic word-based counter to confirm the API works for real tokenizers."""

    def word_count(text: str) -> int:
        return max(1, len(text.split()))

    transcript = " ".join(["word"] * 500)
    out = fit_to_budget(transcript, max_tokens=100, count_tokens=word_count)
    assert word_count(out) <= 110
