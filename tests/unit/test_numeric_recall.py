from earningslora.evaluation.numeric_recall import numeric_recall


def test_perfect_recall_simple_numbers():
    transcript = "Q3 revenue was 1.2 billion dollars and EPS came in at 2.45."
    summary = "- Revenue 1.2 billion\n- EPS 2.45"
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0
    assert result.matched == 2
    assert result.total == 2


def test_currency_unit_normalisation():
    """'$1.2 billion' in transcript should match '$1.2B' in summary."""
    transcript = "Revenue grew to $1.2 billion this quarter."
    summary = "- Revenue: $1.2B"
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0


def test_million_abbreviation_match():
    transcript = "Operating income was 250 million."
    summary = "- Operating income: $250M"
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0


def test_comma_separators_match():
    transcript = "Headcount reached 1,234,567 globally."
    summary = "- Headcount: 1234567"
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0


def test_hallucinated_number_drops_recall():
    transcript = "Revenue was $1.2 billion."
    summary = "- Revenue $1.2B\n- Margin 47%"  # 47% not in transcript
    result = numeric_recall(transcript, summary)
    assert result.recall == 0.5
    assert result.matched == 1
    assert result.total == 2
    assert 47.0 in result.unmatched_examples


def test_summary_with_no_numbers_is_vacuously_grounded():
    transcript = "Revenue was $1.2 billion."
    summary = "- Strong quarter overall.\n- Outlook positive."
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0
    assert result.total == 0


def test_empty_transcript_means_any_number_unmatched():
    result = numeric_recall("", "- Revenue: $1B")
    assert result.recall == 0.0


def test_percentage_match():
    transcript = "Gross margin expanded to 47% from 45% a year ago."
    summary = "- Gross margin: 47%"
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0


def test_relative_tolerance_accepts_minor_rounding():
    transcript = "Revenue was 1.2345 billion."
    summary = "- Revenue: 1.2346 billion"  # 0.008% off — within REL_TOL
    result = numeric_recall(transcript, summary)
    assert result.recall == 1.0


def test_result_str_format():
    result = numeric_recall("$1B in revenue.", "- $1B")
    assert "numeric_recall=" in str(result)
    assert "1/1" in str(result)
