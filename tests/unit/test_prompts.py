from earningslora.utils.prompts import (
    JUDGE_RUBRIC,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    render_judge_prompt,
    render_user_prompt,
)


def test_user_prompt_renders_transcript():
    rendered = render_user_prompt("Q3 revenue was $1.2B.")
    assert "Q3 revenue was $1.2B." in rendered
    assert "{transcript}" not in rendered


def test_judge_prompt_renders_all_three():
    rendered = render_judge_prompt(
        transcript="transcript text",
        summary_a="bullet a",
        summary_b="bullet b",
    )
    assert "transcript text" in rendered
    assert "bullet a" in rendered
    assert "bullet b" in rendered
    assert "{summary_a}" not in rendered


def test_system_prompt_mentions_faithfulness_constraints():
    """Guard rail: the system prompt must instruct the model not to invent figures."""
    lower = SYSTEM_PROMPT.lower()
    assert "do not invent" in lower or "faithful" in lower


def test_user_prompt_template_has_placeholder():
    assert "{transcript}" in USER_PROMPT_TEMPLATE


def test_judge_rubric_returns_strict_json_contract():
    """The rubric must ask for strict JSON output so we can parse responses."""
    assert "strict JSON" in JUDGE_RUBRIC or "Return strict JSON" in JUDGE_RUBRIC
    assert '"winner"' in JUDGE_RUBRIC
    assert '"faithfulness"' in JUDGE_RUBRIC
