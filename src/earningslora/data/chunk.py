"""Long-transcript handling for the 4096-token training context.

Earnings calls vary from ~3k to ~15k tokens. When the transcript exceeds the
configured `max_seq_len`, we truncate by section priority:

  1. Prepared remarks (kept in full where possible)
  2. Q&A (truncated last)
  3. Boilerplate / forward-looking statements (dropped first)

The token-counting function is injected so both an approximate (chars/4)
default and a real `tokenizer.encode` can be used. Tests use the default.

Implemented in Weekend 1.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Q&A section markers — varies across transcripts.
_QNA_HEADERS = re.compile(
    r"^\s*(?:question[- ]?and[- ]?answer|q\s*&\s*a|q\s*and\s*a)"
    r"\s*(?:session|portion)?\s*[:.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# "Operator:" line — first one usually marks the end of the boilerplate.
_OPERATOR_LINE = re.compile(r"^\s*Operator\s*[:.]", re.MULTILINE)

# Forward-looking / safe-harbor language — boilerplate fingerprint.
_FORWARD_LOOKING = re.compile(
    r"forward[- ]looking statements?|safe\s*harbo[u]?r|"
    r"private securities litigation reform act",
    re.IGNORECASE,
)

_BOILERPLATE_HEAD_CHARS = 2500


def approx_token_count(text: str) -> int:
    """Coarse approximation: ~4 chars per token. Fast, tokenizer-free, good enough for budgeting."""
    return max(1, len(text) // 4)


def split_sections(transcript: str) -> dict[str, str]:
    """Detect prepared-remarks vs Q&A vs boilerplate sections.

    Returns a dict with keys 'prepared', 'qna', 'boilerplate' — missing sections
    are returned as empty strings, never absent keys.
    """
    if not transcript:
        return {"prepared": "", "qna": "", "boilerplate": ""}

    qna_match = _QNA_HEADERS.search(transcript)
    if qna_match:
        prepared = transcript[: qna_match.start()].strip()
        qna = transcript[qna_match.end() :].strip()
    else:
        prepared = transcript.strip()
        qna = ""

    boilerplate = ""
    head = prepared[:_BOILERPLATE_HEAD_CHARS]
    if _FORWARD_LOOKING.search(head):
        op_match = _OPERATOR_LINE.search(prepared)
        if op_match and op_match.start() < _BOILERPLATE_HEAD_CHARS:
            boilerplate = prepared[: op_match.start()].strip()
            prepared = prepared[op_match.start() :].strip()

    return {"prepared": prepared, "qna": qna, "boilerplate": boilerplate}


def fit_to_budget(
    transcript: str,
    max_tokens: int,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Truncate `transcript` to fit `max_tokens` by section priority.

    Drops boilerplate first, keeps prepared remarks in full where possible,
    truncates Q&A from the tail. If even the prepared remarks don't fit, falls
    back to a proportional head-truncation of the prepared remarks.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    counter = count_tokens or approx_token_count

    if counter(transcript) <= max_tokens:
        return transcript

    sections = split_sections(transcript)
    prepared = sections["prepared"]
    qna = sections["qna"]

    prepared_tokens = counter(prepared)
    if prepared_tokens >= max_tokens:
        ratio = max_tokens / max(prepared_tokens, 1)
        cut = max(1, int(len(prepared) * ratio))
        return prepared[:cut].rstrip()

    remaining = max_tokens - prepared_tokens
    qna_tokens = counter(qna)
    if qna_tokens <= remaining:
        return f"{prepared}\n\n{qna}".strip() if qna else prepared

    if not qna:
        return prepared

    ratio = remaining / max(qna_tokens, 1)
    cut = max(0, int(len(qna) * ratio))
    truncated_qna = qna[:cut].rstrip() if cut > 0 else ""
    if truncated_qna:
        return f"{prepared}\n\n{truncated_qna}".strip()
    return prepared
