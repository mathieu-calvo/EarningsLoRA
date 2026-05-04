"""Format ECTSum (transcript, summary) pairs into chat-template SFT records.

The records are model-agnostic: they hold a `messages` list of OpenAI-style turns
(system / user / assistant). The training loop later applies the base model's
tokenizer chat template to render them into a single string.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from earningslora.utils.prompts import SYSTEM_PROMPT, render_user_prompt


@dataclass(frozen=True)
class ChatRecord:
    """A single SFT example as model-agnostic chat turns."""

    messages: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"messages": list(self.messages)}


def build_record(transcript: str, summary: str) -> ChatRecord:
    """Build a chat-template SFT record from a (transcript, summary) pair.

    The system prompt and user-prompt template live in `utils/prompts.py` so the
    task definition has a single source of truth across SFT, baseline, and judge.
    """
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be non-empty")
    if not summary or not summary.strip():
        raise ValueError("summary must be non-empty")

    return ChatRecord(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(transcript.strip())},
            {"role": "assistant", "content": summary.strip()},
        ]
    )


def build_inference_record(transcript: str) -> ChatRecord:
    """Build a chat record for inference (no assistant turn)."""
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be non-empty")

    return ChatRecord(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(transcript.strip())},
        ]
    )
