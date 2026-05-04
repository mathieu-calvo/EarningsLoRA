"""Single + batched generation with deterministic seed and stop tokens.

Greedy decoding (`do_sample=False`) is the default — eval needs reproducibility,
not creativity. The chat template is applied here so callers don't have to know
about model-specific prompt formatting.
"""

from __future__ import annotations

import logging

from earningslora.data.format import build_inference_record

logger = logging.getLogger(__name__)

DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": False,
    "repetition_penalty": 1.05,
}


def _build_prompt(transcript: str, tokenizer) -> str:
    """Build the chat-templated prompt string (system + user, no assistant turn)."""
    record = build_inference_record(transcript)
    return tokenizer.apply_chat_template(
        record.messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_summary(model, tokenizer, transcript: str, **gen_kwargs) -> str:
    """Generate a bullet summary for a single transcript."""
    import torch

    prompt = _build_prompt(transcript, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **kwargs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_batch(
    model,
    tokenizer,
    transcripts: list[str],
    batch_size: int = 4,
    **gen_kwargs,
) -> list[str]:
    """Batched generation with left-padding. Returns one summary per input order."""
    import torch

    kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
    out: list[str] = []
    for i in range(0, len(transcripts), batch_size):
        chunk = transcripts[i : i + batch_size]
        prompts = [_build_prompt(t, tokenizer) for t in chunk]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **kwargs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = int(inputs["attention_mask"][j].sum().item())
            # Account for left-padding: the actual prompt ends at position
            # input_ids.shape[1]; new tokens start there.
            new_token_start = inputs["input_ids"].shape[1]
            generated = output[new_token_start:]
            out.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
            del input_len  # silence linter; kept for future debugging

    return out
