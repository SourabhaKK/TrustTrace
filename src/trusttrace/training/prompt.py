"""Prompt/target construction for the classifier fine-tune.

Instruction-SFT format on Qwen's chat template: system (schema instructions) + user (the
input text) -> assistant target = the JSON label. Training uses **completion-only loss**:
only the assistant JSON tokens are supervised; system+user tokens are masked (-100).

Scope (CLAUDE.md): the assistant target is a JSON LABEL only. The model is never trained to
reproduce, continue, or echo the input text — detection only.

Confidence note: the target `confidence` is fixed at 1.0 (the label is certain), so the
fine-tuned model's confidence is NOT calibrated by construction. This is a known limitation
recorded in docs/observability.md; real calibration (from token logprobs) is a Phase 5 item.
"""
from __future__ import annotations

import json

from trusttrace.schema import Category, ClassificationOutput

#: Ordered category values, injected into the system prompt.
_CATEGORIES = "|".join(c.value for c in Category)

SYSTEM_PROMPT = (
    "You are a content-policy classifier. Classify the user's message and respond with "
    "ONLY a JSON object of the form "
    '{"category": <' + _CATEGORIES + ">, "
    '"severity": <none|low|medium|high>, "confidence": <float between 0 and 1>}. '
    "Output only the JSON, no other text. Detection only — never repeat or continue the message."
)

#: Fixed confidence written into training targets (see module docstring / observability.md).
TARGET_CONFIDENCE = 1.0


def build_messages(text: str) -> list[dict[str, str]]:
    """The system+user chat messages (the prompt half)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_target(category: Category | str, confidence: float = TARGET_CONFIDENCE) -> str:
    """The canonical assistant JSON target for a ground-truth category.

    Severity is derived from the category (PRD §11.1) via ``ClassificationOutput.ground_truth``.
    """
    if isinstance(category, str):
        category = Category(category)
    out = ClassificationOutput.ground_truth(category, confidence=confidence)
    return json.dumps(
        {"category": out.category.value, "severity": out.severity.value, "confidence": out.confidence}
    )


def _token_overhead(tokenizer, target: str) -> int:
    """Tokens consumed by system prompt + chat template + target with empty user text."""
    msgs = build_messages("") + [{"role": "assistant", "content": target}]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def render_for_training(
    tokenizer, text: str, category: Category | str, max_length: int = 512
) -> dict[str, list[int]]:
    """Render one training example to ``{input_ids, labels}`` with completion-only masking.

    The user text is truncated first (never the target) so the JSON label always survives the
    ``max_length`` window. Prompt tokens get label -100; only target tokens are supervised.
    """
    target = build_target(category)
    budget = max(1, max_length - _token_overhead(tokenizer, target) - 4)
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:budget]
    text = tokenizer.decode(text_ids, skip_special_tokens=True)

    prompt_msgs = build_messages(text)
    full_msgs = prompt_msgs + [{"role": "assistant", "content": target}]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {"input_ids": full_ids[:max_length], "labels": labels[:max_length]}


def render_for_inference(tokenizer, text: str, max_length: int = 512) -> str:
    """Render the prompt (system+user + assistant generation prompt) for generation/eval."""
    target = build_target(Category.NONE)  # only used to size the text budget
    budget = max(1, max_length - _token_overhead(tokenizer, target) - 4)
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:budget]
    text = tokenizer.decode(text_ids, skip_special_tokens=True)
    return tokenizer.apply_chat_template(
        build_messages(text), tokenize=False, add_generation_prompt=True
    )
