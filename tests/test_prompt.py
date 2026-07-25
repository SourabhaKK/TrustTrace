"""Tests for training prompt/target construction and completion-only masking.

The pure functions (build_target/build_messages) run offline. The masking test needs the
real Qwen tokenizer; it downloads tokenizer files (~a few MB, not the model) and SKIPS
cleanly if unavailable offline. Fixtures use benign text only (CLAUDE.md scope).
"""
from __future__ import annotations

import json

import pytest

from trusttrace.schema import Category, ClassificationOutput
from trusttrace.training.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    build_target,
    render_for_training,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.parametrize(
    "category, expected_severity",
    [
        (Category.NONE, "none"),
        (Category.TOXIC, "low"),
        (Category.INSULT, "medium"),
        (Category.OBSCENE, "medium"),
        (Category.IDENTITY_HATE, "high"),
        (Category.SEVERE_TOXIC, "high"),
        (Category.THREAT, "high"),
    ],
)
def test_build_target_is_schema_valid_json(category, expected_severity):
    target = build_target(category)
    parsed = json.loads(target)
    assert parsed == {"category": category.value, "severity": expected_severity, "confidence": 1.0}
    # round-trips through the strict schema
    ClassificationOutput.model_validate(parsed)


def test_build_messages_structure():
    msgs = build_messages("hello")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"] == "hello"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 — offline / no access
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")


def test_render_masks_prompt_supervises_only_target(tokenizer):
    out = render_for_training(tokenizer, "a benign sample message", Category.INSULT, max_length=512)
    input_ids, labels = out["input_ids"], out["labels"]

    assert len(input_ids) == len(labels)
    assert any(l == -100 for l in labels), "prompt tokens must be masked"
    supervised = [i for i, l in zip(input_ids, labels) if l != -100]
    assert supervised, "target tokens must be supervised"

    # the supervised region decodes to the JSON target
    decoded = tokenizer.decode(supervised, skip_special_tokens=True)
    assert build_target(Category.INSULT) in decoded

    # supervised labels equal the corresponding input_ids (teacher forcing on the target)
    for i, l in zip(input_ids, labels):
        assert l == -100 or l == i


def test_render_truncates_text_not_target(tokenizer):
    long_text = "word " * 4000  # far exceeds max_length
    out = render_for_training(tokenizer, long_text, Category.THREAT, max_length=256)
    assert len(out["input_ids"]) <= 256
    supervised = [i for i, l in zip(out["input_ids"], out["labels"]) if l != -100]
    decoded = tokenizer.decode(supervised, skip_special_tokens=True)
    # the target survived truncation
    assert build_target(Category.THREAT) in decoded
