"""Tests for the shared model-output parser (trusttrace.parse).

All inputs are benign JSON-ish strings — no toxic content (CLAUDE.md scope).
"""
from __future__ import annotations

import pytest

from trusttrace.parse import parse_classification
from trusttrace.schema import Category, Severity


def test_valid_output_is_schema_valid():
    r = parse_classification('{"category": "insult", "severity": "medium", "confidence": 0.9}')
    assert r.schema_valid is True
    assert r.output is not None
    assert r.output.category is Category.INSULT
    assert r.output.severity is Severity.MEDIUM
    assert r.category == "insult"


def test_json_embedded_in_prose_is_extracted():
    r = parse_classification('Sure! Here is the result: {"category": "threat", "severity": "high", "confidence": 1.0} done')
    assert r.schema_valid is True
    assert r.category == "threat"


def test_extra_key_fails_strict_but_category_still_extracted():
    # extra key -> ClassificationOutput(extra="forbid") rejects it => not schema_valid,
    # but the lenient category is still recovered.
    r = parse_classification('{"category": "toxic", "severity": "low", "confidence": 0.5, "reason": "x"}')
    assert r.schema_valid is False
    assert r.output is None
    assert r.category == "toxic"


def test_unknown_category_value():
    r = parse_classification('{"category": "spam", "severity": "low", "confidence": 0.5}')
    assert r.schema_valid is False   # 'spam' not a valid Category
    assert r.category is None         # not one of the 7 known categories


@pytest.mark.parametrize("text", ["", "no json here", "not json {oops", "{}"])
def test_garbage_is_not_schema_valid(text):
    r = parse_classification(text)
    assert r.schema_valid is False
    assert r.output is None
