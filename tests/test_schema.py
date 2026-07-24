"""Unit tests for the locked taxonomy mapping and structured-output schema (PRD §11.1).

Scope: these tests use binary LABEL VECTORS and benign placeholder text only — no
toxic content is authored anywhere (CLAUDE.md hard scope boundary).
"""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from trusttrace.data.load import LABEL_COLUMNS
from trusttrace.data.subsample import add_schema_columns
from trusttrace.schema import (
    Category,
    ClassificationOutput,
    Severity,
    derive_severity,
    map_labels_to_category,
)


def _labels(**active: int) -> dict[str, int]:
    """Build a full 6-label dict, defaulting every unnamed label to 0."""
    return {col: int(active.get(col, 0)) for col in LABEL_COLUMNS}


# --- map_labels_to_category: priority resolution ---------------------------------

@pytest.mark.parametrize(
    "labels, expected",
    [
        (_labels(), Category.NONE),
        (_labels(toxic=1), Category.TOXIC),
        (_labels(insult=1), Category.INSULT),
        (_labels(obscene=1), Category.OBSCENE),
        (_labels(identity_hate=1), Category.IDENTITY_HATE),
        (_labels(severe_toxic=1), Category.SEVERE_TOXIC),
        (_labels(threat=1), Category.THREAT),
        # priority: threat beats everything
        (_labels(toxic=1, obscene=1, insult=1, threat=1), Category.THREAT),
        # identity_hate beats severe_toxic/obscene/insult/toxic
        (_labels(toxic=1, severe_toxic=1, identity_hate=1), Category.IDENTITY_HATE),
        # severe_toxic beats obscene/insult/toxic
        (_labels(toxic=1, obscene=1, insult=1, severe_toxic=1), Category.SEVERE_TOXIC),
        # obscene beats insult/toxic
        (_labels(toxic=1, insult=1, obscene=1), Category.OBSCENE),
        # insult beats toxic
        (_labels(toxic=1, insult=1), Category.INSULT),
    ],
)
def test_map_labels_to_category(labels, expected):
    assert map_labels_to_category(labels) is expected


# --- derive_severity: category -> severity ---------------------------------------

@pytest.mark.parametrize(
    "category, expected",
    [
        (Category.NONE, Severity.NONE),
        (Category.TOXIC, Severity.LOW),
        (Category.INSULT, Severity.MEDIUM),
        (Category.OBSCENE, Severity.MEDIUM),
        (Category.IDENTITY_HATE, Severity.HIGH),
        (Category.SEVERE_TOXIC, Severity.HIGH),
        (Category.THREAT, Severity.HIGH),
    ],
)
def test_derive_severity(category, expected):
    assert derive_severity(category) is expected


def test_every_category_has_a_severity():
    for category in Category:
        assert isinstance(derive_severity(category), Severity)


# --- ClassificationOutput: strict schema -----------------------------------------

def test_valid_output_parses():
    out = ClassificationOutput(category="threat", severity="high", confidence=0.9)
    assert out.category is Category.THREAT
    assert out.severity is Severity.HIGH


def test_ground_truth_fills_derived_severity():
    out = ClassificationOutput.ground_truth(Category.OBSCENE)
    assert out.severity is Severity.MEDIUM
    assert out.confidence == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"category": "threat", "severity": "high", "confidence": 1.5},   # out of range
        {"category": "threat", "severity": "high", "confidence": -0.1},  # out of range
        {"category": "not_a_category", "severity": "high", "confidence": 0.5},  # bad enum
        {"category": "threat", "severity": "extreme", "confidence": 0.5},  # bad enum
        {"category": "threat", "severity": "high"},  # missing confidence
        {"category": "threat", "severity": "high", "confidence": 0.5, "extra": 1},  # extra key
    ],
)
def test_invalid_output_rejected(kwargs):
    with pytest.raises(ValidationError):
        ClassificationOutput(**kwargs)


# --- integration: every mapped row builds a valid ClassificationOutput ------------

def test_add_schema_columns_and_validate_all_rows():
    # Synthetic, benign-text frame exercising all seven categories.
    rows = [
        {"text": "sample text 0", **_labels()},                         # none
        {"text": "sample text 1", **_labels(toxic=1)},                  # toxic
        {"text": "sample text 2", **_labels(toxic=1, insult=1)},        # insult
        {"text": "sample text 3", **_labels(obscene=1)},                # obscene
        {"text": "sample text 4", **_labels(identity_hate=1)},          # identity_hate
        {"text": "sample text 5", **_labels(severe_toxic=1)},           # severe_toxic
        {"text": "sample text 6", **_labels(threat=1, toxic=1)},        # threat
    ]
    df = add_schema_columns(pd.DataFrame(rows))

    assert set(df["category"]) == {c.value for c in Category}
    for _, row in df.iterrows():
        # severity must be the derived one, and the row must satisfy the strict schema.
        out = ClassificationOutput.ground_truth(Category(row["category"]))
        assert out.severity.value == row["severity"]
