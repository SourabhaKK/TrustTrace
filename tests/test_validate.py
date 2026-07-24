"""Tests for the loud data validator.

Scope: synthetic frames built from binary label vectors + benign placeholder text.
No toxic content is authored (CLAUDE.md hard scope boundary).
"""
from __future__ import annotations

import pandas as pd
import pytest

from trusttrace.data.load import LABEL_COLUMNS
from trusttrace.data.subsample import add_schema_columns
from trusttrace.data.validate import DataValidationError, validate_split
from trusttrace.schema import Category


def _labels(**active: int) -> dict[str, int]:
    return {col: int(active.get(col, 0)) for col in LABEL_COLUMNS}


def _valid_frame() -> pd.DataFrame:
    """Two benign rows per category (all 7 present), schema columns derived correctly."""
    single = {
        Category.NONE: _labels(),
        Category.TOXIC: _labels(toxic=1),
        Category.INSULT: _labels(insult=1),
        Category.OBSCENE: _labels(obscene=1),
        Category.IDENTITY_HATE: _labels(identity_hate=1),
        Category.SEVERE_TOXIC: _labels(severe_toxic=1),
        Category.THREAT: _labels(threat=1),
    }
    rows = []
    for i, labels in enumerate(single.values()):
        for j in range(2):
            rows.append({"text": f"sample text {i}-{j}", **labels})
    return add_schema_columns(pd.DataFrame(rows))


def test_valid_frame_passes():
    assert validate_split(_valid_frame(), "valid") is True


def test_null_text_fails():
    df = _valid_frame()
    df.loc[0, "text"] = None
    with pytest.raises(DataValidationError, match="null values in 'text'"):
        validate_split(df, "null_text")


def test_empty_text_fails():
    df = _valid_frame()
    df.loc[0, "text"] = "   "
    with pytest.raises(DataValidationError, match="empty-text"):
        validate_split(df, "empty_text")


def test_inconsistent_severity_fails():
    df = _valid_frame()
    df.loc[df.index[0], "severity"] = "low"  # break category<->severity consistency
    with pytest.raises(DataValidationError, match="severity != derive_severity"):
        validate_split(df, "bad_severity")


def test_invalid_category_value_fails():
    df = _valid_frame()
    df.loc[df.index[0], "category"] = "not_a_category"
    with pytest.raises(DataValidationError, match="invalid category values"):
        validate_split(df, "bad_category")


def test_category_label_mismatch_fails():
    df = _valid_frame()
    # flip a raw label so derived category no longer matches the stored category
    df.loc[df.index[0], "threat"] = 1
    with pytest.raises(DataValidationError, match="map_labels_to_category"):
        validate_split(df, "mismatch")


def test_missing_category_coverage_fails():
    df = _valid_frame()
    df = df[df["category"] != Category.THREAT.value]  # drop an entire category
    with pytest.raises(DataValidationError, match="categories absent"):
        validate_split(df, "missing_cat")


def test_missing_column_fails():
    df = _valid_frame().drop(columns=["severity"])
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_split(df, "missing_col")
