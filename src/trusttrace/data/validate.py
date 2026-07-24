"""Loud data-quality validation for the processed splits.

Fails hard (raises ``DataValidationError`` / non-zero exit) on ANY violation so bad
data can never silently flow into training. Checks, per split:

- required columns present;
- no nulls in ``text`` / ``category`` / ``severity``; no empty ``text``;
- ``category`` / ``severity`` values within the locked enums;
- ``severity`` consistent with ``derive_severity(category)``;
- derived ``category`` consistent with the retained raw labels;
- raw label columns are strictly binary (0/1);
- every category represented in the split.

Scope: reports counts and category names only — never row text (CLAUDE.md).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from trusttrace.data.build_dataset import OUTPUT_COLUMNS, OUTPUT_DIR
from trusttrace.data.load import LABEL_COLUMNS
from trusttrace.schema import Category, Severity, derive_severity, map_labels_to_category

log = logging.getLogger(__name__)

_CATEGORY_VALUES = {c.value for c in Category}
_SEVERITY_VALUES = {s.value for s in Severity}


class DataValidationError(Exception):
    """Raised when a processed split fails one or more validation checks."""


def validate_split(df: pd.DataFrame, name: str = "dataset") -> bool:
    """Validate one split. Returns ``True`` on success; raises on any violation."""
    missing_cols = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"[{name}] missing columns: {missing_cols}")

    errors: list[str] = []

    for col in ("text", "category", "severity"):
        n_null = int(df[col].isna().sum())
        if n_null:
            errors.append(f"{n_null} null values in '{col}'")

    n_empty = int((df["text"].fillna("").str.strip().str.len() == 0).sum())
    if n_empty:
        errors.append(f"{n_empty} empty-text rows")

    bad_cat = sorted(set(df["category"].dropna()) - _CATEGORY_VALUES)
    if bad_cat:
        errors.append(f"invalid category values: {bad_cat}")
    bad_sev = sorted(set(df["severity"].dropna()) - _SEVERITY_VALUES)
    if bad_sev:
        errors.append(f"invalid severity values: {bad_sev}")

    if not bad_cat:
        inconsistent = sum(
            sev != derive_severity(Category(cat)).value
            for cat, sev in zip(df["category"], df["severity"])
            if pd.notna(cat) and pd.notna(sev)
        )
        if inconsistent:
            errors.append(f"{inconsistent} rows where severity != derive_severity(category)")

    if all(c in df.columns for c in LABEL_COLUMNS):
        for col in LABEL_COLUMNS:
            n_bad = int((~df[col].isin([0, 1])).sum())
            if n_bad:
                errors.append(f"{n_bad} non-binary values in label '{col}'")
        mismatched = sum(
            map_labels_to_category({c: row[c] for c in LABEL_COLUMNS}).value != row["category"]
            for _, row in df.iterrows()
        )
        if mismatched:
            errors.append(f"{mismatched} rows where category != map_labels_to_category(raw labels)")

    absent = sorted(_CATEGORY_VALUES - set(df["category"].dropna()))
    if absent:
        errors.append(f"categories absent from split: {absent}")

    if errors:
        raise DataValidationError(
            f"[{name}] {len(errors)} validation failure(s):\n  - " + "\n  - ".join(errors)
        )
    log.info("[%s] OK: %d rows passed all checks", name, len(df))
    return True


def validate_files(output_dir: Path = OUTPUT_DIR) -> bool:
    """Validate all three split files under ``output_dir``. Raises on any failure."""
    for name in ("train", "val", "test"):
        path = output_dir / f"{name}.parquet"
        if not path.exists():
            raise DataValidationError(f"missing split file: {path} (run build_dataset first)")
        validate_split(pd.read_parquet(path), name)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        validate_files()
    except DataValidationError as exc:
        log.error("DATA VALIDATION FAILED\n%s", exc)
        sys.exit(1)
    print("OK: all splits passed validation.")
