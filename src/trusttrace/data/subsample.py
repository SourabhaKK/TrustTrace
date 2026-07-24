"""Deterministic, class-balanced stratified subsample of the Jigsaw dataset.

Raw Jigsaw is dominated by clean rows and has very rare threat/identity_hate classes.
For fast iteration and to give the fine-tune usable signal on rare categories, we take
a roughly class-balanced subsample keyed on the derived primary ``category`` (a per-
category cap), rather than a proportional sample that would starve the rare classes.

Scope: aggregate counts only — no raw comment text is printed/logged (CLAUDE.md).
"""
from __future__ import annotations

import logging

import pandas as pd

from trusttrace.data.load import LABEL_COLUMNS, load_jigsaw
from trusttrace.schema import Category, derive_severity, map_labels_to_category

log = logging.getLogger(__name__)

DEFAULT_SEED = 42
#: Max rows kept per derived category. ~350 x 7 categories -> ~2.4k rows, within the
#: PRD's ~1,500-3,000 target and deliberately balanced across categories.
DEFAULT_PER_CATEGORY_CAP = 350


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``category`` column (str values of ``Category``)."""
    df = df.copy()
    df["category"] = df[LABEL_COLUMNS].apply(
        lambda row: map_labels_to_category(row.to_dict()).value, axis=1
    )
    return df


def add_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add both ``category`` and derived ``severity`` columns (str enum values)."""
    df = add_category(df)
    df["severity"] = df["category"].apply(
        lambda c: derive_severity(Category(c)).value
    )
    return df


def category_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Ordered category -> count map (all 7 categories present, zeros included)."""
    counts = df["category"].value_counts().to_dict()
    return {c.value: int(counts.get(c.value, 0)) for c in Category}


def stratified_subsample(
    df: pd.DataFrame,
    per_category_cap: int = DEFAULT_PER_CATEGORY_CAP,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Class-balanced subsample: up to ``per_category_cap`` rows per derived category.

    Deterministic given ``seed``. Shuffled so downstream splits aren't category-ordered.
    """
    if "category" not in df.columns:
        df = add_category(df)
    parts = [
        group.sample(n=min(len(group), per_category_cap), random_state=seed)
        for _, group in df.groupby("category", sort=False)
    ]
    sub = (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)  # shuffle
        .reset_index(drop=True)
    )
    return sub


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = load_jigsaw()
    sub = stratified_subsample(raw)
    dist = category_distribution(sub)
    print(f"Subsampled rows: {len(sub)} (seed={DEFAULT_SEED}, cap={DEFAULT_PER_CATEGORY_CAP})")
    print("Category distribution (no row content shown):")
    for cat, count in dist.items():
        print(f"  {cat:14s}: {count}")
    assert all(v > 0 for v in dist.values()), "every category must be represented"
    assert 1500 <= len(sub) <= 3000, "subsample size must be within PRD target range"
    print("OK: all 7 categories present; size within target range.")
