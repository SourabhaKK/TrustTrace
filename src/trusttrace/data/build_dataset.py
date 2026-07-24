"""Build the processed train/val/test splits from the Jigsaw subsample.

Pipeline: load -> class-balanced subsample -> add schema columns -> stratified
70/15/15 split on category -> write ``data/{train,val,test}.parquet``.

The written parquet files carry ``text``, ``category``, ``severity``, and the 6 raw
Jigsaw labels (retained for eval in Phase 3). ``data/`` is git-ignored, so raw comment
text is never committed. Only aggregate counts are printed/logged (CLAUDE.md scope).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from trusttrace.data.load import LABEL_COLUMNS, load_jigsaw
from trusttrace.data.subsample import (
    add_schema_columns,
    category_distribution,
    stratified_subsample,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data")
SEED = 42
#: Columns persisted per split: schema fields + retained raw labels for eval.
OUTPUT_COLUMNS = ["text", "category", "severity", *LABEL_COLUMNS]


def build_splits(df: pd.DataFrame | None = None, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Return stratified 70/15/15 train/val/test splits of the processed subsample."""
    if df is None:
        df = load_jigsaw()
    sub = add_schema_columns(stratified_subsample(df, seed=seed))[OUTPUT_COLUMNS]

    train, temp = train_test_split(
        sub, test_size=0.30, random_state=seed, stratify=sub["category"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["category"]
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path = OUTPUT_DIR) -> None:
    """Write each split to ``output_dir/<name>.parquet``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        path = output_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path, len(part))


def _log_distribution(name: str, df: pd.DataFrame) -> None:
    log.info("%s: %d rows", name, len(df))
    for cat, count in category_distribution(df).items():
        log.info("  %-14s %d", cat, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    splits = build_splits()
    for name, part in splits.items():
        _log_distribution(name, part)
        missing = [c for c, v in category_distribution(part).items() if v == 0]
        assert not missing, f"{name} split is missing categories: {missing}"
    write_splits(splits)
    total = sum(len(p) for p in splits.values())
    print(f"OK: wrote train/val/test totaling {total} rows; all 7 categories present in every split.")
