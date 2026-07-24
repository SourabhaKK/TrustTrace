"""Load the Jigsaw Toxic Comment Classification dataset via HuggingFace ``datasets``.

Scope boundary (CLAUDE.md): detection-only. This module NEVER prints or logs raw
comment text — only aggregate counts. The 6 raw binary label columns are retained
downstream for eval; the mapping to the locked category/severity schema happens in
``trusttrace.schema`` and ``trusttrace.data.subsample``.

The canonical competition dataset (``jigsaw-toxic-comment-classification-challenge``)
requires a manual Kaggle download. To keep the pipeline reproducible we try a list of
self-contained HuggingFace mirrors first and fall back to a local CSV if none load.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

#: The six Jigsaw binary label columns, in a stable order.
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

#: Column names that may hold the comment text across different mirrors.
_TEXT_ALIASES = ["comment_text", "text", "comment", "content"]

#: Candidate HF dataset ids exposing the Jigsaw train split with the 6 label columns.
#: Tried in order; the first that yields the expected schema wins.
_CANDIDATE_HF_IDS = [
    "Arsive/toxicity_classification_jigsaw",
    "thesofakillers/jigsaw-toxic-comment-classification-challenge",
    "tcapelle/jigsaw-toxic-comment",
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a frame with columns ``text`` + the 6 label ints, or ``None`` if the
    source frame lacks the expected Jigsaw schema."""
    lower = {c.lower(): c for c in df.columns}
    text_col = next((lower[a] for a in _TEXT_ALIASES if a in lower), None)
    if text_col is None or not all(lbl in lower for lbl in LABEL_COLUMNS):
        return None
    out = pd.DataFrame({"text": df[text_col].astype("string")})
    for lbl in LABEL_COLUMNS:
        # Jigsaw test mirrors use -1 for unlabeled rows; treat only 1 as positive.
        out[lbl] = (pd.to_numeric(df[lower[lbl]], errors="coerce").fillna(0) == 1).astype(int)
    out = out.dropna(subset=["text"])
    out = out[out["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return out


def load_jigsaw(split: str = "train") -> pd.DataFrame:
    """Load Jigsaw from the first working HF mirror.

    Returns a DataFrame with columns ``text`` + ``LABEL_COLUMNS`` (int 0/1).
    Raises ``RuntimeError`` with fallback instructions if no mirror loads.
    """
    from datasets import load_dataset

    last_err: Exception | None = None
    for hf_id in _CANDIDATE_HF_IDS:
        try:
            ds = load_dataset(hf_id, split=split)
            df = _normalize(ds.to_pandas())
            if df is not None and len(df) > 0:
                log.info("Loaded Jigsaw from '%s': %d rows", hf_id, len(df))
                return df
            log.warning("'%s' loaded but schema did not match; trying next.", hf_id)
        except Exception as exc:  # noqa: BLE001 — try the next mirror
            last_err = exc
            log.warning("Failed to load '%s': %s", hf_id, exc)

    raise RuntimeError(
        "Could not load Jigsaw from any known HF mirror. Fallback: download "
        "'jigsaw-toxic-comment-classification-challenge' train.csv from Kaggle "
        "(e.g. via `kagglehub`), place it at data/raw/train.csv, and call "
        f"load_jigsaw_csv(). Last error: {last_err!r}"
    )


def load_jigsaw_csv(path: str = "data/raw/train.csv") -> pd.DataFrame:
    """Fallback loader for a locally-downloaded Kaggle CSV."""
    df = _normalize(pd.read_csv(path))
    if df is None:
        raise ValueError(
            f"{path} is missing expected Jigsaw columns {LABEL_COLUMNS} + a text column."
        )
    log.info("Loaded Jigsaw from local CSV '%s': %d rows", path, len(df))
    return df


def _label_summary(df: pd.DataFrame) -> dict[str, int]:
    """Aggregate, text-free summary safe to print/log."""
    summary = {lbl: int(df[lbl].sum()) for lbl in LABEL_COLUMNS}
    any_label = (df[LABEL_COLUMNS].sum(axis=1) > 0).sum()
    summary["_total_rows"] = len(df)
    summary["_any_label"] = int(any_label)
    summary["_clean_rows"] = int(len(df) - any_label)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = load_jigsaw()
    print("Columns:", list(frame.columns))
    print("Aggregate label summary (no row content shown):")
    for key, val in _label_summary(frame).items():
        print(f"  {key}: {val}")
