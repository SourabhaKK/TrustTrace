"""Tokenized dataset + collator for QLoRA training.

Loads the Phase 1 parquet splits and renders each row to ``{input_ids, labels}`` via
``prompt.render_for_training`` (completion-only masking). A simple right-padding causal
collator pads ``input_ids`` with the pad token, ``labels`` with -100, and builds the mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import datasets
import pandas as pd
import torch

from trusttrace.training.prompt import render_for_training

DATA_DIR = Path("data")


def load_split(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load ``data/<name>.parquet`` (train/val/test)."""
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m trusttrace.data.build_dataset` first.")
    return pd.read_parquet(path)


def build_dataset(
    df: pd.DataFrame, tokenizer, max_length: int = 512, limit: int | None = None
) -> datasets.Dataset:
    """Tokenize a split into a HuggingFace ``Dataset`` of ``{input_ids, labels}``."""
    if limit is not None:
        df = df.head(limit)
    ds = datasets.Dataset.from_pandas(df[["text", "category"]], preserve_index=False)
    return ds.map(
        lambda row: render_for_training(tokenizer, row["text"], row["category"], max_length),
        remove_columns=ds.column_names,
        desc="tokenize",
    )


@dataclass
class CausalCollator:
    """Right-pad a batch of ``{input_ids, labels}`` and build the attention mask."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
