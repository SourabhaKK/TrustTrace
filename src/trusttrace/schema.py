"""Locked output taxonomy and structured-output schema (PRD §11.1).

This module holds the single source of truth for how the 6 Jigsaw binary labels
collapse into the ``{category, severity, confidence}`` schema used across the whole
system (data layer, training targets, and the serving layer's Pydantic validation).

Phase 1 uses the ``Category`` enum + ``map_labels_to_category`` for stratification;
``Severity`` derivation and the ``ClassificationOutput`` Pydantic model are added in C3.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class Category(str, Enum):
    """Primary policy category — the single highest-priority active Jigsaw label."""

    NONE = "none"
    TOXIC = "toxic"
    INSULT = "insult"
    OBSCENE = "obscene"
    IDENTITY_HATE = "identity_hate"
    SEVERE_TOXIC = "severe_toxic"
    THREAT = "threat"


#: Category resolution priority, highest first (PRD §11.1):
#: threat > identity_hate > severe_toxic > obscene > insult > toxic.
CATEGORY_PRIORITY: tuple[Category, ...] = (
    Category.THREAT,
    Category.IDENTITY_HATE,
    Category.SEVERE_TOXIC,
    Category.OBSCENE,
    Category.INSULT,
    Category.TOXIC,
)


def map_labels_to_category(labels: Mapping[str, int]) -> Category:
    """Collapse the 6 Jigsaw binary labels to the single primary ``Category``.

    Returns the highest-priority label that is active (==1); ``Category.NONE`` if
    all six are zero. ``labels`` is any mapping keyed by the Jigsaw column names
    (``toxic``, ``severe_toxic``, ``obscene``, ``threat``, ``insult``, ``identity_hate``).
    """
    for category in CATEGORY_PRIORITY:
        if int(labels.get(category.value, 0)) == 1:
            return category
    return Category.NONE
