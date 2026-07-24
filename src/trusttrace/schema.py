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

from pydantic import BaseModel, ConfigDict, Field


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


class Severity(str, Enum):
    """Severity level, derived deterministically from the primary category."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: Category -> Severity mapping (PRD §11.1).
_CATEGORY_SEVERITY: dict[Category, Severity] = {
    Category.NONE: Severity.NONE,
    Category.TOXIC: Severity.LOW,
    Category.INSULT: Severity.MEDIUM,
    Category.OBSCENE: Severity.MEDIUM,
    Category.IDENTITY_HATE: Severity.HIGH,
    Category.SEVERE_TOXIC: Severity.HIGH,
    Category.THREAT: Severity.HIGH,
}


def derive_severity(category: Category) -> Severity:
    """Map a primary ``Category`` to its locked ``Severity`` (PRD §11.1)."""
    return _CATEGORY_SEVERITY[category]


class ClassificationOutput(BaseModel):
    """The locked per-input structured output (PRD §11.1).

    ``extra='forbid'`` makes this a strict schema: unknown keys are rejected, which
    is exactly what the serving layer's ``schema.valid_rate`` metric will measure.
    ``confidence`` is a model-reported field in ``[0, 1]`` and is NOT ground-truth
    (see PRD §11.1) — it is monitored for calibration, not used for label accuracy.
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)

    @classmethod
    def ground_truth(cls, category: Category, confidence: float = 1.0) -> "ClassificationOutput":
        """Build a schema-consistent output from a category, filling derived severity.

        Used to construct training targets / eval references where severity must match
        the locked category→severity mapping.
        """
        return cls(category=category, severity=derive_severity(category), confidence=confidence)
