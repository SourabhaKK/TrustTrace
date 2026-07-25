"""Parse a model's raw text output into the structured classification schema.

Shared by the eval harness (training/evaluate.py) and the serving layer
(serving/classifier.py). Separates two notions:

- ``schema_valid`` — the output strictly parses into ``ClassificationOutput``
  (``extra="forbid"``). This is what the SigNoz schema-validity metric measures.
- ``category`` — a lenient best-effort predicted category (present even if the JSON has
  extra keys / wrong severity), used for accuracy/F1.

Scope (CLAUDE.md): callers use only the parsed structured fields — the raw model text is
never surfaced in API responses or logs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from trusttrace.schema import Category, ClassificationOutput

_CATEGORY_VALUES = {c.value for c in Category}
_JSON_RE = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing one raw model output."""

    schema_valid: bool
    output: ClassificationOutput | None
    category: str | None


def parse_classification(text: str) -> ParseResult:
    """Extract the first JSON object from ``text`` and parse it.

    Returns a ``ParseResult``; ``output`` is a validated ``ClassificationOutput`` iff the
    JSON strictly conforms to the schema, else ``None``. ``category`` is the lenient
    predicted category (or ``None`` if absent/unknown).
    """
    match = _JSON_RE.search(text or "")
    if match is None:
        return ParseResult(False, None, None)
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return ParseResult(False, None, None)

    output: ClassificationOutput | None
    try:
        output = ClassificationOutput.model_validate(obj)
    except Exception:  # noqa: BLE001 — strict schema; any failure => not schema-valid
        output = None

    cat = obj.get("category") if isinstance(obj, dict) else None
    category = cat if cat in _CATEGORY_VALUES else None
    return ParseResult(schema_valid=output is not None, output=output, category=category)
