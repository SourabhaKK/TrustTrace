"""Custom canary metrics (PRD §11.4), recorded per request and exported to SigNoz.

Four Histograms (windowed aggregations done in SigNoz):
- ``trusttrace.inference.latency_ms{path}`` — per-path generation latency
- ``trusttrace.tokens.total{path}``         — per-path generated-token count
- ``trusttrace.schema.valid_rate{path}``    — per request 1.0/0.0; avg = validity rate
- ``trusttrace.quality_delta``              — finetuned_valid - base_valid; avg = validity-rate
                                              delta (finetuned - base). Alert fires when it drops.

Scope (CLAUDE.md): attributes are ``path`` + numeric values only — never raw text or per-request
category values.
"""
from __future__ import annotations

_instruments: dict[str, object] = {}


def init_metrics(meter=None) -> None:
    """Create the four Histogram instruments. ``meter`` defaults to the global OTLP meter;
    tests pass an in-memory-reader meter to inspect recordings."""
    if meter is None:
        from trusttrace.observability.otel import get_meter

        meter = get_meter("trusttrace.serving")
    _instruments["latency"] = meter.create_histogram(
        "trusttrace.inference.latency_ms", unit="ms", description="per-path generation latency"
    )
    _instruments["tokens"] = meter.create_histogram(
        "trusttrace.tokens.total", unit="1", description="per-path generated token count"
    )
    _instruments["valid"] = meter.create_histogram(
        "trusttrace.schema.valid_rate", unit="1", description="per-request schema validity (0/1)"
    )
    _instruments["delta"] = meter.create_histogram(
        "trusttrace.quality_delta", unit="1", description="finetuned_valid - base_valid per request"
    )


def _get(name: str):
    if name not in _instruments:
        init_metrics()
    return _instruments[name]


def record_path(path: str, latency_ms: float, tokens: int, schema_valid: bool) -> None:
    """Record the per-path metrics for one path of one request."""
    attrs = {"path": path}
    _get("latency").record(latency_ms, attrs)
    _get("tokens").record(tokens, attrs)
    _get("valid").record(1.0 if schema_valid else 0.0, attrs)


def record_quality_delta(delta: float) -> None:
    """Record the base-vs-finetuned quality delta for one request."""
    _get("delta").record(delta)
