"""Tests for the Phase 2 serving stub and its tracing.

Uses an in-memory span exporter to prove requests are traced without needing a live
SigNoz. OTLP export is disabled so the suite is fully offline. Scope: benign text only.
"""
from __future__ import annotations

import os

# Must be set before importing the app (configure_tracing runs at import time).
os.environ["OTEL_DISABLE_OTLP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from trusttrace.observability import otel  # noqa: E402
from trusttrace.schema import Category, ClassificationOutput, Severity  # noqa: E402
from trusttrace.serving.app import app  # noqa: E402

_memory = InMemorySpanExporter()
otel.configure_tracing().add_span_processor(SimpleSpanProcessor(_memory))
client = TestClient(app)


def setup_function(_) -> None:
    _memory.clear()


def test_healthz_ok():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_returns_schema_valid_stub():
    resp = client.post("/classify", json={"text": "sample text"})
    assert resp.status_code == 200
    out = ClassificationOutput.model_validate(resp.json())
    assert out.category is Category.NONE
    assert out.severity is Severity.NONE
    assert out.confidence == 0.0


def test_classify_rejects_empty_text():
    resp = client.post("/classify", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_classify_records_trace_classify_span():
    client.post("/classify", json={"text": "sample text here"})
    names = [s.name for s in _memory.get_finished_spans()]
    assert "trace.classify" in names


def test_span_records_length_not_raw_text():
    marker = "uniquemarker1234567890"
    client.post("/classify", json={"text": marker})
    spans = {s.name: s for s in _memory.get_finished_spans()}
    assert "trace.classify" in spans
    attrs = dict(spans["trace.classify"].attributes)
    assert attrs.get("input.char_len") == len(marker)
    # The raw text must not leak into ANY span attribute (all spans in the trace).
    for span in _memory.get_finished_spans():
        for value in (span.attributes or {}).values():
            assert marker not in str(value)
