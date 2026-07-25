"""Tests for the dual-model serving endpoint (Phase 4).

A fake classifier is injected via FastAPI ``dependency_overrides`` so the API contract,
Pydantic enforcement, and tracing are tested WITHOUT loading a model / GPU. Benign text only.
"""
from __future__ import annotations

import os

os.environ["OTEL_DISABLE_OTLP"] = "1"  # offline; span export disabled

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from trusttrace.observability import otel  # noqa: E402
from trusttrace.parse import ParseResult  # noqa: E402
from trusttrace.schema import Category, ClassificationOutput  # noqa: E402
from trusttrace.serving.app import app, classifier_dependency  # noqa: E402
from trusttrace.serving.classifier import DualResult  # noqa: E402

_memory = InMemorySpanExporter()
otel.configure_tracing().add_span_processor(SimpleSpanProcessor(_memory))
client = TestClient(app)


class _FakeClassifier:
    def __init__(self, finetuned: ParseResult, base: ParseResult):
        self._ft, self._base = finetuned, base

    def classify(self, text: str) -> DualResult:
        return DualResult(finetuned=self._ft, base=self._base)


def _valid(category: Category) -> ParseResult:
    out = ClassificationOutput.ground_truth(category)
    return ParseResult(schema_valid=True, output=out, category=category.value)


def _invalid() -> ParseResult:
    return ParseResult(schema_valid=False, output=None, category=None)


def _use(finetuned: ParseResult, base: ParseResult) -> None:
    app.dependency_overrides[classifier_dependency] = lambda: _FakeClassifier(finetuned, base)


def setup_function(_) -> None:
    _memory.clear()


def teardown_function(_) -> None:
    app.dependency_overrides.clear()


def test_healthz_ok():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_returns_both_paths():
    _use(_valid(Category.INSULT), _invalid())
    resp = client.post("/classify", json={"text": "sample benign message"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["primary"] == "finetuned"
    # fine-tuned path: valid, schema-conformant output
    assert body["finetuned"]["schema_valid"] is True
    assert body["finetuned"]["output"] == {"category": "insult", "severity": "medium", "confidence": 1.0}
    # base path: invalid -> null output (never a raw string)
    assert body["base"]["schema_valid"] is False
    assert body["base"]["output"] is None


def test_both_paths_valid():
    _use(_valid(Category.THREAT), _valid(Category.NONE))
    body = client.post("/classify", json={"text": "sample"}).json()
    assert body["finetuned"]["output"]["category"] == "threat"
    assert body["base"]["output"]["category"] == "none"


def test_classify_rejects_empty_text():
    resp = client.post("/classify", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_trace_classify_span_records_length_not_raw_text():
    _use(_valid(Category.NONE), _invalid())
    marker = "uniquemarker0987654321"
    client.post("/classify", json={"text": marker})
    spans = {s.name: s for s in _memory.get_finished_spans()}
    assert "trace.classify" in spans
    attrs = dict(spans["trace.classify"].attributes)
    assert attrs.get("input.char_len") == len(marker)
    for span in _memory.get_finished_spans():
        for value in (span.attributes or {}).values():
            assert marker not in str(value)  # raw text never leaks into any span
