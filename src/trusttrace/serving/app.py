"""FastAPI serving layer — Phase 2 stub.

A single stub ``/classify`` endpoint returns a FIXED dummy ``ClassificationOutput`` — no
model is loaded yet (models arrive in Phase 3/4). The point of this phase is to prove the
request path is traced end-to-end into SigNoz, not to classify anything.

Scope (CLAUDE.md): detection-only; the request text is NEVER written to span attributes
or logs (trace payloads are displayed in the SigNoz UI). Only its length is recorded.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from trusttrace import __version__
from trusttrace.observability.otel import get_tracer, instrument_fastapi
from trusttrace.schema import Category, ClassificationOutput, Severity

app = FastAPI(title="TrustTrace", version=__version__)
instrument_fastapi(app)
tracer = get_tracer("trusttrace.serving")


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassificationOutput)
def classify(req: ClassifyRequest) -> ClassificationOutput:
    with tracer.start_as_current_span("trace.classify") as span:
        # Length only — never the raw text (scope boundary).
        span.set_attribute("input.char_len", len(req.text))
        span.set_attribute("model.stage", "stub")
        # STUB: no model yet. Fixed, benign output so the path is traceable.
        return ClassificationOutput(
            category=Category.NONE, severity=Severity.NONE, confidence=0.0
        )
