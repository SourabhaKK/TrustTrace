"""FastAPI serving layer — dual-model canary (Phase 4).

Every ``/classify`` request is evaluated by BOTH the base model and the fine-tuned model
(one shared 4-bit model with the LoRA adapter toggled — see serving/classifier.py). Both
outputs are strictly Pydantic-validated and returned. Maps to FR1.

Scope (CLAUDE.md): detection-only. The response contains only parsed structured fields
(never raw model text); span attributes are length-/flag-only. If a path's raw output does
not conform to the schema, that path returns ``schema_valid=false`` + ``output=null``.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from trusttrace import __version__
from trusttrace.observability.otel import configure_metrics, get_tracer, instrument_fastapi
from trusttrace.schema import ClassificationOutput
from trusttrace.serving.classifier import DualClassifier, get_classifier

app = FastAPI(title="TrustTrace", version=__version__)
instrument_fastapi(app)
configure_metrics()  # install the OTLP MeterProvider for the custom canary metrics
tracer = get_tracer("trusttrace.serving")


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)


class PathResult(BaseModel):
    """One model path's result: strict schema validity + the validated output (or null)."""

    schema_valid: bool
    output: ClassificationOutput | None = None


class DualClassifyResponse(BaseModel):
    primary: str  # which path is the canary primary
    finetuned: PathResult
    base: PathResult


def classifier_dependency() -> DualClassifier:
    """FastAPI dependency — the lazily-loaded singleton. Overridden in tests."""
    return get_classifier()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=DualClassifyResponse)
def classify(
    req: ClassifyRequest,
    classifier: DualClassifier = Depends(classifier_dependency),
) -> DualClassifyResponse:
    with tracer.start_as_current_span("trace.classify") as span:
        span.set_attribute("input.char_len", len(req.text))  # length only — never the text
        result = classifier.classify(req.text)
        span.set_attribute("finetuned.schema_valid", result.finetuned.schema_valid)
        span.set_attribute("base.schema_valid", result.base.schema_valid)
        return DualClassifyResponse(
            primary="finetuned",
            finetuned=PathResult(
                schema_valid=result.finetuned.schema_valid, output=result.finetuned.output
            ),
            base=PathResult(schema_valid=result.base.schema_valid, output=result.base.output),
        )
