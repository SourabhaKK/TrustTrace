"""OpenTelemetry tracing setup, exporting spans to SigNoz over OTLP.

The OTLP endpoint is read from ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default
``http://localhost:4317`` — the gRPC endpoint a Foundry-installed SigNoz exposes), so no
endpoint is hard-coded. Toggles via env:

- ``OTEL_CONSOLE_EXPORT=1`` — also print spans to the console (local verification).
- ``OTEL_DISABLE_OTLP=1``   — skip the OTLP exporter entirely (offline tests).

Scope (CLAUDE.md): callers must never put raw input text into span attributes — trace
payloads are displayed in the SigNoz UI. Use lengths/hashes, not content.
"""
from __future__ import annotations

import logging
import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from trusttrace import __version__

log = logging.getLogger(__name__)

SERVICE_NAME = "trusttrace"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4317"

_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def configure_tracing(service_name: str = SERVICE_NAME) -> TracerProvider:
    """Install a global ``TracerProvider`` exporting to SigNoz via OTLP.

    Idempotent: the first call wins; later calls return the same provider so importing
    the app more than once (e.g. under uvicorn reload / tests) does not stack exporters.
    """
    global _provider
    if _provider is not None:
        return _provider

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": __version__,
            "deployment.environment": os.getenv("OTEL_ENV", "local"),
        }
    )
    provider = TracerProvider(resource=resource)

    if not _truthy(os.getenv("OTEL_DISABLE_OTLP")):
        insecure = not endpoint.lower().startswith("https")
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
        )
        log.info("OTLP span export -> %s (insecure=%s)", endpoint, insecure)

    if _truthy(os.getenv("OTEL_CONSOLE_EXPORT")):
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def get_tracer(name: str = SERVICE_NAME):
    """Return a tracer, configuring tracing on first use."""
    configure_tracing()
    return trace.get_tracer(name)


def instrument_fastapi(app) -> None:
    """Attach OpenTelemetry ASGI instrumentation to a FastAPI app."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, tracer_provider=configure_tracing())


def configure_metrics(service_name: str = SERVICE_NAME) -> MeterProvider:
    """Install a global ``MeterProvider`` exporting metrics to SigNoz via OTLP.

    Idempotent (mirrors ``configure_tracing``). Respects the same env toggles:
    ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_DISABLE_OTLP``, ``OTEL_CONSOLE_EXPORT``.
    """
    global _meter_provider
    if _meter_provider is not None:
        return _meter_provider

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": __version__,
            "deployment.environment": os.getenv("OTEL_ENV", "local"),
        }
    )
    readers = []
    if not _truthy(os.getenv("OTEL_DISABLE_OTLP")):
        insecure = not endpoint.lower().startswith("https")
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=endpoint, insecure=insecure),
                export_interval_millis=5000,
            )
        )
        log.info("OTLP metric export -> %s (insecure=%s)", endpoint, insecure)
    if _truthy(os.getenv("OTEL_CONSOLE_EXPORT")):
        readers.append(
            PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=2000)
        )

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    _meter_provider = provider
    return provider


def get_meter(name: str = SERVICE_NAME):
    """Return a meter, configuring metrics on first use."""
    configure_metrics()
    return metrics.get_meter(name)
