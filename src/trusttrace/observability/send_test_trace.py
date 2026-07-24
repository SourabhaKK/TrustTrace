"""Emit a single trivial span to verify traces reach SigNoz (TASKS.md Phase 2, item 3).

Run this AFTER SigNoz is up (Foundry) and ``OTEL_EXPORTER_OTLP_ENDPOINT`` points at it,
then look for service ``trusttrace`` / span ``trusttrace.test_trace`` in the SigNoz UI.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
        python -m trusttrace.observability.send_test_trace

Add ``OTEL_CONSOLE_EXPORT=1`` to also print the span locally (no SigNoz required).
"""
from __future__ import annotations

import logging
import os

from trusttrace.observability.otel import DEFAULT_OTLP_ENDPOINT, configure_tracing, get_tracer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    provider = configure_tracing()
    tracer = get_tracer("trusttrace.test")

    with tracer.start_as_current_span("trusttrace.test_trace") as span:
        span.set_attribute("test.kind", "connectivity")
        trace_id = format(span.get_span_context().trace_id, "032x")

    provider.force_flush()
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    print(f"Emitted span 'trusttrace.test_trace' (trace_id={trace_id}) -> {endpoint}")
    print("If SigNoz is up, find it under service 'trusttrace' in the Traces view.")


if __name__ == "__main__":
    main()
