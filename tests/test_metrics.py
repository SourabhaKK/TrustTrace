"""Tests for the custom canary metrics (trusttrace.observability.metrics).

Uses an InMemoryMetricReader with an injected meter (no global provider / OTLP), so we can
assert exact recorded values + path attributes offline.
"""
from __future__ import annotations

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from trusttrace.observability import metrics


def _fresh_reader() -> InMemoryMetricReader:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.init_metrics(provider.get_meter("test"))  # inject test meter
    return reader


def _by_name(reader: InMemoryMetricReader) -> dict:
    out = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out[metric.name] = metric
    return out


def _sum_by_path(metric) -> dict[str, float]:
    return {dp.attributes.get("path"): dp.sum for dp in metric.data.data_points}


def test_all_four_metrics_emitted_with_path_tags():
    reader = _fresh_reader()
    metrics.record_path("finetuned", latency_ms=12.5, tokens=8, schema_valid=True)
    metrics.record_path("base", latency_ms=40.0, tokens=25, schema_valid=False)
    metrics.record_quality_delta(1.0)

    by_name = _by_name(reader)
    assert {
        "trusttrace.inference.latency_ms",
        "trusttrace.tokens.total",
        "trusttrace.schema.valid_rate",
        "trusttrace.quality_delta",
    } <= set(by_name)

    # per-path latency + tokens recorded under the right path tag
    assert _sum_by_path(by_name["trusttrace.inference.latency_ms"]) == {"finetuned": 12.5, "base": 40.0}
    assert _sum_by_path(by_name["trusttrace.tokens.total"]) == {"finetuned": 8, "base": 25}

    # schema validity: finetuned 1.0 (valid), base 0.0 (invalid)
    assert _sum_by_path(by_name["trusttrace.schema.valid_rate"]) == {"finetuned": 1.0, "base": 0.0}

    # quality_delta has no path tag; the single recorded value is 1.0
    delta = by_name["trusttrace.quality_delta"]
    assert sum(dp.sum for dp in delta.data.data_points) == 1.0


def test_validity_rate_averages_over_requests():
    reader = _fresh_reader()
    # 3 finetuned requests: 2 valid, 1 invalid -> sum 2, count 3 -> avg 2/3
    for valid in (True, True, False):
        metrics.record_path("finetuned", latency_ms=1.0, tokens=1, schema_valid=valid)
    metric = _by_name(reader)["trusttrace.schema.valid_rate"]
    dp = next(d for d in metric.data.data_points if d.attributes.get("path") == "finetuned")
    assert dp.sum == 2.0 and dp.count == 3  # SigNoz avg = 2/3 validity rate
