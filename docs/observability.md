# Observability — Tracing & SigNoz Runbook

Phase 2 wires OpenTelemetry tracing into the FastAPI service and exports it to SigNoz.
This is the foundation for the graded observability layer (custom metrics, quality-delta,
dashboards, and alerting come in Phase 5).

> **Scope note (CLAUDE.md).** Trace payloads are displayed in the SigNoz UI. Raw comment
> text is therefore **never** placed in span attributes or logs — only derived values like
> `input.char_len`. Keep this invariant when adding spans in later phases.

## How tracing is wired

- `src/trusttrace/observability/otel.py` — `configure_tracing()` installs a global
  `TracerProvider` with a `trusttrace` resource (`service.name`, `service.version`,
  `deployment.environment`) and a `BatchSpanProcessor` → OTLP gRPC exporter. Endpoint from
  `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4317`). Idempotent.
- `src/trusttrace/serving/app.py` — FastAPI app; `instrument_fastapi(app)` adds automatic
  server spans, and `POST /classify` opens a manual `trace.classify` span.

### Env toggles (see `.env.example`)

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | SigNoz OTLP gRPC endpoint (default `http://localhost:4317`) |
| `OTEL_ENV` | value of `deployment.environment` (default `local`) |
| `OTEL_CONSOLE_EXPORT=1` | also print spans to the console (local debugging) |
| `OTEL_DISABLE_OTLP=1` | skip OTLP export entirely (offline tests) |

## Local verification (no SigNoz required)

```bash
pip install -r requirements.txt
# well-formed span to console:
OTEL_DISABLE_OTLP=1 OTEL_CONSOLE_EXPORT=1 python -m trusttrace.observability.send_test_trace
# serving + tracing tests (offline, in-memory exporter):
pytest
```

## SigNoz install via Foundry (WSL2 + native Docker Engine)

SigNoz is installed with **Foundry** per the hackathon requirement. On Windows this must
run in **WSL2 with native Docker Engine** — *not* Docker Desktop (Foundry's documented
constraint). Reference: <https://signoz.io/docs/install/docker/>.

```bash
# inside WSL2 (native Docker Engine, >=4GB to Docker; ports 8080/4317/4318[/8000] free)
curl -fsSL https://signoz.io/foundry.sh | bash      # installs foundryctl
foundryctl gauge                                    # validate prerequisites
foundryctl forge -f casting.yaml                    # renders pours/, writes casting.yaml.lock
foundryctl cast  -f casting.yaml                    # start the stack
```

- SigNoz UI: `http://localhost:8080/` · OTLP: `4317` (gRPC) / `4318` (HTTP).
- **Commit `casting.yaml` + `casting.yaml.lock` exactly as Foundry produces them** (FR7 —
  judges re-run Foundry against these). `pours/` is generated output and is git-ignored.
- Optional SigNoz MCP server: add `spec.mcp.spec.enabled: true` to `casting.yaml`, re-cast
  (port `8000`), then create a service-account API key in the SigNoz UI and register it
  (`claude mcp add ... SIGNOZ-API-KEY: <key>`).

## Closing Phase 2 items 3 & 4 (once SigNoz is up)

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # or the WSL2 IP
python -m trusttrace.observability.send_test_trace          # item 3: trivial trace
uvicorn trusttrace.serving.app:app --port 8000              # or any free port
curl -XPOST localhost:8000/classify -H 'content-type: application/json' -d '{"text":"sample"}'
```

Then confirm in the SigNoz UI (Traces view) that service `trusttrace` shows the
`trusttrace.test_trace` span and a `trace.classify` span for the request.

### Ground-truth verification (don't trust "it ran" / "ready" logs)

A successful OTLP export or a collector `"Everything is ready"` log does NOT prove a trace
was stored. Verify against ClickHouse directly (inside the WSL2 host where SigNoz runs):

```bash
docker exec signoz-telemetrystore-clickhouse-0-0 clickhouse-client --param_svc=trusttrace -q \
  "SELECT serviceName, name, count() FROM signoz_traces.distributed_signoz_index_v3 \
   WHERE serviceName = {svc:String} AND timestamp > now() - INTERVAL 15 MINUTE \
   GROUP BY serviceName, name FORMAT PrettyCompact"
```
(Use `{svc:String}` bound params — ClickHouse treats `"double quotes"` as identifiers, not string literals.)

### Troubleshooting: OTLP connections reset / no traces land

If `4317`/`4318` reset every connection (from host, WSL2, *and* in-network) and ClickHouse
shows 0 rows, check the OPAMP handshake:

```bash
docker logs signoz-signoz-0   2>&1 | grep "cannot create agent without orgId"
docker logs signoz-ingester-1 2>&1 | grep "Server returned an error"
```

`cannot create agent without orgId` means **SigNoz first-run onboarding (create admin
account → organization) has not been completed**. Until it is, the server refuses to
register the otel-collector over OPAMP, so the collector never gets a pipeline config and
its OTLP receivers reject all traffic. Fix: complete onboarding at http://localhost:8080,
then re-send. (Observed and resolved 2026-07-24.)

## Custom canary metrics (Phase 5)

Recorded per request in `serving/classifier.py` and exported to SigNoz via OTLP (defined in
`observability/metrics.py`). All are Histograms; windowed aggregation happens in SigNoz.
Attributes are `path` + numeric values only — never raw text or per-request category values.

| Metric | Tags | Meaning / aggregation |
|---|---|---|
| `trusttrace.inference.latency_ms` | `path=base\|finetuned` | per-path generation latency (avg / p95) |
| `trusttrace.tokens.total` | `path=base\|finetuned` | per-path generated-token count (sum = token cost) |
| `trusttrace.schema.valid_rate` | `path=base\|finetuned` | 0/1 per request; **avg = schema-validity rate** |
| `trusttrace.quality_delta` | — | `finetuned_valid(0/1) − base_valid(0/1)` per request; **avg = validity-rate delta** |

**Quality-delta definition.** There is no ground truth at serving time and `confidence` is a
fixed 1.0 (see Known Limitations), so quality is measured by **schema-validity**:
`quality_delta = finetuned_schema_valid − base_schema_valid`. Healthy traffic sits near
**+0.86** (fine-tuned ~1.0 valid, base ~0.14). When the fine-tuned path degrades, the delta
falls — the alert signal.

### Dashboard + alert (import in the SigNoz UI)

Config lives in `artifacts/signoz/`:
- **`dashboard.json`** — 4 panels (latency, token cost, schema-validity rate, quality-delta),
  base vs fine-tuned. Import: SigNoz UI → Dashboards → New → *Import JSON*.
- **`alert.json`** — fires when **`avg(trusttrace.quality_delta)` over 5m < −0.10** (i.e.
  fine-tuned schema-validity >10pp below base; PRD §11.4). Import: Alerts → New → import / or
  recreate the threshold rule with these values.

> These are authored to SigNoz's import format but were not import-tested here (SigNoz was down
> during authoring); verify/tweak fields in the UI on import. The metric names + the
> quality-delta threshold above are the authoritative spec.

### Verifying metrics land (ClickHouse ground-truth)

```bash
docker exec signoz-telemetrystore-clickhouse-0-0 clickhouse-client -q \
  "SELECT DISTINCT metric_name FROM signoz_metrics.distributed_samples_v4 \
   WHERE metric_name LIKE 'trusttrace.%' FORMAT PrettyCompact"
```
(Table name may vary by SigNoz version — e.g. `samples_v4` / `distributed_samples_v4`; list
`SHOW TABLES FROM signoz_metrics` if needed.)

## Known limitations

- **Confidence is not calibrated (as of Phase 3).** The fine-tuned classifier emits a
  `confidence` field, but training targets fix `confidence = 1.0` (the ground-truth label is
  certain), so the model learns to report high confidence regardless of true uncertainty.
  Phase 3 therefore evaluates only **schema-validity + category accuracy/F1**, not confidence
  calibration. Real calibration (deriving confidence from the model's token logprobs rather
  than a trained constant) is a **Phase 5** enhancement. This is recorded here from the start
  so the README/demo narrative states it as a known limitation rather than reconstructing it
  later. Any confidence-based SigNoz alert (PRD §11.4) must account for this until Phase 5.

## Status

Phase 2 COMPLETE (2026-07-24). SigNoz installed via Foundry (WSL2), `casting.yaml`/lock
committed, and traces confirmed landing in SigNoz (ClickHouse ground-truth + UI) for both
`send_test_trace` and the instrumented FastAPI `/classify` stub. OTLP endpoint:
`http://localhost:4317` (gRPC). Next: Phase 3 (fine-tuning) / Phase 5 (real metrics + spans).
