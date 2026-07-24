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

## Status

- Done: OTel tracing foundation, test-trace tool, instrumented FastAPI stub (locally
  verified — 42 tests pass; console shows `trace.classify`).
- Pending (needs the WSL2 Foundry backend): install, commit `casting.yaml`/lock, and the
  live "trace lands in SigNoz" confirmation of items 3 & 4.
