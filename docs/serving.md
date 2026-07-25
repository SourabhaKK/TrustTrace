# Serving — Dual-model Canary API

`POST /classify` evaluates every request with **both** the base model and the fine-tuned
model (one shared 4-bit model with the LoRA adapter toggled — see
`src/trusttrace/serving/classifier.py`) and returns both, each strictly Pydantic-validated.

> Detection only (CLAUDE.md): the response contains only parsed structured fields — never
> raw model text. If a path's output doesn't conform to the schema it returns
> `schema_valid=false` + `output=null`.

## API

```
GET  /healthz  -> {"status":"ok"}

POST /classify  {"text": "<input>"}   ->
{
  "primary": "finetuned",
  "finetuned": { "schema_valid": true,  "output": {"category":"...","severity":"...","confidence":1.0} },
  "base":      { "schema_valid": false, "output": null }
}
```

`output` is a validated `ClassificationOutput` (PRD §11.1) or `null`. Concurrency-safe: a
`threading.Lock` serializes the adapter-toggled inference so overlapping requests can't race
on the shared adapter state.

## Run locally (GPU)

```bash
pip install -r requirements.txt          # + a CUDA torch build (e.g. torch==2.5.1+cu121)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # SigNoz (optional)
uvicorn trusttrace.serving.app:app --port 8000
curl -XPOST localhost:8000/classify -H 'content-type: application/json' -d '{"text":"..."}'
```
The base model (`Qwen/Qwen2.5-1.5B-Instruct`) is fetched from HuggingFace on first request;
the LoRA adapter is loaded from `artifacts/adapter/`. First request loads the model (~15–20s).

## Run in Docker (GPU)

The `Dockerfile` is a multistage CUDA 12.1 build. **Requires GPU passthrough** — on Windows
that means WSL2 + the NVIDIA Container Toolkit (native Docker Engine), not Docker Desktop.

```bash
docker build -t trusttrace-serve .
docker run --gpus all -p 8000:8000 \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  trusttrace-serve
```

- `--gpus all` is required (bitsandbytes/nf4 need CUDA).
- Mount the HF cache so the base model isn't re-downloaded on each container start.
- Point `OTEL_EXPORTER_OTLP_ENDPOINT` at your SigNoz collector (from a container, the host is
  typically `host.docker.internal`; on Linux add `--add-host=host.docker.internal:host-gateway`).

> Status: the Dockerfile is written and the serving code is verified live on the GPU
> (`uvicorn` + `/classify`, both paths + span tree). The container **build/run is deferred to
> the author**, since GPU-in-container needs the WSL2 NVIDIA toolkit set up.
