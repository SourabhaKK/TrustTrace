# syntax=docker/dockerfile:1
#
# TrustTrace serving image — dual-model canary (base + QLoRA adapter) on GPU.
# Multistage: a builder installs the serving deps into a venv layered on the CUDA/torch base
# (via --system-site-packages so torch comes from the base, not reinstalled), then a slim
# runtime copies just the venv + code. Requires an NVIDIA GPU at run time (bitsandbytes/nf4):
#   docker build -t trusttrace-serve .
#   docker run --gpus all -p 8000:8000 \
#     -e OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     trusttrace-serve
# (See docs/serving.md. The base model is pulled from HF on first request; mount the HF
# cache to avoid re-downloading.)

# ---- Builder: serving deps on top of the CUDA 12.1 / torch 2.5.1 base ----
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv --system-site-packages /venv
ENV PATH=/venv/bin:$PATH
# Serving deps only (torch + CUDA come from the base image; no training/data deps).
RUN pip install \
    "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2.0" "safetensors" \
    "transformers>=4.45" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" \
    "opentelemetry-sdk>=1.44" "opentelemetry-exporter-otlp>=1.44" \
    "opentelemetry-instrumentation-fastapi>=0.65b0"

# ---- Runtime ----
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS runtime
COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:4317 \
    OTEL_ENV=docker
WORKDIR /app
# Code + the trained LoRA adapter (base model is fetched from HF at first request).
COPY src/ /app/src/
COPY artifacts/adapter/ /app/artifacts/adapter/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "trusttrace.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
