FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-runtime.txt requirements-runtime.lock requirements-registry-client.lock ./
RUN python -m pip wheel --wheel-dir /wheels -r requirements-runtime.lock -r requirements-registry-client.lock

FROM python:3.11-slim
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/xm2325/regulated_ml_platform" \
      org.opencontainers.image.description="Regulated AI MLOps platform API" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=container \
    PLATFORM_VERSION=1.2.0 \
    SERVICE_VERSION=1.2.0 \
    MODEL_RELEASE_VERSION=0.6.0 \
    REGISTRY_CACHE_DIR=/var/lib/regulated-ai/registry-cache \
    GIT_COMMIT=${GIT_COMMIT}
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /var/lib/regulated-ai/registry-cache \
    && chown -R 10001:10001 /var/lib/regulated-ai
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY --chown=app:app src ./src
COPY --chown=app:app models ./models
COPY --chown=app:app config ./config
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
