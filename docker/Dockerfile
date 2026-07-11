FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-runtime.txt requirements-runtime.lock ./
RUN python -m pip wheel --wheel-dir /wheels -r requirements-runtime.lock

FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/xm2325/regulated_ml_platform" \
      org.opencontainers.image.description="Regulated AI MLOps platform API" \
      org.opencontainers.image.version="0.8.0" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=container SERVICE_VERSION=0.8.0
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --home-dir /app app
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
