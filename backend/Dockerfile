FROM python:3.12.12-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS runtime-dependencies
COPY requirements.lock ./requirements.lock
RUN pip install --requirement requirements.lock && useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/shopper/references && chown appuser:appuser /var/lib/shopper/references
USER appuser

FROM runtime-dependencies AS runtime
COPY app ./app
COPY workers ./workers
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM runtime-dependencies AS capture-dependencies
USER root
RUN apt-get update && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER appuser

FROM capture-dependencies AS capture
COPY --from=runtime /app /app
CMD ["python", "-m", "workers.reference_capture.main"]

FROM capture-dependencies AS vision-dependencies
USER root
COPY requirements-vision.lock ./requirements-vision.lock
RUN pip install --requirement requirements-vision.lock \
    && mkdir -p /outputs && chown appuser:appuser /outputs
USER appuser
ENV YOLO_CONFIG_DIR=/tmp/ultralytics YOLO_AUTOINSTALL=false

FROM vision-dependencies AS vision
COPY --from=runtime /app /app
CMD ["python", "-m", "workers.vision_streamer.main"]

FROM base AS test
COPY requirements-dev.lock ./requirements-dev.lock
RUN pip install --requirement requirements-dev.lock && useradd --create-home --uid 10001 appuser
COPY . .
USER appuser
CMD ["pytest", "-v", "-p", "no:cacheprovider"]
