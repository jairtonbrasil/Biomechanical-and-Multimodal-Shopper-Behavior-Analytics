"""Bounded timing summaries and structured, credential-free worker diagnostics."""

import json
import logging
import math
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from typing import Any


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields},
            allow_nan=False,
            default=str,
            separators=(",", ":"),
        ),
        flush=True,
    )


class WorkerLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = {
            name: getattr(record, name)
            for name in (
                "camera_id",
                "source_epoch",
                "attempt",
                "error_type",
                "width",
                "height",
                "source_fps",
                "timestamp_source",
            )
            if hasattr(record, name)
        }
        return json.dumps(
            {"timestamp": datetime.now(UTC).isoformat(), "event": record.getMessage(), **fields},
            default=str,
            allow_nan=False,
            separators=(",", ":"),
        )


def configure_worker_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(WorkerLogFormatter())
    logger = logging.getLogger("workers.vision_streamer.capture")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logging.getLogger("aiokafka").setLevel(logging.CRITICAL)


class Diagnostics:
    def __init__(self, *, sample_limit: int = 1024) -> None:
        self.started = time.monotonic()
        self.sample_limit = sample_limit
        self.counts: Counter = Counter()
        self.timings: dict[str, deque] = defaultdict(lambda: deque(maxlen=sample_limit))

    def record(self, name: str, milliseconds: float) -> None:
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("Timings require finite nonnegative milliseconds")
        self.timings[name].append(milliseconds)

    def snapshot(self) -> dict:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        percentiles = {}
        for name, values in sorted(self.timings.items()):
            ordered = sorted(values)
            percentiles[name] = {
                "p50": round(ordered[int((len(ordered) - 1) * 0.50)], 3),
                "p95": round(ordered[int((len(ordered) - 1) * 0.95)], 3),
                "samples": len(ordered),
                "window": f"last_{self.sample_limit}_samples",
                "method": "lower_order_statistic",
            }
        return {
            "elapsed_seconds": round(elapsed, 3),
            "counters": dict(self.counts),
            "processed_fps": round(self.counts["frames_processed"] / elapsed, 3),
            "timings_ms": percentiles,
            "physical_camera_latency_ms": None,
            "physical_camera_latency_reason": "Exposure-to-wall-clock mapping is unavailable",
        }
