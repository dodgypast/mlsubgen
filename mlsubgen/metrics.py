"""Metrics for Prometheus through node_exporter's textfile collector (0.5.0.11, 2026-10-03).

The worker writes one small file while it runs — whether it is mid-job, which job, how many are queued, when it
last reported — and node_exporter serves it with everything else on :9100. The point is one alert rule: a
transcription keeps every core busy for the length of a film, and a CPU-saturation alert that cannot tell a job
from a fault fires every time. With `mlsubgen_worker_busy` in Prometheus the rule reads
    cpu > 95 % for 30 min  AND  mlsubgen_worker_busy == 0
and a busy worker is not an alert.

Set MLSUBGEN_METRICS_FILE to a `.prom` path inside the collector's directory (the worker must be allowed to write
there); unset, nothing is written and the worker is unchanged. Writes are atomic (rename), so a scrape never sees
half a file. On the worker's exit the file says busy 0 and up 0, so a dead worker reads as idle rather than stale.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

PATH = os.environ.get("MLSUBGEN_METRICS_FILE", "").strip()


def enabled() -> bool:
    return bool(PATH)


def publish(busy: bool, queued: int, job_id: int | None = None, up: bool = True) -> None:
    if not PATH:
        return
    lines = [
        "# HELP mlsubgen_worker_busy 1 while the mlsubgen worker runs a job",
        "# TYPE mlsubgen_worker_busy gauge",
        f"mlsubgen_worker_busy {1 if busy else 0}",
        "# HELP mlsubgen_worker_job_id the id of the running job, 0 when idle",
        "# TYPE mlsubgen_worker_job_id gauge",
        f"mlsubgen_worker_job_id {job_id or 0}",
        "# HELP mlsubgen_jobs_queued jobs waiting in the queue",
        "# TYPE mlsubgen_jobs_queued gauge",
        f"mlsubgen_jobs_queued {queued}",
        "# HELP mlsubgen_worker_up 1 while the worker process is alive",
        "# TYPE mlsubgen_worker_up gauge",
        f"mlsubgen_worker_up {1 if up else 0}",
        "# HELP mlsubgen_worker_heartbeat unix time of the worker's last report",
        "# TYPE mlsubgen_worker_heartbeat gauge",
        f"mlsubgen_worker_heartbeat {int(time.time())}",
        "",
    ]
    p = Path(PATH)
    tmp = p.with_name(p.name + ".tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass                                               # metrics are best effort; the worker's job is the queue
