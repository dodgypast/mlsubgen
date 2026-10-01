"""The job queue: one SQLite table in ~/mlsubgen/mlsubgen.db. A job is a `mlsubgen run` invocation — absolute paths plus
options — and its status. `mlsubgen serve` (the mlsubgen-worker service) runs queued jobs one at a time and survives
reboots: the process holds no state, the queue and the per-file work files do, so a job interrupted by a reboot
is simply picked up again and resumes from cached ASR. `mlsubgen` in a folder enqueues when the service is up and
runs inline when it is not (`--now` / `--queue` force either)."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import config

QUEUED, RUNNING, CANCELLING, DONE, FAILED, CANCELLED = "queued", "running", "cancelling", "done", "failed", "cancelled"
PAUSING, PAUSED = "pausing", "paused"   # 2026-09-27: a job stopped ON PURPOSE, kept out of the queue until `mlsubgen resume`;
                                         # survives reboots (the worker never requeues a paused job) and keeps its work files

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created    TEXT    NOT NULL,
    paths      TEXT    NOT NULL,             -- JSON list of absolute files/folders
    opts       TEXT    NOT NULL,             -- JSON list of the remaining `mlsubgen run` arguments
    label      TEXT    NOT NULL,
    status     TEXT    NOT NULL,             -- queued | running | cancelling | pausing | paused | done | failed | cancelled
    attempts   INTEGER NOT NULL DEFAULT 0,   -- starts (including resumes after an interruption)
    failures   INTEGER NOT NULL DEFAULT 0,   -- non-zero exits that were not interruptions
    not_before REAL    NOT NULL DEFAULT 0,   -- epoch seconds; a retry waits until then
    started    TEXT,
    finished   TEXT,
    exit_code  INTEGER,
    pid        INTEGER,
    log        TEXT,                         -- log file of the current / last attempt
    note       TEXT    NOT NULL DEFAULT ''   -- last event, human-readable
);
"""


@dataclass
class Job:
    id: int
    created: str
    paths: list[str]
    opts: list[str]
    label: str
    status: str
    attempts: int
    failures: int
    not_before: float
    started: str | None
    finished: str | None
    exit_code: int | None
    pid: int | None
    log: str | None
    note: str

    @staticmethod
    def from_row(r: sqlite3.Row) -> "Job":
        return Job(r["id"], r["created"], json.loads(r["paths"]), json.loads(r["opts"]), r["label"], r["status"],
                   r["attempts"], r["failures"], r["not_before"], r["started"], r["finished"], r["exit_code"],
                   r["pid"], r["log"], r["note"])

    def created_epoch(self) -> float:
        try:
            return time.mktime(time.strptime(self.created, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            return 0.0

    def argv(self) -> list[str]:
        """The `mlsubgen run` arguments the worker passes to the child. `--now` keeps it from re-enqueueing itself;
        `--since <created>` tells an --overwrite run which .srt files THIS job already rewrote (2026-09-27: before
        this, a rebooted or paused overwrite job started the folder over, because --overwrite means "every file
        is to do" and the finished files' work files are gone)."""
        return ["run", *self.paths, *self.opts, "--now", "--since", f"{self.created_epoch():.0f}"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def connect(db: Path = config.DB_PATH) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=30, isolation_level=None)   # autocommit; every statement is its own txn
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


def make_label(paths: list[str], opts: list[str]) -> str:
    shown = " ".join(paths[:2]) + (f" (+{len(paths) - 2})" if len(paths) > 2 else "")
    return (shown + " " + " ".join(opts)).strip()


def add(conn: sqlite3.Connection, paths: list[str], opts: list[str]) -> int:
    cur = conn.execute("INSERT INTO jobs (created, paths, opts, label, status) VALUES (?, ?, ?, ?, ?)",
                       (now(), json.dumps(paths), json.dumps(opts), make_label(paths, opts), QUEUED))
    return int(cur.lastrowid)


def get(conn: sqlite3.Connection, job_id: int) -> Job | None:
    r = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(r) if r else None


def update(conn: sqlite3.Connection, job_id: int, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def list_jobs(conn: sqlite3.Connection, limit: int = 20, all_jobs: bool = False) -> list[Job]:
    q = "SELECT * FROM jobs ORDER BY id DESC" + ("" if all_jobs else f" LIMIT {int(limit)}")
    return [Job.from_row(r) for r in conn.execute(q).fetchall()][::-1]


def next_queued(conn: sqlite3.Connection) -> Job | None:
    """Oldest queued job whose retry delay has passed."""
    r = conn.execute("SELECT * FROM jobs WHERE status = ? AND not_before <= ? ORDER BY id LIMIT 1",
                     (QUEUED, time.time())).fetchone()
    return Job.from_row(r) if r else None


def requeue_running(conn: sqlite3.Connection, note: str) -> list[int]:
    """At worker start: anything still marked running belonged to a worker that died (power cut, kill -9).
    Its child is gone with it; the job goes back to the queue and resumes from the work files. A job caught
    mid-pause stays paused — that was the intent."""
    for r in conn.execute("SELECT id FROM jobs WHERE status = ?", (PAUSING,)).fetchall():
        update(conn, r["id"], status=PAUSED, pid=None, note=f"paused (the worker restarted at {now()} while it was pausing)")
    ids = [r["id"] for r in conn.execute("SELECT id FROM jobs WHERE status IN (?, ?)", (RUNNING, CANCELLING)).fetchall()]
    for i in ids:
        update(conn, i, status=QUEUED, not_before=0, pid=None, note=note)
    return ids


def cancel(conn: sqlite3.Connection, job_id: int) -> str:
    """queued / paused → cancelled at once; running → cancelling (the worker interrupts the child within seconds)."""
    j = get(conn, job_id)
    if j is None:
        return "no such job"
    if j.status == QUEUED:
        update(conn, job_id, status=CANCELLED, finished=now(), note="cancelled before it started")
        return "cancelled"
    if j.status == PAUSED:
        update(conn, job_id, status=CANCELLED, finished=now(), note="cancelled while paused; finished files keep their .srt")
        return "cancelled"
    if j.status in (RUNNING, PAUSING):
        update(conn, job_id, status=CANCELLING, note="cancel requested")
        return "cancelling — the worker is stopping it; finished files keep their .srt"
    if j.status == CANCELLING:
        return "already cancelling"
    return f"nothing to cancel (job is {j.status})"


def pause(conn: sqlite3.Connection, job_id: int) -> str:
    """queued → paused at once; running → pausing (the worker interrupts the child, which keeps every finished
    stage in its work files, and the job waits — across reboots — for `mlsubgen resume`)."""
    j = get(conn, job_id)
    if j is None:
        return "no such job"
    if j.status == QUEUED:
        update(conn, job_id, status=PAUSED, pid=None, note="paused before it started")
        return "paused"
    if j.status == RUNNING:
        update(conn, job_id, status=PAUSING, note="pause requested")
        return "pausing — the worker is stopping it; it resumes from its work files on `mlsubgen resume`"
    if j.status in (PAUSING, PAUSED):
        return f"already {j.status}"
    return f"nothing to pause (job is {j.status})"


def resume(conn: sqlite3.Connection, job_id: int) -> str:
    """paused → queued (the run picks up from the work files; an --overwrite job skips what it already rewrote)."""
    j = get(conn, job_id)
    if j is None:
        return "no such job"
    if j.status == PAUSED:
        update(conn, job_id, status=QUEUED, not_before=0, pid=None, note=f"resumed at {now()}")
        return "queued — resumes where it stopped"
    if j.status == PAUSING:
        return "still pausing — try again in a moment"
    return f"nothing to resume (job is {j.status})"


def retry(conn: sqlite3.Connection, job_id: int) -> str:
    """failed / cancelled / done → queued again (a finished folder job simply picks up new files)."""
    j = get(conn, job_id)
    if j is None:
        return "no such job"
    if j.status in (PAUSED, PAUSING):
        return f"job is {j.status} — `mlsubgen resume {job_id}` continues it"
    if j.status in (QUEUED, RUNNING, CANCELLING):
        return f"job is already {j.status}"
    update(conn, job_id, status=QUEUED, failures=0, not_before=0, finished=None, exit_code=None, pid=None,
           note=f"requeued by hand at {now()}")
    return "queued"


FINAL = (DONE, FAILED, CANCELLED)


def purge(conn: sqlite3.Connection, ids: list[int] | None = None, statuses: tuple[str, ...] = (FAILED, CANCELLED)) -> int:
    """Remove finished jobs from the queue table so they stop showing in the listing — by id, or every job in
    `statuses` (default: failed and cancelled; pass FINAL for done as well). Never a queued or running job; the
    job logs in LOG_DIR are untouched."""
    if ids:
        q = f"DELETE FROM jobs WHERE id IN ({','.join('?' * len(ids))}) AND status IN (?, ?, ?)"
        cur = conn.execute(q, (*ids, *FINAL))
    else:
        cur = conn.execute(f"DELETE FROM jobs WHERE status IN ({','.join('?' * len(statuses))})", tuple(statuses))
    return cur.rowcount


def fmt_table(jobs: list[Job]) -> str:
    if not jobs:
        return "no jobs yet — `mlsubgen /folder` queues one while the service is up (mlsubgen help)"
    lines = [f"{'id':>4}  {'status':<10} {'tries':>5}  {'started':<16}  label / note"]
    for j in jobs:
        label = j.label if len(j.label) <= 70 else j.label[:67] + "…"
        lines.append(f"{j.id:>4}  {j.status:<10} {j.attempts:>5}  {(j.started or j.created)[:16]:<16}  {label}")
        if j.note:
            lines.append(f"{'':>4}  {'':<10} {'':>5}  {'':<16}  ↳ {j.note}")
        if j.log and j.status in (RUNNING, CANCELLING, PAUSING):
            lines.append(f"{'':>4}  {'':<10} {'':>5}  {'':<16}  ↳ tail -f {j.log}")
    return "\n".join(lines)
