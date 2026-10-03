"""`mlsubgen serve` — the worker behind the mlsubgen-worker service.

It is a supervisor, not a pipeline: it takes the oldest queued job, waits until the machine is ready for it
(media mounts present, GPU visible, the LLM server answering, no other mlsubgen run on the GPU), then runs the
job as a child `mlsubgen run … --now` with the child's output in its own log file. One job at a time — there is one
GPU. Stopping the service (a reboot, `systemctl --user stop mlsubgen-worker`) sends the child SIGINT, which is the
same clean interruption as Ctrl-C: the work files keep every finished stage, the job goes back to `queued`, and
the next start resumes it. `mlsubgen cancel ID` does the same for one job and leaves it cancelled. A job that exits
with an error is retried a few times with a delay, then marked failed with its log path in the note.
"""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, jobs, metrics

STOP_GRACE_SEC = 90          # after a stop request, how long the child may take before it is killed
INTERRUPT_CODES = (130, -2)  # exit codes of a SIGINT-interrupted `mlsubgen run`
NON_RUN_COMMANDS = {"serve", "web", "jobs", "log", "cancel", "retry", "pause", "resume", "purge", "models", "tracks", "help",
                    "selftest", "clean", "bench", "add", "-h", "--help", "--version"}


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.LOG_DIR / "worker.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ── the lock: one worker, and the way `mlsubgen` knows whether to queue or run inline ───────────────────────
def _open_lock() -> int:
    config.WORKER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    return os.open(config.WORKER_LOCK, os.O_RDWR | os.O_CREAT, 0o644)


def is_running() -> bool:
    """True when a worker holds the lock (the service is up), so `mlsubgen` should queue rather than run inline."""
    if not config.WORKER_LOCK.exists():
        return False
    fd = _open_lock()
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


# ── readiness ────────────────────────────────────────────────────────────────────────────────────────────
def nfs_mountpoints(fstab: Path = Path("/etc/fstab")) -> list[str]:
    """Network mounts declared in fstab — an unmounted one is an empty directory, which would look like an empty
    folder of videos and finish the job with nothing done."""
    out: list[str] = []
    try:
        for line in fstab.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and not parts[0].startswith("#") and parts[2] in ("nfs", "nfs4", "cifs", "smb3"):
                out.append(parts[1].rstrip("/") or "/")
    except OSError:
        pass
    return out


def media_roots() -> list[str]:
    """Where the videos live: MLSUBGEN_MEDIA_ROOTS (colon-separated; set in the Docker compose file), else the network
    mounts in fstab (the host), else the home directory. The folder picker is confined to these."""
    env = [r for r in os.environ.get("MLSUBGEN_MEDIA_ROOTS", "").split(":") if r]
    roots = env or nfs_mountpoints() or [str(config.HOME)]
    return [os.path.normpath(r) for r in roots]


def _under(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip("/") + "/")


def paths_ready(paths: list[str], mounts: list[str] | None = None, roots: list[str] | None = None) -> str | None:
    """None when every job path is reachable; else why not. An fstab network mount must be mounted; a media root
    must be a non-empty directory (inside a container an unmounted share is just an empty bind-mounted folder)."""
    mounts = nfs_mountpoints() if mounts is None else mounts
    roots = media_roots() if roots is None else roots
    for p in paths:
        for m in mounts:
            if _under(p, m) and not os.path.ismount(m):
                return f"{m} is not mounted"
        for r in roots:
            if _under(p, r) and Path(r).is_dir() and not any(True for _ in os.scandir(r)):
                return f"{r} is empty — share not mounted yet?"
        if not Path(p).exists():
            return f"{p} does not exist (mount not up yet?)"
    return None


def gpu_ready() -> str | None:
    if not any(os.access(os.path.join(d, "nvidia-smi"), os.X_OK) for d in os.environ.get("PATH", "").split(":")):
        return None                                        # no nvidia-smi: nothing to check
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return None if r.returncode == 0 else f"nvidia-smi failed: {(r.stderr or r.stdout).strip()[:120]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"nvidia-smi: {e}"


def llm_ready(opts: list[str]) -> str | None:
    """The job's LLM server answers. `--backend`/`--url` in the job's options are honoured."""
    backend, url = config.LLM_BACKEND, config.LLM_URL
    for i, t in enumerate(opts):
        if t == "--backend" and i + 1 < len(opts):
            backend = opts[i + 1]
        elif t.startswith("--backend="):
            backend = t.split("=", 1)[1]
        elif t == "--url" and i + 1 < len(opts):
            url = opts[i + 1]
        elif t.startswith("--url="):
            url = t.split("=", 1)[1]
    probe = url.rstrip("/") + ("/api/tags" if backend == "ollama" else "/v1/models")
    try:
        with urllib.request.urlopen(urllib.request.Request(probe), timeout=5) as resp:
            resp.read(64)
        return None
    except urllib.error.HTTPError:
        return None                                        # it answered — whether it likes the request is the run's business
    except Exception as e:  # noqa: BLE001
        return f"{backend} at {url} not answering ({e})"


def other_mlsubgen_running(ignore: set[int]) -> str | None:
    """A `mlsubgen run` started outside the queue (a hand-run `--now`, an old-style detached run) owns the GPU."""
    me = os.getpid()
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        if pid == me or pid in ignore:
            continue
        try:
            argv = (pid_dir / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        if len(argv) >= 3 and argv[1] == "-m" and argv[2] == "mlsubgen" and "python" in os.path.basename(argv[0]):
            rest = argv[3:]
            if not rest or rest[0] not in NON_RUN_COMMANDS:
                return f"another mlsubgen run is active outside the queue (pid {pid}: {' '.join(rest)[:80]})"
    return None


def not_ready(job: jobs.Job) -> str | None:
    """Mounts, GPU, LLM server — the things that should come back within minutes (these are on a deadline)."""
    return paths_ready(job.paths) or gpu_ready() or llm_ready(job.opts)


# ── the worker ───────────────────────────────────────────────────────────────────────────────────────────
class Worker:
    def __init__(self, poll: float = config.WORKER_POLL_SEC):
        self.poll = poll
        self.stop = threading.Event()
        self.stop_at: float | None = None
        self.child: subprocess.Popen | None = None
        self.conn = jobs.connect()

    # signals: SIGINT/SIGTERM = stop the worker; the running child gets SIGINT, the job is requeued
    def _on_signal(self, signum, frame) -> None:      # noqa: ARG002
        if not self.stop.is_set():
            _log(f"stop requested ({signal.Signals(signum).name})")
            self.stop.set()
            self.stop_at = time.time()
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.send_signal(signal.SIGINT)
            except OSError:
                pass

    def serve(self) -> int:
        fd = _open_lock()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _log("another worker holds the lock — is mlsubgen-worker already running?")
            return 1
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        resumed = jobs.requeue_running(self.conn, note=f"worker restarted at {jobs.now()} — resuming from the work files")
        _log(f"worker started (pid {os.getpid()}, poll {self.poll:.0f}s, db {config.DB_PATH})"
             + (f" — requeued {resumed}" if resumed else ""))
        waiting: dict[int, tuple[float, str]] = {}        # job id → (since, last reason)
        if metrics.enabled():
            _log(f"metrics → {metrics.PATH}")
        try:
            while not self.stop.is_set():
                job = jobs.next_queued(self.conn)
                metrics.publish(busy=False, queued=self._queued())
                if job is None:
                    self.stop.wait(self.poll)
                    continue
                busy = other_mlsubgen_running(set())
                if busy:                                   # a run outside the queue: legitimate and open-ended, no deadline
                    if waiting.get(job.id, (0.0, ""))[1] != busy:
                        _log(f"job {job.id} waiting: {busy}")
                        jobs.update(self.conn, job.id, note=f"waiting: {busy}")
                    waiting[job.id] = (time.time(), busy)
                    self.stop.wait(self.poll)
                    continue
                reason = not_ready(job)
                if reason:
                    since, last = waiting.get(job.id, (time.time(), ""))
                    if reason != last:
                        _log(f"job {job.id} waiting: {reason}")
                        jobs.update(self.conn, job.id, note=f"waiting: {reason}")
                    waiting[job.id] = (since, reason)
                    if time.time() - since > config.JOB_READY_TIMEOUT_SEC:
                        _log(f"job {job.id} failed: still not ready after {config.JOB_READY_TIMEOUT_SEC}s ({reason})")
                        jobs.update(self.conn, job.id, status=jobs.FAILED, finished=jobs.now(),
                                    note=f"gave up waiting: {reason} — `mlsubgen retry {job.id}` once it is fixed")
                        waiting.pop(job.id, None)
                    else:
                        self.stop.wait(self.poll)
                    continue
                waiting.pop(job.id, None)
                self.run_job(job)
        finally:
            _log("worker stopped")
            metrics.publish(busy=False, queued=self._queued(), up=False)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        return 0

    def _queued(self) -> int:
        try:
            return self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (jobs.QUEUED,)).fetchone()[0]
        except Exception:                                  # noqa: BLE001
            return 0

    def run_job(self, job: jobs.Job) -> None:
        attempt = job.attempts + 1
        log_path = config.LOG_DIR / f"job{job.id}-{time.strftime('%Y%m%d-%H%M')}.log"
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "mlsubgen", *job.argv()]
        env = {**os.environ, "PYTHONPATH": str(config.CODE_DIR) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
        with open(log_path, "ab") as lf:
            lf.write(f"# mlsubgen job {job.id}, attempt {attempt}: mlsubgen {' '.join(cmd[3:])}\n".encode())
            lf.flush()
            try:
                # own session: a Ctrl-C in a terminal running `mlsubgen serve` reaches only the worker, which forwards
                # exactly one SIGINT (two in a row would abort the child's clean-up)
                self.child = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                              cwd=str(config.MLSUBGEN_HOME), env=env, start_new_session=True)
            except OSError as e:
                _log(f"job {job.id}: cannot start {cmd[0]}: {e}")
                jobs.update(self.conn, job.id, status=jobs.FAILED, finished=jobs.now(), note=f"cannot start: {e}")
                return
            jobs.update(self.conn, job.id, status=jobs.RUNNING, attempts=attempt, started=jobs.now(), finished=None,
                        exit_code=None, pid=self.child.pid, log=str(log_path), note="")
            _log(f"job {job.id} started (attempt {attempt}, pid {self.child.pid}): {job.label}  → {log_path.name}")
            metrics.publish(busy=True, queued=self._queued(), job_id=job.id)
            cancelled = paused = False
            beat = time.time()
            while True:
                try:
                    rc = self.child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if time.time() - beat >= 60:               # a heartbeat a minute while the job runs
                    metrics.publish(busy=True, queued=self._queued(), job_id=job.id)
                    beat = time.time()
                cur = jobs.get(self.conn, job.id)
                if cur is not None and cur.status == jobs.CANCELLING and not cancelled:
                    cancelled = True
                    _log(f"job {job.id}: cancel requested — interrupting the child")
                    self.child.send_signal(signal.SIGINT)
                elif cur is not None and cur.status == jobs.PAUSING and not paused and not cancelled:
                    paused = True                      # the same clean interruption as a stop; the job then waits
                    _log(f"job {job.id}: pause requested — interrupting the child")
                    self.child.send_signal(signal.SIGINT)
                if self.stop.is_set() and self.stop_at and time.time() - self.stop_at > STOP_GRACE_SEC:
                    _log(f"job {job.id}: child did not stop within {STOP_GRACE_SEC}s — killing it")
                    self.child.kill()
                    self.stop_at = time.time() + 3600
        self.child = None
        metrics.publish(busy=False, queued=self._queued())
        finished = jobs.now()
        if cancelled:
            jobs.update(self.conn, job.id, status=jobs.CANCELLED, finished=finished, exit_code=rc, pid=None,
                        note=f"cancelled at {finished} (exit {rc}); finished files keep their .srt")
            _log(f"job {job.id} cancelled (exit {rc})")
        elif paused:
            jobs.update(self.conn, job.id, status=jobs.PAUSED, exit_code=rc, pid=None,
                        note=f"paused at {finished} (exit {rc}) — every finished stage is in the work files; "
                             f"`mlsubgen resume {job.id}` continues from there, reboots leave it paused")
            _log(f"job {job.id} paused (exit {rc})")
        elif self.stop.is_set():
            jobs.update(self.conn, job.id, status=jobs.QUEUED, not_before=0, exit_code=rc, pid=None,
                        note=f"interrupted at {finished} by a worker stop (exit {rc}) — resumes on the next start")
            _log(f"job {job.id} interrupted by the stop (exit {rc}) — requeued")
        elif rc == 0:
            jobs.update(self.conn, job.id, status=jobs.DONE, finished=finished, exit_code=0, pid=None,
                        note=f"done at {finished}")
            _log(f"job {job.id} done")
        elif rc in INTERRUPT_CODES:
            jobs.update(self.conn, job.id, status=jobs.QUEUED, not_before=time.time() + 60, exit_code=rc, pid=None,
                        note=f"interrupted from outside the queue at {finished} (exit {rc}) — requeued; "
                             f"`mlsubgen cancel {job.id}` stops it for good")
            _log(f"job {job.id} interrupted from outside (exit {rc}) — requeued")
        else:
            failures = job.failures + 1
            if failures < config.JOB_MAX_FAILURES:
                delay = config.JOB_RETRY_DELAY_SEC
                jobs.update(self.conn, job.id, status=jobs.QUEUED, failures=failures, not_before=time.time() + delay,
                            exit_code=rc, pid=None,
                            note=f"attempt {attempt} failed (exit {rc}) at {finished} — retry in {delay // 60} min; "
                                 f"log: {log_path}")
                _log(f"job {job.id} failed (exit {rc}), retry {failures}/{config.JOB_MAX_FAILURES - 1} in {delay}s")
            else:
                jobs.update(self.conn, job.id, status=jobs.FAILED, failures=failures, finished=finished, exit_code=rc,
                            pid=None, note=f"failed {failures} times (last exit {rc}) — log: {log_path}; "
                                           f"`mlsubgen retry {job.id}` after fixing it")
                _log(f"job {job.id} FAILED after {failures} attempts (exit {rc}) — {log_path}")
