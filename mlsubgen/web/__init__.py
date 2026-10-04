"""`mlsubgen web` — the browser front end. A view over the job queue (mlsubgen.db) and the job logs, plus a form that
queues new jobs and a page of skipped files that can be re-queued with --assume-ja. It never runs the pipeline
itself: the mlsubgen-worker service does, and a job queued here is exactly `mlsubgen /folder [options]`.
No login unless MLSUBGEN_WEB_AUTH=user:password is set (HTTP Basic, every route) — bind it to a LAN or Tailscale
address, or put it behind a reverse proxy with its own login, rather than on the internet."""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__, config, jobs, worker
from ..config import TRANSLATORS

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
FINAL = (jobs.DONE, jobs.FAILED, jobs.CANCELLED)


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────
def media_roots() -> list[str]:
    """Folders the picker may browse (see worker.media_roots)."""
    return worker.media_roots()


def form_paths(form) -> list[Path]:
    """The selection of a new-job form: ticked files (`paths`, repeated) if any, else the single `path` field."""
    picked = [safe_path(x) for x in form.getlist("paths") if x.strip()]
    if picked:
        return picked
    raw = (form.get("path") or "").strip()
    if not raw:
        raise HTTPException(400, "pick a folder or file")
    return [safe_path(raw)]


def last_folder() -> str:
    """The folder of the most recent job's first path — what the path field defaults to."""
    conn = jobs.connect()
    try:
        latest = jobs.list_jobs(conn, limit=1)
    finally:
        conn.close()
    if not latest or not latest[0].paths:
        return ""
    p = Path(latest[0].paths[0])
    return str(p.parent if p.suffix.lower() in config.VIDEO_EXTS else p)


def safe_path(raw: str) -> Path:
    """A path under one of the media roots, or 400 — the picker is not a file manager for the whole box."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise HTTPException(400, "absolute paths only")
    p = Path(os.path.normpath(p))
    for root in media_roots():
        if p == Path(root) or str(p).startswith(root.rstrip("/") + "/"):
            return p
    raise HTTPException(400, f"{p} is outside the media roots ({', '.join(media_roots())})")


def build_opts(f: dict) -> list[str]:
    """Form fields → `mlsubgen run` options (only what differs from the defaults, so the label stays readable)."""
    opts: list[str] = []
    # the "Subtitles in" list: one tick box per offered target (config.TARGET_LANGS — the withheld ones are not listed)
    targets = [t for t in config.TARGET_LANGS if f.get(f"target_{t}")]
    extra_t = (f.get("targets") or "").strip()
    if extra_t:
        targets += [t.strip().lower() for t in extra_t.split(",") if t.strip() and t.strip().lower() not in targets]
    if not targets:
        raise HTTPException(400, "tick at least one subtitle language")
    if ",".join(targets) != config.DEFAULT_TARGETS:
        opts += ["--target", ",".join(targets)]
    src = (f.get("source") or "auto").strip()
    if src and src != "auto":
        opts += ["--source", src]
    tr = (f.get("translator") or "auto").strip()
    if tr and tr != "auto":
        opts += ["-t", tr]
    # the hardware profile (2026-10-04): auto = by the card's memory; a chosen one goes to the job's own process as
    # --profile, which sets its routes, translator sizes and the languages it may be asked for
    prof = (f.get("profile") or "auto").strip()
    if prof and prof != "auto":
        if prof not in config.PROFILES:
            raise HTTPException(400, f"unknown profile {prof!r}")
        opts += ["--profile", prof]
        bad = [t for t in targets if t in config.PROFILE_WITHHELD.get(prof, set())]
        if bad:
            raise HTTPException(400, f"not offered on the {prof} profile: {', '.join(config.LANG_NAMES.get(b, b) for b in bad)}")
    asr = (f.get("asr") or "dual").strip()
    if asr and asr != config.ASR_ENGINE:
        opts += ["--asr", asr]
    for flag in ("overwrite", "keep_work", "keep_source", "no_recursive", "force_translate"):
        if f.get(flag):
            opts.append("--" + flag.replace("_", "-"))
    subs = (f.get("subs") or "auto").strip()
    if subs != "auto":
        opts += ["--subs", subs]
    spk = (f.get("speakers") or "off").strip().lower()
    if spk != "off":
        opts += ["--speakers", spk]
    batch = (f.get("batch") or "").strip()
    if batch and batch != "10":
        opts += ["--batch", str(int(batch))]
    ctx = (f.get("context") or "").strip()
    if ctx:
        opts += ["--context", ctx]
    genre = (f.get("genre") or "").strip()
    if genre:
        opts += ["--genre", genre]
    gl = (f.get("glossary") or "").strip()
    if gl:
        d = config.MLSUBGEN_HOME / "glossaries"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{time.strftime('%Y%m%d-%H%M%S')}.tsv"
        path.write_text(gl + "\n", encoding="utf-8")
        opts += ["--glossary", str(path)]
    extra = (f.get("extra") or "").strip()
    if extra:
        opts += extra.split()
    return opts


def gpu_status() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        used, total, util = [int(x.strip()) for x in r.stdout.strip().split(",")]
        return {"ok": r.returncode == 0, "used_mb": used, "total_mb": total, "util": util}
    except Exception:  # noqa: BLE001
        return {"ok": False}


def status() -> dict:
    conn = jobs.connect()
    try:
        running = [j for j in jobs.list_jobs(conn, all_jobs=True) if j.status in (jobs.RUNNING, jobs.CANCELLING, jobs.PAUSING)]
        queued = conn.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (jobs.QUEUED,)).fetchone()[0]
        latest = jobs.list_jobs(conn, limit=1)
    finally:
        conn.close()
    return {"worker": worker.is_running(), "llm": worker.llm_ready([]) is None, "gpu": gpu_status(),
            "running": [job_dict(j) for j in running], "queued": queued, "version": __version__,
            "latest": job_dict(latest[0]) if latest else None, "now": time.time(),
            "other_run": worker.other_mlsubgen_running(set())}


_BANNER = re.compile(r"^mlsubgen [\d.]+: (\d+) file\(s\)")
_SCAN_FOUND = re.compile(r"^\[scan\] (\d+) video\(s\) found")
_SCAN_PROGRESS = re.compile(r"^\[scan\] (\d+)/(\d+) checked · (\d+) to do")


def progress_of(j: jobs.Job) -> dict | None:
    """'file 6 of 20' for a running job, from its log: the banner gives the total, every stage-1 header
    ('=== file') is a file entering the pipeline, a stage-2 header ('=== file → en') means translating."""
    if j.status not in (jobs.RUNNING, jobs.CANCELLING, jobs.PAUSING) or not j.log or not Path(j.log).exists():
        return None
    total = started = 0
    current = phase = ""
    scan_done = scan_total = scan_todo = 0
    try:
        with open(j.log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("[scan] "):
                    m = _SCAN_FOUND.match(line)
                    if m:
                        scan_total, scan_done, scan_todo = int(m.group(1)), 0, 0
                    m = _SCAN_PROGRESS.match(line)
                    if m:
                        scan_done, scan_total, scan_todo = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif line.startswith("mlsubgen "):
                    m = _BANNER.match(line)
                    if m:
                        total, started = int(m.group(1)), 0      # a retry restarts the count
                elif line.startswith("=== "):
                    head = line[4:].strip()
                    if " ⇄ " in head:
                        current, phase = head.split(" ⇄ ")[0], "reconciling transcripts"
                        continue
                    name, _, target = head.partition(" → ")
                    if target:
                        current, phase = name, f"translating → {target}"
                    else:
                        started += 1; current, phase = name, "detecting / transcribing"
                elif line.startswith("[lid] ") and phase.startswith("detecting"):
                    phase = "transcribing"
                elif line.startswith("[srt] "):
                    phase = "writing subtitles"
    except OSError:
        return None
    if not total:
        if scan_total:                                          # still in the planning pass: no banner yet
            return {"total": scan_total, "started": 0, "current": "", "phase": "scanning",
                    "label": f"scanning {scan_done} of {scan_total} · {scan_todo} to do so far"}
        return None
    return {"total": total, "started": started, "current": current, "phase": phase,
            "label": f"file {min(started, total)} of {total}"}


def job_dict(j: jobs.Job) -> dict:
    d = j.__dict__.copy()
    d["final"] = j.status in FINAL
    d["progress"] = progress_of(j)
    return d


_SRT_NAME = re.compile(r"^(.*)\.([A-Za-z]{2,3})\.srt$")


_SUBS_EMBEDDED = re.compile(r"^(\w+) subtitles are embedded ")
_SUBS_TRANSCRIPT = re.compile(r"^(?:embedded )?(\w+) subtitles \(([^)]*)\): (\d+) cues — used as the transcript")
_TL_PROGRESS = re.compile(r"(\d+)/(\d+)")


def parse_log(path: Path) -> dict:
    """Per-file outcomes and the current progress line, from a job's log."""
    files: dict[str, dict] = {}
    order: list[str] = []
    cur: str | None = None
    existing = 0
    summary = round_ = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"files": [], "existing": 0, "summary": "", "round": ""}

    def entry(name: str) -> dict:
        if name not in files:
            files[name] = {"name": name, "status": "queued", "detail": ""}
            order.append(name)
        return files[name]

    for line in lines:
        if line.startswith("=== "):
            head = line[4:].strip()
            if " ⇄ " in head:                                          # "=== file ⇄ merge": the dual-engine reconciliation
                cur = head.split(" ⇄ ")[0]
                e = entry(cur); e["status"] = "working"; e["detail"] = "reconciling the two transcripts"
                continue
            cur, _, target = head.partition(" → ")                     # stage 2 headers read "=== file → en"
            e = entry(cur); e["status"] = "working"; e["detail"] = f"translating → {target}" if target else "starting"
        elif line.startswith("[skip] "):
            name, _, reason = line[7:].partition(": ")
            e = entry(name.strip()); e["status"] = "skipped"; e["detail"] = reason.strip()
        elif line.startswith("skip (exists): "):
            existing += 1
        elif line.startswith("[scan] "):
            round_ = line[7:].strip()                                 # the first "━━━ round" line replaces it
        elif line.startswith("[srt] "):
            name, _, rest = line[6:].partition(": ")
            m = _SRT_NAME.match(name.strip())
            stem = m.group(1) if m else name.strip()
            for k in files:
                if Path(k).stem == stem:
                    files[k]["status"] = "done"; files[k]["detail"] = rest.strip()
                    break
        elif line.startswith("[subs] nothing to write") and cur:
            e = entry(cur); e["status"] = "done"; e["detail"] = "every target already embedded — nothing written"
        elif line.startswith("[subs] ") and cur:
            e = entry(cur); body = line[7:]
            m = _SUBS_EMBEDDED.match(body)
            if m:
                e["detail"] = (e["detail"] + " · " if e["detail"] not in ("", "starting") else "") + f"{m.group(1)} embedded — left alone"
            elif " used as the transcript" in body:
                m2 = _SUBS_TRANSCRIPT.search(body)
                e["detail"] = f"transcript: {m2.group(3)} {m2.group(1)} cues ({m2.group(2)})" if m2 else "transcript from subtitles"
            elif " — ignored" in body or "have only" in body:
                e["detail"] = "subtitle track unusable — transcribing the audio"
        elif line.startswith("━━━ "):
            round_ = line.strip("━ ").strip()
        elif line.startswith("done: "):
            summary = line[6:].strip()
        elif cur and files.get(cur, {}).get("status") == "working":
            e = files[cur]
            if line.startswith("[asr] chunk"):
                e["detail"] = "ASR " + line[6:].split("  ")[0].strip()
            elif line.startswith("[tl:"):
                prog = line.split("] ", 1)[1].split("  ")[0].strip()
                m = _TL_PROGRESS.search(prog)
                if m and m.group(1) == m.group(2):
                    e["detail"] = f"translated ({m.group(1)} cues) — written at the end of the round"
                else:
                    e["detail"] = "translating " + prog
            elif line.startswith("[tl] cached"):
                e["detail"] = "translation cached — written at the end of the round"
            elif line.startswith("[audio]"):
                e["detail"] = "audio extracted"
            elif line.startswith("[vad]"):
                e["detail"] = line[6:].split(",")[0].strip()
            elif line.startswith("[merge] ") and not line.startswith("[merge] ⚠"):
                e["detail"] = "transcripts reconciled — building cues"
            elif line.startswith("[asr] dual done"):
                e["detail"] = line[6:].split(":", 1)[-1].strip()
            elif line.startswith("[lid]"):
                e["detail"] = "language: " + line[6:].split("  (")[0].strip()
            elif line.startswith("[chunks]"):
                e["detail"] = line[9:].strip()
            elif line.startswith("[cues]"):
                e["detail"] = line[7:].split("  ")[0].strip() + " — waiting for the translator"
            elif line.startswith("[asr] cached") or line.startswith("[asr] loaded"):
                e["detail"] = line[6:].strip()
    return {"files": [files[k] for k in order], "existing": existing, "summary": summary, "round": round_}


def read_skipped() -> list[dict]:
    """skipped.log → the latest verdict per file, newest first."""
    latest: dict[str, dict] = {}
    p = config.LOG_DIR / "skipped.log"
    if not p.exists():
        return []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        when, path, reason = parts[0], parts[1], "\t".join(parts[2:])
        latest[path] = {"when": when, "path": path, "name": Path(path).name, "reason": reason,
                        "language": reason.startswith("audio is ") or reason.startswith("language could not"),
                        "n": latest.get(path, {}).get("n", 0) + 1,
                        # a subtitle in any of the default languages — English is not special (2026-10-01)
                        "has_srt": any(Path(path).with_name(f"{Path(path).stem}.{t}.srt").exists()
                                       for t in config.DEFAULT_TARGETS.split(","))}
    return sorted(latest.values(), key=lambda d: d["when"], reverse=True)


# ── app ──────────────────────────────────────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(title="mlsubgen", version=__version__, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    # HTTP Basic auth when MLSUBGEN_WEB_AUTH=user:password is set (0.5.0.10): one credential, every route including
    # the JSON API and the log stream; compared in constant time. Off when the variable is absent — the LAN-only
    # arrangement — and a reverse proxy with its own login is still the better answer for anything reachable
    # from outside; this is the minimum that keeps a curious housemate out of the job queue.
    auth = (os.environ.get("MLSUBGEN_WEB_AUTH") or "").strip()
    if auth and ":" in auth:
        import base64
        import secrets
        expected = base64.b64encode(auth.encode("utf-8")).decode("ascii")

        @app.middleware("http")
        async def basic_auth(request: Request, call_next):
            header = request.headers.get("authorization", "")
            given = header[6:].strip() if header.lower().startswith("basic ") else ""
            if given and secrets.compare_digest(given, expected):
                return await call_next(request)
            from fastapi.responses import Response
            return Response("mlsubgen: sign in", status_code=401, headers={"WWW-Authenticate": 'Basic realm="mlsubgen", charset="UTF-8"'})

    # the static files are cached by the browser under their URL: a change to app.css or app.js within one version
    # was invisible until a hard reload (2026-10-04), so their newest modification time is part of the URL
    def asset_version() -> int:
        try:
            return int(max((HERE / "static" / f).stat().st_mtime for f in ("app.css", "app.js")))
        except OSError:
            return 0

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"version": __version__, "roots": media_roots(), "asset_v": asset_version(), **ctx})

    # pages
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        defaults = config.DEFAULT_TARGETS.split(",")
        # the subtitle-language dropdown: the defaults first, then every other language by its English name
        langs = [(c, config.LANG_NAMES[c], config.NATIVE_NAMES.get(c, "")) for c in defaults if c in config.TARGET_LANGS]
        langs += sorted(((c, n, config.NATIVE_NAMES.get(c, "")) for c, n in config.TARGET_LANGS.items() if c not in defaults),
                        key=lambda x: x[1])
        profiles = [(p, f"{'≥ ' + str(int(d['min_vram_gb'])) + ' GB' if d.get('min_vram_gb') else 'small'} card; translator {d['routes'].get(('*', '*'), '?')}"
                     + (f"; withheld: {', '.join(sorted(config.PROFILE_WITHHELD[p]))}" if config.PROFILE_WITHHELD.get(p) else ""))
                    for p, d in config.PROFILES.items()]
        return render(request, "index.html", translators=list(TRANSLATORS), default_targets=defaults, langs=langs,
                      last_folder=last_folder(), profiles=profiles, active_profile=config.PROFILE,
                      vram_gb=round(config.VRAM_GB or 0, 1),
                      routes=[(f"{a}→{b}", m) for (a, b), m in config.TRANSLATE_ROUTES.items()],
                      sources=sorted(config.LANG_NAMES.items(), key=lambda kv: kv[1]),
                      default_genre="a documentary / interview programme")

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: int):
        conn = jobs.connect()
        try:
            j = jobs.get(conn, job_id)
        finally:
            conn.close()
        if j is None:
            raise HTTPException(404, "no such job")
        return render(request, "job.html", job=job_dict(j))

    @app.get("/skipped", response_class=HTMLResponse)
    def skipped_page(request: Request):
        return render(request, "skipped.html", skipped=read_skipped())

    # json
    @app.get("/api/languages")
    def api_languages(profile: str = "auto"):
        """The languages offered as targets on a profile (2026-10-04) — the form's tick boxes follow the profile select."""
        name = config.PROFILE if profile in ("", "auto") else profile
        if name not in config.PROFILES:
            raise HTTPException(400, f"unknown profile {profile!r}")
        offered = config.targets_for_profile(name)
        withheld = {c: config.WITHHELD_REASON.get(c, "measured below shippable on this profile's translators")
                    for c in config.PROFILE_WITHHELD.get(name, set())}
        return {"profile": name, "offered": sorted(offered), "withheld": withheld}

    @app.get("/api/status")
    def api_status():
        return status()

    @app.get("/api/jobs")
    def api_jobs(all: bool = False):
        conn = jobs.connect()
        try:
            return [job_dict(j) for j in jobs.list_jobs(conn, all_jobs=all)][::-1]
        finally:
            conn.close()

    @app.post("/api/jobs/purge")
    def api_purge(done: bool = False):
        conn = jobs.connect()
        try:
            n = jobs.purge(conn, statuses=jobs.FINAL if done else (jobs.FAILED, jobs.CANCELLED))
        finally:
            conn.close()
        return {"result": f"removed {n} job(s)"}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: int):
        conn = jobs.connect()
        try:
            j = jobs.get(conn, job_id)
        finally:
            conn.close()
        if j is None:
            raise HTTPException(404, "no such job")
        return job_dict(j)

    @app.get("/api/jobs/{job_id}/files")
    def api_job_files(job_id: int):
        j = api_job(job_id)
        if not j.get("log"):
            return {"files": [], "existing": 0, "summary": "", "round": ""}
        return parse_log(Path(j["log"]))

    @app.get("/api/jobs/{job_id}/log")
    def api_job_log(job_id: int, n: int = 200):
        j = api_job(job_id)
        if not j.get("log") or not Path(j["log"]).exists():
            return {"lines": []}
        return {"lines": Path(j["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]}

    @app.get("/api/jobs/{job_id}/stream")
    async def api_job_stream(job_id: int, n: int = 200):
        """Server-sent log tail. The log path is re-read every second, so a page opened while the job is still
        queued picks the log up when the job starts, and a retry's fresh log file replaces the old one."""

        def state() -> tuple[str | None, str | None]:
            conn = jobs.connect()
            try:
                j = jobs.get(conn, job_id)
            finally:
                conn.close()
            return (j.status, j.log) if j else (None, None)

        async def gen():
            path: Path | None = None
            pos = 0
            idle = 0
            told_waiting = False
            while True:
                status, log = state()
                if status is None:
                    yield "event: end\ndata: " + json.dumps("gone") + "\n\n"
                    return
                p = Path(log) if log else None
                if p != path:                         # the job started, or a retry opened a new log file
                    if path is not None and p is not None:
                        yield "event: log\ndata: " + json.dumps([f"── new log file: {p.name} ──"]) + "\n\n"
                    path, pos = p, 0
                if path and path.exists():
                    with open(path, "rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                    if chunk:
                        first = pos == 0
                        pos += len(chunk)
                        idle = 0
                        lines = chunk.decode("utf-8", "replace").splitlines()
                        yield "event: log\ndata: " + json.dumps(lines[-n:] if first else lines) + "\n\n"
                        await asyncio.sleep(0.5)
                        continue
                elif not told_waiting:
                    told_waiting = True
                    yield "event: log\ndata: " + json.dumps([f"# job {job_id} is {status} — the log appears here when it starts"]) + "\n\n"
                idle += 1
                if status in jobs.FINAL and idle >= 3:
                    yield "event: end\ndata: " + json.dumps(status) + "\n\n"
                    return
                if idle % 15 == 0:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel(job_id: int):
        conn = jobs.connect()
        try:
            return {"result": jobs.cancel(conn, job_id)}
        finally:
            conn.close()

    @app.post("/api/jobs/{job_id}/retry")
    def api_retry(job_id: int):
        conn = jobs.connect()
        try:
            return {"result": jobs.retry(conn, job_id)}
        finally:
            conn.close()

    @app.post("/api/jobs/{job_id}/pause")
    def api_pause(job_id: int):
        conn = jobs.connect()
        try:
            return {"result": jobs.pause(conn, job_id)}
        finally:
            conn.close()

    @app.post("/api/jobs/{job_id}/resume")
    def api_resume(job_id: int):
        conn = jobs.connect()
        try:
            return {"result": jobs.resume(conn, job_id)}
        finally:
            conn.close()

    @app.get("/api/settings")
    def api_settings():
        return {"targets": config.DEFAULT_TARGETS.split(","), "targets_source": config.DEFAULT_TARGETS_SOURCE,
                "profile": config.PROFILE, "settings_path": str(config.SETTINGS_PATH)}

    @app.post("/api/settings")
    async def api_settings_set(request: Request):
        """Save the default subtitle languages (the form's "make these the default"); an empty list clears the
        saved value so the environment or the built-in default applies again. The worker's jobs read the file
        when they start, so the next job uses the new default without a restart."""
        form = await request.form()
        raw = (form.get("targets") or "").strip()
        codes = [t.strip().lower() for t in raw.replace(";", ",").split(",") if t.strip()]
        unknown = [c for c in codes if c not in config.TARGET_LANGS]
        if unknown:
            raise HTTPException(400, f"not an offered subtitle language: {', '.join(unknown)}")
        val, src = config.set_default_targets(codes)
        return {"targets": val.split(","), "targets_source": src,
                "result": f"default subtitle languages: {val}" + ("" if codes else f" ({src})")}

    @app.get("/api/models")
    def api_models():
        """The Models panel: translator presets in Ollama, ASR models in the HF cache, any pull in progress."""
        from .. import models
        return models.status()

    @app.post("/api/models/pull")
    async def api_models_pull(request: Request):
        """Queue a download: a preset name, an Ollama tag, an ASR model name, 'asr' or 'defaults'."""
        from .. import models
        form = await request.form()
        name = (form.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name required")
        key, msg = models.request_pull(name)
        return {"key": key, "result": msg}

    @app.get("/api/ls")
    def api_ls(path: str = ""):
        if not path:
            return {"path": "", "parent": None, "dirs": [{"name": r, "path": r} for r in media_roots()],
                    "videos": 0, "srt": 0}
        p = safe_path(path)
        if not p.is_dir():
            raise HTTPException(404, f"{p} is not a folder")
        dirs, files, srt = [], [], 0
        targets = config.DEFAULT_TARGETS.split(",")
        try:
            entries = sorted(p.iterdir(), key=lambda x: x.name.lower())
            names = {x.name for x in entries}
            for x in entries:
                if x.name.startswith("."):
                    continue
                if x.is_dir():
                    dirs.append({"name": x.name, "path": str(x)})
                elif x.suffix.lower() in config.VIDEO_EXTS:
                    files.append({"name": x.name, "path": str(x),
                                  "have": [t for t in targets if f"{x.stem}.{t}.srt" in names]})
                elif x.suffix.lower() == ".srt":
                    srt += 1
        except PermissionError:
            raise HTTPException(403, f"cannot read {p}")
        parent = str(p.parent) if str(p) not in media_roots() else ""
        return {"path": str(p), "parent": parent, "dirs": dirs, "files": files, "videos": len(files), "srt": srt,
                "targets": targets}

    @app.post("/api/jobs")
    async def api_add(request: Request):
        form = await request.form()
        f = dict(form)
        paths = form_paths(form)
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise HTTPException(400, f"does not exist: {missing[0]}")
        opts = build_opts(f)
        conn = jobs.connect()
        try:
            jid = jobs.add(conn, [str(p) for p in paths], opts)
        finally:
            conn.close()
        if f.get("json"):
            return {"id": jid, "label": jobs.make_label([str(p) for p in paths], opts)}
        return RedirectResponse(f"/jobs/{jid}", status_code=303)

    @app.post("/api/preview")
    async def api_preview(request: Request):
        """What `mlsubgen` would do for this form, without doing it (a `--dry-run` in a subprocess)."""
        form = await request.form()
        f = dict(form)
        paths = [str(p) for p in form_paths(form)]
        opts = build_opts(f)
        cmd = [sys.executable, "-m", "mlsubgen", "run", *paths, *opts, "--dry-run", "--now"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(config.MLSUBGEN_HOME),
                               env={**os.environ, "PYTHONPATH": str(config.CODE_DIR)})
            out = (r.stderr + r.stdout).strip()
        except subprocess.TimeoutExpired:
            out = "(timed out after 120 s — a very large or slow folder; queue it anyway and watch the log)"
        return {"command": "mlsubgen " + " ".join([*paths, *opts]), "output": out}

    @app.post("/api/skipped/queue")
    async def api_skipped_queue(request: Request):
        f = await request.form()
        paths = [safe_path(x) for x in f.getlist("paths")]
        if not paths:
            raise HTTPException(400, "nothing selected")
        opts = ["--source", "ja"] if f.get("assume_ja") else []
        conn = jobs.connect()
        try:
            jid = jobs.add(conn, [str(p) for p in paths], opts)
        finally:
            conn.close()
        return RedirectResponse(f"/jobs/{jid}", status_code=303)

    return app


def serve(host: str = config.WEB_HOST, port: int = config.WEB_PORT) -> int:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0
