"""Model readiness and downloads — `mlsubgen pull` and the web UI's Models panel (2026-10-02).

Two kinds of model: the translator LLMs live in Ollama (pulled through its /api/pull, streamed progress), the
ASR models live in the Hugging Face cache (Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B, whisper large-v3 — downloaded
with huggingface_hub / faster-whisper's own downloader, resumable). Pulls run one at a time on a background
thread so the web process stays responsive; `status()` is what both the CLI and the page read.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .config import TRANSLATORS

# label → Hugging Face repo id. whisper large-v3 is what faster-whisper downloads for "large-v3".
ASR_MODELS = {
    "Qwen3-ASR-1.7B": config.ASR_MODEL_QWEN,
    "Qwen3-ForcedAligner-0.6B": config.ALIGNER_MODEL,
    f"whisper {config.ASR_MODEL_WHISPER}": f"Systran/faster-whisper-{config.ASR_MODEL_WHISPER}",
}


def hub_dir() -> Path:
    """Where huggingface_hub keeps its snapshots (HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")) / "hub"


def _cached(repo: str) -> tuple[bool, int]:
    """(present, bytes) for a repo in the HF cache: present when a snapshot has at least one file."""
    d = hub_dir() / ("models--" + repo.replace("/", "--"))
    snaps = d / "snapshots"
    if not snaps.is_dir() or not any(p.is_file() for s in snaps.iterdir() for p in s.rglob("*")):
        return False, 0
    size = sum(p.stat().st_size for p in (d / "blobs").glob("*") if p.is_file()) if (d / "blobs").is_dir() else 0
    return True, size


def _ollama_tags(url: str) -> dict[str, int] | None:
    """{model name: bytes} from Ollama, or None when it is unreachable."""
    try:
        req = urllib.request.Request(url.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, int] = {}
    for m in tags.get("models", []):
        name = m.get("name", "")
        out[name] = int(m.get("size") or 0)
        if name.endswith(":latest"):
            out[name[:-7]] = out[name]
    return out


# ── the pull queue ───────────────────────────────────────────────────────────────────────────────────────
_PULLS: dict[str, dict] = {}            # key → {"state": queued|running|done|error, "completed", "total", "message", "when"}
_LOCK = threading.Lock()
_QUEUE: "queue.Queue[tuple[str, str, str]]" = queue.Queue()
_THREAD: threading.Thread | None = None


def _set(key: str, **kw) -> None:
    with _LOCK:
        _PULLS.setdefault(key, {})
        _PULLS[key].update(kw, when=time.time())


def _worker() -> None:
    while True:
        key, kind, target = _QUEUE.get()
        try:
            _set(key, state="running", completed=0, total=0, message="starting")
            if kind == "ollama":
                _pull_ollama(key, target, config.LLM_URL, lambda done, total, msg: _set(key, completed=done, total=total, message=msg))
            else:
                _pull_hf(key, target, lambda msg: _set(key, message=msg))
            _set(key, state="done", message="ready")
        except Exception as e:  # noqa: BLE001
            _set(key, state="error", message=str(e)[:300])
        finally:
            _QUEUE.task_done()


def _ensure_thread() -> None:
    global _THREAD
    if _THREAD is None or not _THREAD.is_alive():
        _THREAD = threading.Thread(target=_worker, name="mlsubgen-pull", daemon=True)
        _THREAD.start()


def _pull_ollama(key: str, tag: str, url: str, progress) -> None:
    data = json.dumps({"model": tag, "stream": True}).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/pull", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                progress(int(msg.get("completed") or 0), int(msg.get("total") or 0), msg.get("status", ""))
                if msg.get("status") == "success":
                    return
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama at {url} unreachable: {e}") from None
    raise RuntimeError("the pull stream ended without 'success'")


def _pull_hf(key: str, repo: str, progress) -> None:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 — downloads are disabled (set MLSUBGEN_ONLINE=1 for `mlsubgen pull`, or HF_HUB_OFFLINE=0)")
    progress("downloading from huggingface.co (resumable)")
    if repo.startswith("Systran/faster-whisper-"):
        from faster_whisper import download_model
        download_model(repo.split("faster-whisper-", 1)[1])      # faster-whisper's own downloader: same cache, same repo
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(repo)


def request_pull(name: str, url: str | None = None) -> tuple[str, str]:
    """Queue a pull. `name` is a translator preset, an Ollama tag, an ASR label, 'asr' (all three ASR models) or
    'defaults' (ASR + the models of the default routes). Returns (key, message)."""
    _ensure_thread()
    url = url or config.LLM_URL
    queued = []
    for key, kind, target in resolve(name):
        with _LOCK:
            state = (_PULLS.get(key) or {}).get("state")
        if state in ("queued", "running"):
            continue
        _set(key, state="queued", completed=0, total=0, message="queued")
        _QUEUE.put((key, kind, target))
        queued.append(key)
    if not queued:
        return name, "already queued or running"
    return queued[0], f"queued: {', '.join(queued)}"


def resolve(name: str) -> list[tuple[str, str, str]]:
    """name → [(key, kind, target)]; kind is 'ollama' (target = tag) or 'hf' (target = repo)."""
    name = name.strip()
    if name in ("asr", "ASR"):
        return [(label, "hf", repo) for label, repo in ASR_MODELS.items()]
    if name == "defaults":
        out = resolve("asr")
        for preset in dict.fromkeys(config.TRANSLATE_ROUTES.values()):
            out += resolve(preset)
        return out
    if name in ASR_MODELS:
        return [(name, "hf", ASR_MODELS[name])]
    if name in TRANSLATORS:
        return [(name, "ollama", TRANSLATORS[name].model)]
    for label, repo in ASR_MODELS.items():
        if name == repo:
            return [(label, "hf", repo)]
    return [(name, "ollama", name)]                     # a raw Ollama tag


def status(url: str | None = None) -> dict:
    """Everything the Models panel shows: each translator preset and each ASR model with ready/size/progress."""
    url = url or config.LLM_URL
    tags = _ollama_tags(url)
    with _LOCK:
        pulls = {k: dict(v) for k, v in _PULLS.items()}
    translators = []
    for name, tr in TRANSLATORS.items():
        have = tags is not None and (tr.model in tags or tr.model + ":latest" in tags)
        translators.append({"name": name, "model": tr.model, "note": tr.note, "ready": have,
                            "size": (tags or {}).get(tr.model, 0), "pull": pulls.get(name)})
    asr = []
    for label, repo in ASR_MODELS.items():
        present, size = _cached(repo)
        asr.append({"name": label, "model": repo, "ready": present, "size": size, "pull": pulls.get(label)})
    return {"ollama": {"url": url, "reachable": tags is not None},
            "routes": [{"pair": f"{a}→{b}", "preset": m} for (a, b), m in config.TRANSLATE_ROUTES.items()],
            "translators": translators, "asr": asr,
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1", "active": any(p.get("state") in ("queued", "running") for p in pulls.values())}


def pull_now(name: str, url: str | None = None, out=None) -> int:
    """Blocking pull with progress on `out` (the CLI). Returns 0 when everything requested is ready."""
    import sys
    out = out or sys.stderr
    url = url or config.LLM_URL
    rc = 0
    for key, kind, target in resolve(name):
        try:
            if kind == "ollama":
                tags = _ollama_tags(url)
                if tags is None:
                    raise RuntimeError(f"Ollama at {url} unreachable (sudo systemctl start ollama ?)")
                if target in tags or target + ":latest" in tags:
                    print(f"{key:<26} {target:<40} ready ({tags.get(target, 0) / 1e9:.1f} GB)", file=out); continue
                print(f"{key:<26} {target:<40} pulling …", file=out)
                last = [0.0]

                def show(done, total, msg, _last=last):
                    if total and time.time() - _last[0] > 1.0:
                        print(f"\r   {done / 1e9:6.1f} / {total / 1e9:.1f} GB  {100 * done / total:5.1f}%  {msg[:40]:<40}", end="", file=out, flush=True)
                        _last[0] = time.time()
                _pull_ollama(key, target, url, show)
                print(f"\r   done {' ' * 60}", file=out)
            else:
                present, size = _cached(target)
                if present:
                    print(f"{key:<26} {target:<40} ready ({size / 1e9:.1f} GB)", file=out); continue
                print(f"{key:<26} {target:<40} downloading …", file=out)
                _pull_hf(key, target, lambda msg: None)
                present, size = _cached(target)
                print(f"   done ({size / 1e9:.1f} GB)", file=out)
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED: {e}", file=out)
            rc = 1
    return rc


def parse_pull_line(line: str) -> tuple[int, int, str, bool]:
    """One line of Ollama's streamed /api/pull → (completed, total, status, finished). Exposed for the selftest."""
    msg = json.loads(line)
    if msg.get("error"):
        raise RuntimeError(msg["error"])
    return int(msg.get("completed") or 0), int(msg.get("total") or 0), msg.get("status", ""), msg.get("status") == "success"
