"""Model readiness and downloads — `mlsubgen pull` and the web UI's Models panel (2026-10-01).

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


# label → (download URL, the file it must leave under SPEAKER_MODEL_DIR). From sherpa-onnx's GitHub releases: no account.
SPEAKER_MODELS = {
    "speaker segmentation (pyannote 3.0, ONNX)": (config.SPEAKER_SEGMENTATION_URL, config.SPEAKER_SEGMENTATION_FILE),
    "speaker embedding (3D-Speaker ERes2Net)": (config.SPEAKER_EMBEDDING_URL, config.SPEAKER_EMBEDDING_FILE),
}


def _speaker_cached(rel: str) -> tuple[bool, int]:
    p = Path(config.SPEAKER_MODEL_DIR) / rel
    return (True, p.stat().st_size) if p.is_file() else (False, 0)


def _pull_url(key: str, url: str, progress) -> None:
    """Download one speaker model from GitHub into SPEAKER_MODEL_DIR; a .tar.bz2 is unpacked there."""
    import shutil
    import tarfile
    dest_dir = Path(config.SPEAKER_MODEL_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    tmp = dest_dir / (name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mlsubgen"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                progress(done, total, f"downloading {name}")
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {url}: {e}") from None
    if name.endswith(".tar.bz2"):
        with tarfile.open(tmp, "r:bz2") as tar:
            tar.extractall(dest_dir, filter="data")
        tmp.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp), str(dest_dir / name))
    (dest_dir / "NOTICE.txt").write_text(SPEAKER_NOTICE, encoding="utf-8")


SPEAKER_NOTICE = """Speaker models used by mlsubgen (--speakers), downloaded from sherpa-onnx's GitHub releases:

sherpa-onnx-pyannote-segmentation-3-0/model.onnx
  pyannote segmentation-3.0, Copyright (c) the pyannote.audio authors (Herve Bredin et al.), MIT License —
  https://huggingface.co/pyannote/segmentation-3.0 — exported to ONNX and redistributed by the sherpa-onnx
  project (Apache-2.0), https://github.com/k2-fsa/sherpa-onnx. mlsubgen uses this ONNX build, so no Hugging Face
  account, token or acceptance of that repository's access conditions is involved.

3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  3D-Speaker ERes2Net speaker embedding model, Copyright (c) Alibaba, Apache License 2.0 —
  https://github.com/modelscope/3D-Speaker — ONNX export redistributed by sherpa-onnx.

mlsubgen combines them with sherpa-onnx's clustering into its own diarization step. It does not reproduce
pyannote's full diarization pipeline.
"""


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
            elif kind == "url":
                _pull_url(key, target, lambda done, total, msg: _set(key, completed=done, total=total, message=msg))
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
    if name in ("speakers", "speaker", "diarization"):
        return [(label, "url", url) for label, (url, _) in SPEAKER_MODELS.items()]
    if name in SPEAKER_MODELS:
        return [(name, "url", SPEAKER_MODELS[name][0])]
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
    speakers = []
    for label, (u, rel) in SPEAKER_MODELS.items():
        present, size = _speaker_cached(rel)
        speakers.append({"name": label, "model": rel, "ready": present, "size": size, "pull": pulls.get(label)})
    return {"ollama": {"url": url, "reachable": tags is not None},
            "profile": config.PROFILE, "vram_gb": config.VRAM_GB,
            "routes": [{"pair": f"{a}→{b}", "preset": m} for (a, b), m in config.TRANSLATE_ROUTES.items()],
            "translators": translators, "asr": asr, "speakers": speakers,
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
            elif kind == "url":
                rel = next(r for u, r in SPEAKER_MODELS.values() if u == target)
                present, size = _speaker_cached(rel)
                if present:
                    print(f"{key:<40} ready ({size / 1e6:.0f} MB)", file=out); continue
                print(f"{key:<40} downloading from GitHub …", file=out)
                _pull_url(key, target, lambda done, total, msg: None)
                present, size = _speaker_cached(rel)
                print(f"   done ({size / 1e6:.0f} MB) → {Path(config.SPEAKER_MODEL_DIR) / rel}", file=out)
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
