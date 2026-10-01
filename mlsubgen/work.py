"""The work file: one JSON per video under ~/mlsubgen/work. ASR is the expensive stage; it is cached here so a
re-translation with another model never re-runs it. Nothing is written beside the video except the .srt."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from . import config


def work_path(video: Path, work_dir: Path = config.WORK_DIR, clip: tuple[float, float] | None = None) -> Path:
    st = video.stat()
    key = f"{video.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if clip:
        key += f"|{clip[0]:.1f}-{clip[1]:.1f}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return work_dir / f"{safe_stem(video.stem)}.{h}.json"


def safe_stem(stem: str, max_bytes: int = 150) -> str:
    """Filesystem-safe, byte-limited version of a video's stem (Japanese titles are 3 bytes a character and the
    filename limit is 255 bytes; spaces and brackets become '_')."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return safe.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def load(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def temp_beside(path: Path) -> Path:
    """A short hidden temp name in the same directory as `path` — same filesystem, so the final rename is atomic
    (NFS included), and short, so it fits however long the real name is."""
    h = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:10]
    return path.with_name(f".mlsubgen-{h}.tmp")


def atomic_write_text(path: Path, text: str) -> None:
    """Write beside the video without ever leaving a half-written file: a power cut mid-write would otherwise leave a
    truncated .srt that every later run treats as done."""
    tmp = temp_beside(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def asr_key(engine: str, model: str, context: str) -> str:
    return f"{engine}|{model}|{hashlib.sha1(context.encode('utf-8')).hexdigest()[:8]}"
