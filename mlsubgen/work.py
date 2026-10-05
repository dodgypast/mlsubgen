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


def condense(path: Path) -> None:
    """Shrink a finished file's work file to its provenance (0.5.13): the stages, their verdicts, the models, the
    counts — what `mlsubgen why` answers from — and drop the bulk (ASR words, the dual transcripts, the cues, the
    translated text, the evidence lines, the speaker turns). A re-run that needs the ASR again re-runs it; the
    record of what happened stays, a few KB per video."""
    data = load(path)
    if not data:
        return
    out: dict = {k: v for k, v in data.items() if k in ("video", "duration", "source", "lid", "ocr_targets", "ocr_gate", "skipped")}
    out["condensed"] = time.strftime("%Y-%m-%d %H:%M")
    if data.get("speakers"):
        s = data["speakers"]
        out["speakers"] = {k: v for k, v in s.items() if k != "turns"} | {"voices": len({t[2] for t in s.get("turns", [])}), "turn_count": len(s.get("turns", []))}
    if data.get("asr"):
        out["asr"] = {k: {kk: vv for kk, vv in e.items() if kk in ("engines", "mode", "elapsed", "merge", "merge_stats", "merge_model", "retimed")}
                      for k, e in data["asr"].items()}
    if data.get("cues"):
        out["cues"] = {k: {"stats": e.get("stats", {}), "labelled": e.get("labelled"), "cue_count": len(e.get("cues", []))} for k, e in data["cues"].items()}
    if data.get("terms"):
        out["terms"] = data["terms"]                                   # a few dozen names and their renderings
    if data.get("characters"):
        out["characters"] = data["characters"]                         # the sheet, the voices, the rules
    if data.get("translations"):
        out["translations"] = {k: {kk: vv for kk, vv in t.items() if kk != "cues"} | {"cue_count": len(t.get("cues", [])),
                               "copied": sum(1 for c in t.get("cues", []) if c.get("flags") and "copied" in c["flags"])}
                               for k, t in data["translations"].items()}
    if data.get("media_context"):
        mc = data["media_context"]
        out["media_context"] = {k: v for k, v in mc.items() if k != "results"} | {"results": [{"url": r.get("url")} for r in mc.get("results", [])]}
    if data.get("evidence_track"):
        ev = data["evidence_track"]
        out["evidence_track"] = {k: v for k, v in ev.items() if k != "lines"} | {"line_count": len(ev.get("lines", []))}
    save(path, out)


def reopen(data: dict) -> dict:
    """A condensed record opened by a new run: keep what is still a complete cache (the terms, the sheet, the media
    context, the OCR records, a remembered skip) and drop the counts-only stubs so every other stage recomputes."""
    if not data.get("condensed"):
        return data
    keep = ("video", "skipped", "ocr_targets", "ocr_gate", "media_context", "terms", "characters")
    return {k: v for k, v in data.items() if k in keep}


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
