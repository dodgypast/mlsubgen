"""Bitmap subtitle tracks → text (0.4.8, 2026-10-02).

Blu-ray remuxes carry their subtitles as PGS bitmaps (`hdmv_pgs_subtitle`), and until now mlsubgen ignored every
one of them — a film with English subtitles in it was transcribed because the planner could only read text tracks.

This module decodes a PGS stream into images with their on-screen times (no third-party PGS library: the format
is small — presentation compositions, windows, palettes, run-length-encoded objects) and hands each image to an
OCR engine. Engines live behind one function so they can be swapped and measured:
  - tesseract: the `tesseract` binary, with the language's tessdata pack (eng, jpn, tha …) — fast, the baseline
  - (later) a vision model through Ollama — slower, reads stylised fonts
The result is an .srt-shaped list of cues that the embedded-subtitle planner uses like any text track, cached
beside the work files so a track is OCR'd once.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config

# tesseract's language packs by mlsubgen code (the ones with clear names; others fall back to `eng` + a warning)
TESSERACT_LANGS = {
    "en": "eng", "ja": "jpn", "th": "tha", "ko": "kor", "zh": "chi_sim", "yue": "chi_tra", "de": "deu", "fr": "fra",
    "es": "spa", "it": "ita", "pt": "por", "nl": "nld", "ru": "rus", "uk": "ukr", "pl": "pol", "tr": "tur",
    "ar": "ara", "hi": "hin", "vi": "vie", "id": "ind", "ms": "msa", "tl": "tgl", "sv": "swe", "da": "dan",
    "no": "nor", "fi": "fin", "cs": "ces", "sk": "slk", "hu": "hun", "ro": "ron", "el": "ell", "he": "heb",
    "fa": "fas", "bn": "ben", "ta": "tam", "km": "khm", "lo": "lao", "my": "mya", "ca": "cat", "bg": "bul",
    "hr": "hrv", "sl": "slv", "lt": "lit", "lv": "lav", "et": "est",
}


@dataclass
class Bitmap:
    start: float
    end: float
    width: int
    height: int
    rgba: bytes             # width * height * 4


@dataclass
class OcrCue:
    start: float
    end: float
    text: str


# ── PGS decoding ─────────────────────────────────────────────────────────────────────────────────────────────
def _rle_decode(data: bytes, width: int, height: int) -> bytearray:
    """PGS object run-length coding → one palette index per pixel (width*height)."""
    out = bytearray(width * height)
    pos = 0
    x = y = 0
    n = len(data)
    while pos < n and y < height:
        b = data[pos]; pos += 1
        if b != 0:
            if x < width:
                out[y * width + x] = b
            x += 1
            continue
        if pos >= n:
            break
        b2 = data[pos]; pos += 1
        if b2 == 0:                               # end of line
            x = 0; y += 1
            continue
        flag = b2 & 0xC0
        if flag == 0x00:                          # 00LLLLLL: L zeros
            run, color = b2 & 0x3F, 0
        elif flag == 0x40:                        # 01LLLLLL LLLLLLLL: L zeros
            run = ((b2 & 0x3F) << 8) | data[pos]; pos += 1; color = 0
        elif flag == 0x80:                        # 10LLLLLL CCCCCCCC: L of colour C
            run = b2 & 0x3F; color = data[pos]; pos += 1
        else:                                     # 11LLLLLL LLLLLLLL CCCCCCCC
            run = ((b2 & 0x3F) << 8) | data[pos]; pos += 1; color = data[pos]; pos += 1
        if y < height:
            end = min(x + run, width)
            if color and end > x:
                base = y * width
                out[base + x:base + end] = bytes([color]) * (end - x)
        x += run
    return out


def _ycbcr_to_rgb(y: int, cb: int, cr: int) -> tuple[int, int, int]:
    c = y - 16; d = cb - 128; e = cr - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


def decode_sup(data: bytes) -> list[Bitmap]:
    """Every PGS display set that puts a subtitle up → a Bitmap with its on-screen interval. Objects are composed
    into one image per display set (two-object sets — top and bottom lines — are stacked)."""
    palettes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    objects: dict[int, dict] = {}            # id → {"w", "h", "data" (bytearray being assembled), "expect"}
    pos = 0
    n = len(data)
    out: list[Bitmap] = []
    pending: dict | None = None               # the display set being assembled: start, palette id, object refs
    while pos + 13 <= n:
        if data[pos:pos + 2] != b"PG":
            pos += 1
            continue
        pts = int.from_bytes(data[pos + 2:pos + 6], "big") / 90000.0
        seg_type = data[pos + 10]
        size = int.from_bytes(data[pos + 11:pos + 13], "big")
        payload = data[pos + 13:pos + 13 + size]
        pos += 13 + size
        if seg_type == 0x16:                                 # PCS
            if len(payload) < 11:
                continue
            n_obj = payload[10]
            palette_id = payload[9]
            if n_obj == 0:
                if pending is not None and pending.get("image") is not None and pts > pending["start"]:
                    img = pending["image"]
                    out.append(Bitmap(pending["start"], pts, img[0], img[1], img[2]))
                pending = None
                continue
            if pending is not None and pending.get("image") is not None and pts > pending["start"]:
                img = pending["image"]                   # a new subtitle replacing the old one without a clear
                out.append(Bitmap(pending["start"], pts, img[0], img[1], img[2]))
            refs = []
            p = 11
            for _ in range(n_obj):                       # composition object: id(2) window(1) cropped(1) x(2) y(2) [crop 8]
                if p + 8 > len(payload):
                    break
                obj_id = int.from_bytes(payload[p:p + 2], "big")
                cropped = payload[p + 3] & 0x80
                x = int.from_bytes(payload[p + 4:p + 6], "big")
                y = int.from_bytes(payload[p + 6:p + 8], "big")
                p += 8 + (8 if cropped else 0)
                refs.append((obj_id, x, y))
            pending = {"start": pts, "palette": palette_id, "refs": refs, "image": None}
        elif seg_type == 0x14:                               # PDS
            if len(payload) < 2:
                continue
            pid = payload[0]
            pal = palettes.setdefault(pid, {})
            p = 2
            while p + 5 <= len(payload):
                idx, y, cr, cb, a = payload[p], payload[p + 1], payload[p + 2], payload[p + 3], payload[p + 4]
                r, g, b = _ycbcr_to_rgb(y, cb, cr)
                pal[idx] = (r, g, b, a)
                p += 5
        elif seg_type == 0x15:                               # ODS
            if len(payload) < 7:
                continue
            obj_id = int.from_bytes(payload[0:2], "big")
            flags = payload[3]
            if flags & 0x80:                                 # first in sequence
                w = int.from_bytes(payload[7:9], "big"); h = int.from_bytes(payload[9:11], "big")
                objects[obj_id] = {"w": w, "h": h, "data": bytearray(payload[11:]), "done": bool(flags & 0x40)}
            elif obj_id in objects:
                objects[obj_id]["data"] += payload[4:]
                objects[obj_id]["done"] = bool(flags & 0x40)
        elif seg_type == 0x80:                               # END of display set → compose the pending image
            if pending is None or pending["image"] is not None:
                continue
            pal = palettes.get(pending["palette"], {})
            parts = []
            for obj_id, x, y in pending["refs"]:
                o = objects.get(obj_id)
                if not o or o["w"] <= 0 or o["h"] <= 0:
                    continue
                idx = _rle_decode(bytes(o["data"]), o["w"], o["h"])
                parts.append((x, y, o["w"], o["h"], idx))
            if not parts:
                continue
            # compose: stack by vertical position into one image, keeping relative x offsets (numpy: a 1080p
            # subtitle is a few hundred thousand pixels, and a track has a thousand of them)
            import numpy as np
            lut = np.zeros((256, 4), dtype=np.uint8)
            for k, c in pal.items():
                lut[k] = c
            min_x = min(p[0] for p in parts)
            max_x = max(p[0] + p[2] for p in parts)
            width = max_x - min_x
            ordered = sorted(parts, key=lambda p: p[1])
            height = sum(p[3] for p in ordered) + 8 * (len(parts) - 1)
            canvas = np.zeros((height, width, 4), dtype=np.uint8)
            cy = 0
            for x, y, w, h, idx in ordered:
                block = lut[np.frombuffer(bytes(idx), dtype=np.uint8).reshape(h, w)]
                ox = x - min_x
                canvas[cy:cy + h, ox:ox + w] = block
                cy += h + 8
            pending["image"] = (width, height, canvas.tobytes())
    return out


def extract_sup(video: Path, s_index: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as f:
        sup = Path(f.name)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video), "-map", f"0:s:{s_index}", "-c", "copy",
                        "-f", "sup", str(sup)], check=True, capture_output=True)
        return sup.read_bytes()
    finally:
        sup.unlink(missing_ok=True)


# ── images for OCR ───────────────────────────────────────────────────────────────────────────────────────────
def to_png(bm: Bitmap, scale: int = 2) -> bytes:
    """Black text on white, upscaled: what OCR engines read best. Subtitle bitmaps are light text with a dark
    outline on transparency; luminance over an opaque white background is inverted so the glyphs are dark."""
    from PIL import Image, ImageOps
    img = Image.frombytes("RGBA", (bm.width, bm.height), bm.rgba)
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    bg.alpha_composite(img)
    g = ImageOps.invert(bg.convert("L"))                 # light glyphs → dark on light
    g = ImageOps.autocontrast(g)
    g = ImageOps.expand(g, border=12 * scale, fill=255)
    if scale > 1:
        g = g.resize((g.width * scale, g.height * scale), Image.LANCZOS)
    g = g.point(lambda v: 255 if v > 140 else 0)         # binarise: the outline goes with the background
    buf = io.BytesIO()
    g.save(buf, format="PNG")
    return buf.getvalue()


# ── engines ──────────────────────────────────────────────────────────────────────────────────────────────────
def tesseract_available(lang: str) -> tuple[bool, str]:
    exe = shutil.which("tesseract")
    if not exe:
        return False, "tesseract is not installed (pacman -S tesseract tesseract-data-<lang>; apt: tesseract-ocr tesseract-ocr-<lang>)"
    pack = TESSERACT_LANGS.get(lang, "eng")
    try:
        langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=20).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        langs = []
    if pack not in langs:
        return False, f"tesseract has no '{pack}' language pack (it has: {', '.join(l for l in langs if l != 'List')}); install tesseract-data-{pack}"
    return True, pack


def ocr_tesseract(png: bytes, pack: str) -> str:
    res = subprocess.run(["tesseract", "stdin", "stdout", "-l", pack, "--psm", "6"], input=png, capture_output=True, timeout=60)
    text = res.stdout.decode("utf-8", "replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


import re as _re
_DASH_START = _re.compile(r"(?m)^-(?=\S)")
_DASH_MID = _re.compile(r"(?<=\s)-(?=[^\s\-])")
_PIPE_I = _re.compile(r"(?<![\w|])\|(?=['’]|\s+[a-z]|$)")      # "| hate", "|'m", a trailing "|" — not "Tom | Jerry"
_L_I = _re.compile(r"(?<![\w'])l(?=['’](?:m|ll|ve|d)\b)")


def clean_ocr(text: str, lang: str) -> str:
    """The systematic habits of OCR'd subtitles, measured on a Blu-ray against its own SRT (2026-10-02): dialogue
    dashes written tight (`-thanks.`) where subtitlers space them (`- thanks.`), and tesseract reading a capital I
    as a pipe (`| hate you`). English only for the l'm/l'll kind — French has l'homme."""
    text = _DASH_START.sub("- ", text)
    text = _DASH_MID.sub("- ", text)
    text = _PIPE_I.sub("I", text)
    if lang == "en":
        text = _L_I.sub("I", text)
    lines = [_re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def ocr_track(video: Path, s_index: int, lang: str, engine: str = "tesseract", progress=None,
              workers: int | None = None) -> list[OcrCue]:
    """A bitmap track → cues. `lang` picks the OCR language pack; `progress(done, total)` is called as it goes.
    tesseract is one process per image, so several run at once (`workers`, default OCR_WORKERS)."""
    from concurrent.futures import ThreadPoolExecutor
    bitmaps = [bm for bm in decode_sup(extract_sup(video, s_index)) if bm.width >= 4 and bm.height >= 4]
    if engine != "tesseract":
        raise ValueError(f"unknown OCR engine {engine!r}")
    ok, pack = tesseract_available(lang)
    if not ok:
        raise RuntimeError(pack)
    workers = workers or config.OCR_WORKERS
    texts: list[str] = [""] * len(bitmaps)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, text in enumerate(pool.map(lambda bm: ocr_tesseract(to_png(bm), pack), bitmaps)):
            texts[i] = clean_ocr(text, lang)
            done += 1
            if progress and done % 100 == 0:
                progress(done, len(bitmaps))
    return [OcrCue(bm.start, bm.end, t) for bm, t in zip(bitmaps, texts) if t]


def write_srt(cues: list[OcrCue], out: Path) -> int:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{ts(c.start)} --> {ts(c.end)}", c.text, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return len(cues)


def cached_srt_for(video: Path, s_index: int, engine: str = "tesseract") -> Path:
    """Where a track's OCR result lives once made: beside the work files, never beside the video."""
    stem = video.stem[:80]
    return config.WORK_DIR / "ocr" / f"{stem}.s{s_index}.{engine}.v{config.OCR_VERSION}.srt"


def ocr_track_cached(video: Path, s_index: int, lang: str, engine: str = "tesseract", progress=None) -> tuple[Path, int, bool]:
    """OCR a track once: (srt path, cue count, was it cached). The cache is keyed by file, track, engine and
    OCR_VERSION; the planner and the bench both go through here."""
    out = cached_srt_for(video, s_index, engine)
    if out.is_file() and out.stat().st_size > 0:
        n = sum(1 for l in out.read_text(encoding="utf-8", errors="replace").splitlines() if "-->" in l)
        return out, n, True
    cues = ocr_track(video, s_index, lang, engine, progress)
    return out, write_srt(cues, out), False
