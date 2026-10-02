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
        need = 0 if flag == 0x00 else (1 if flag in (0x40, 0x80) else 2)
        if pos + need > n:                        # a truncated fragment (2026-10-02: a Blu-ray's Chinese track): keep what decoded
            break
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
PREP_MODES = ("binary", "fill", "gray", "fill3x")


def to_png(bm: Bitmap, scale: int = 2, mode: str | None = None) -> bytes:
    """Black text on white, upscaled: what OCR engines read best. Subtitle bitmaps are light text with a dark
    outline on transparency. Modes (OCR_PREP; measured on a Thai track 2026-10-02 — see config):
      binary  luminance over black, inverted, binarised at 140 — the outline mostly goes with the background
      fill    the glyph FILL only: opaque pixels that are light; the outline is dropped by construction, and thin
              marks above the line (Thai tone marks) survive because nothing is thresholded away
      gray    the inverted luminance, no binarisation: tesseract thresholds itself
      fill3x  fill at 3× instead of 2×
    """
    import numpy as np
    from PIL import Image, ImageOps
    mode = mode or config.OCR_PREP
    if mode == "fill3x":
        mode, scale = "fill", 3
    img = Image.frombytes("RGBA", (bm.width, bm.height), bm.rgba)
    if mode == "fill":
        a = np.frombuffer(bm.rgba, dtype=np.uint8).reshape(bm.height, bm.width, 4)
        lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        ink = (a[..., 3] > 64) & (lum > 110)              # opaque and light = the fill; the dark outline is left out
        g = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8), "L")
    else:
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        bg.alpha_composite(img)
        g = ImageOps.invert(bg.convert("L"))             # light glyphs → dark on light
        g = ImageOps.autocontrast(g)
    g = ImageOps.expand(g, border=12 * scale, fill=255)
    if scale > 1:
        g = g.resize((g.width * scale, g.height * scale), Image.LANCZOS)
    if mode == "binary":
        g = g.point(lambda v: 255 if v > 140 else 0)     # binarise: the outline goes with the background
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


def ocr_tesseract(png: bytes, pack: str, psm: int = 6) -> str:
    res = subprocess.run(["tesseract", "stdin", "stdout", "-l", pack, "--psm", str(psm)], input=png, capture_output=True, timeout=60)
    text = res.stdout.decode("utf-8", "replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


# Scripts that stack marks above and below the baseline. tesseract's block mode (psm 6) saw a Thai subtitle's row
# of tone marks and upper vowels as a text line of its own and read it as garbage, then read the base line without
# its marks (2026-10-02: "๓% = = %7%" before "ฉันคิดถึงไอ้บ้านัน"). For these, the subtitle is cut into its text
# lines by the ink profile — a line keeps the marks near it — and each line is read alone (psm 7) at 4×.
STACKED_SCRIPTS = {"th", "lo", "km", "my", "hi", "bn", "ta", "vi", "ar", "fa", "he"}


def split_text_lines(gray, min_gap_ratio: float = 0.45):
    """Cut a prepared (dark-on-light) image into its text lines. Rows with ink form runs; runs closer than
    `min_gap_ratio` × the tallest run's height are the same line (marks and their base); larger gaps separate
    lines. Returns a list of PIL images, top to bottom."""
    import numpy as np
    from PIL import Image
    a = np.asarray(gray.convert("L"))
    ink = (a < 160).sum(axis=1) > 0
    runs: list[list[int]] = []
    for r, on in enumerate(ink):
        if on:
            if runs and runs[-1][1] == r - 1:
                runs[-1][1] = r
            else:
                runs.append([r, r])
    if not runs:
        return [gray]
    tallest = max(e - s + 1 for s, e in runs)
    merged: list[list[int]] = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= tallest * min_gap_ratio:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    pad = max(6, tallest // 4)
    out = []
    for s, e in merged:
        if e - s + 1 < 3:
            continue
        box = (0, max(0, s - pad), gray.width, min(gray.height, e + pad + 1))
        out.append(gray.crop(box))
    return out or [gray]


def ocr_tesseract_lines(bm: "Bitmap", pack: str, lang: str, prep: str | None = None) -> str:
    """The stacked-script path: prepare at 4×, split into text lines, read each alone."""
    from PIL import Image
    png = to_png(bm, scale=4, mode=prep or ("gray" if config.OCR_PREP == "binary" else config.OCR_PREP))
    gray = Image.open(io.BytesIO(png)).convert("L")
    texts = []
    for line in split_text_lines(gray):
        buf = io.BytesIO()
        line.save(buf, format="PNG")
        t = ocr_tesseract(buf.getvalue(), pack, psm=7).replace("\n", " ").strip()
        if t:
            texts.append(t)
    return "\n".join(texts)


import re as _re
_DASH_START = _re.compile(r"(?m)^[-_](?=\S)")                    # a tight dialogue dash, or one read as an underscore
_DASH_MID = _re.compile(r"(?<=\s)[-_](?=[^\s\-_])")
_PIPE_I = _re.compile(r"(?<![\w|])\|(?=['’]|\s+[a-z]|$)")      # "| hate", "|'m", a trailing "|" — not "Tom | Jerry"
_L_I = _re.compile(r"(?<![\w'])l(?=['’](?:m|ll|ve|d)\b|\s+[a-z])")   # l'm, l'll …, and a bare "l knew": never an English word
_ITALIC_I = _re.compile(r"\biI(?=[a-z])")                        # an italic capital I read twice: "iIn Nazi-occupied France"


def clean_ocr(text: str, lang: str) -> str:
    """The systematic habits of OCR'd subtitles, measured on a Blu-ray against its own SRT (2026-10-02): dialogue
    dashes written tight (`-thanks.`) where subtitlers space them (`- thanks.`), and tesseract reading a capital I
    as a pipe (`| hate you`). English only for the l'm/l'll kind — French has l'homme. Thai: tesseract writes
    sara am as nikhahit + sara aa (two code points); the one-character form is what every text uses."""
    if lang == "th":
        text = text.replace("\u0e4d\u0e32", "\u0e33")
    text = _re.sub(r"\\?[cl]dots", "…", text)             # a vision model writing the ellipsis as LaTeX ("什么cdots那是")
    text = _DASH_START.sub("- ", text)
    text = _DASH_MID.sub("- ", text)
    text = _PIPE_I.sub("I", text)
    text = _ITALIC_I.sub("I", text)
    if lang == "en":
        text = _L_I.sub("I", text)
    lines = [_re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


VLM_PROMPT = ("This image is one subtitle from a film, in {language}. Return only the visible subtitle text, exactly as "
              "written: every character, line breaks as line breaks. Do not correct grammar or spelling, do not "
              "paraphrase, do not infer missing words, do not translate, do not add punctuation that is not visible, "
              "do not explain or describe, no quotation marks around it. If the image holds no text, answer with an "
              "empty line.{extra}")
VLM_EXTRA = {"ja": " Ignore any small furigana (ruby readings) printed above the kanji: transcribe the main text only.",
             "zh": " Keep the characters in the script shown (simplified or traditional); do not convert them.",
             "yue": " Keep the characters in the script shown (simplified or traditional); do not convert them."}


def ocr_vlm(png: bytes, lang: str, model: str | None = None, url: str | None = None) -> str:
    """A vision model through Ollama reads the subtitle image (0.4.8). The model is config.OCR_VLM_MODEL (a model
    with the `vision` capability — gemma4 and qwen3.8 both have it); temperature 0."""
    import base64
    import json
    import urllib.request
    model = model or config.OCR_VLM_MODEL
    url = (url or config.LLM_URL).rstrip("/") + "/api/generate"
    # think: false — Gemma 4 otherwise spends its tokens reasoning and returns an empty response (2026-10-02: 138 of
    # 150 Thai images came back empty at eight seconds each; the translator presets switch thinking off the same way)
    body = {"model": model, "prompt": VLM_PROMPT.format(language=config.LANG_NAMES.get(lang, lang), extra=VLM_EXTRA.get(lang, "")),
            "images": [base64.b64encode(png).decode("ascii")], "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 200}, "keep_alive": "10m"}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.loads(resp.read().decode("utf-8")).get("response", "")
    text = text.strip().strip("\"'“”")
    if text.lower() in ("(empty)", "empty", "none", "no text", "[no text]"):
        return ""
    return "\n".join(l.strip() for l in text.splitlines() if l.strip())


def vlm_unload(model: str | None = None, url: str | None = None) -> None:
    """Take the vision model off the card. The OCR stage runs right before the ASR engines load, and the 31B's
    19 GB left resident made faster-whisper fail with CUDA out of memory (2026-10-02)."""
    import json
    import urllib.request
    body = {"model": model or config.OCR_VLM_MODEL, "keep_alive": 0}
    req = urllib.request.Request((url or config.LLM_URL).rstrip("/") + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:                                                # noqa: BLE001 — best effort
        pass


def vlm_png(bm: Bitmap) -> bytes:
    """What a vision model sees: the subtitle as it is on screen (composited on a dark background, 2×) — a VLM
    reads styled text better than a binarised one."""
    from PIL import Image, ImageOps
    img = Image.frombytes("RGBA", (bm.width, bm.height), bm.rgba)
    bg = Image.new("RGBA", img.size, (24, 24, 24, 255))
    bg.alpha_composite(img)
    g = ImageOps.expand(bg.convert("RGB"), border=16, fill=(24, 24, 24))
    g = g.resize((g.width * 2, g.height * 2), Image.LANCZOS)
    buf = io.BytesIO()
    g.save(buf, format="PNG")
    return buf.getvalue()


# Which engine reads which script — measured 2026-10-02 on No Hard Feelings (web release, text tracks from the same
# source as the bitmaps; exact-match share / chrF on 150–300 cues) and Definitely, Maybe (Blu-ray, 1,800 cues):
#   Latin (English)         tesseract  95 % / 99.4          gemma4 31B not needed
#   Thai                    tesseract  50 % / 71 (line by line; 13 % as a block)
#                           gemma4:31b 77 % / 93.7   gemma4:12b 65 % / 89.3   gemma4:e4b 18 % / 66.5
#   Chinese (Simplified)    tesseract  41 % / 77.6          gemma4:31b 79 % / 90.0
#   Chinese (Traditional)   tesseract  37 % / 64.5
# So: Latin-script (and Greek/Cyrillic, alphabets tesseract handles like Latin) → tesseract; CJK and the scripts
# with stacked marks → the profile's vision model, except the 8gb profile's E4B, which does not read them well
# enough to translate from — those tracks are left alone there and the audio is transcribed, as before 0.4.8.
VLM_SCRIPTS = {"ja", "zh", "yue", "ko", "th", "lo", "km", "my", "hi", "bn", "ta", "ar", "fa", "he"}


def engine_for(lang: str) -> str | None:
    """The engine that reads bitmap subtitles in `lang` on this machine, or None when none can."""
    if lang in VLM_SCRIPTS:
        if config.PROFILE == "8gb":
            return None
        return "vlm" if engine_available("vlm", lang)[0] else None
    return "tesseract" if tesseract_available(lang)[0] else None


def engine_available(engine: str, lang: str) -> tuple[bool, str]:
    if engine == "tesseract":
        return tesseract_available(lang)
    if engine == "vlm":
        try:
            import urllib.request
            with urllib.request.urlopen(config.LLM_URL.rstrip("/") + "/api/tags", timeout=5) as r:
                names = {m.get("name") for m in __import__("json").loads(r.read().decode()).get("models", [])}
        except Exception as e:                                      # noqa: BLE001
            return False, f"Ollama not reachable at {config.LLM_URL}: {e}"
        if config.OCR_VLM_MODEL not in names:
            return False, f"vision model {config.OCR_VLM_MODEL} is not in Ollama (ollama pull it, or set OCR_VLM_MODEL)"
        return True, config.OCR_VLM_MODEL
    return False, f"unknown OCR engine {engine!r}"


def ocr_track(video: Path, s_index: int, lang: str, engine: str = "tesseract", progress=None,
              workers: int | None = None, prep: str | None = None) -> list[OcrCue]:
    """A bitmap track → cues. `lang` picks the OCR language pack; `progress(done, total)` is called as it goes.
    tesseract is one process per image, so several run at once (`workers`, default OCR_WORKERS); the vision model
    is asked one image at a time (the GPU is the bottleneck, not the request)."""
    bitmaps = [bm for bm in decode_sup(extract_sup(video, s_index)) if bm.width >= 4 and bm.height >= 4]
    return ocr_track_images(bitmaps, lang, engine, progress, workers, prep)


def ocr_track_images(bitmaps: list[Bitmap], lang: str, engine: str = "tesseract", progress=None,
                     workers: int | None = None, prep: str | None = None) -> list[OcrCue]:
    """The OCR of already-decoded images (ocr_track does the decoding; a bench may read only the first N)."""
    from concurrent.futures import ThreadPoolExecutor
    bitmaps = [bm for bm in bitmaps if bm.width >= 4 and bm.height >= 4]
    ok, why = engine_available(engine, lang)
    if not ok:
        raise RuntimeError(why)
    texts: list[str] = [""] * len(bitmaps)
    done = 0
    if engine == "tesseract":
        pack = why
        workers = workers or config.OCR_WORKERS
        if lang in STACKED_SCRIPTS:
            read = lambda bm: ocr_tesseract_lines(bm, pack, lang, prep)          # noqa: E731
        else:
            read = lambda bm: ocr_tesseract(to_png(bm, mode=prep), pack)        # noqa: E731
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, text in enumerate(pool.map(read, bitmaps)):
                texts[i] = clean_ocr(text, lang)
                done += 1
                if progress and done % 100 == 0:
                    progress(done, len(bitmaps))
    else:
        for i, bm in enumerate(bitmaps):
            try:
                texts[i] = clean_ocr(ocr_vlm(vlm_png(bm), lang), lang)
            except Exception as e:                                  # noqa: BLE001 — one bad image must not sink the track
                texts[i] = ""
                if i < 3:
                    print(f"[ocr] vlm failed on image {i}: {e}", flush=True)
            done += 1
            if progress and done % 50 == 0:
                progress(done, len(bitmaps))
        vlm_unload()                                                  # the card is needed next by the ASR engines
    return [OcrCue(bm.start, bm.end, t) for bm, t in zip(bitmaps, texts) if t]


# ── the OCR gate (0.5.0.4) ───────────────────────────────────────────────────────────────────────────────────
# An OCR'd track is used only when its text looks like subtitles in the expected language. The two engines fail
# differently — tesseract produces symbol salad that still contains letters of the right script, a vision model
# produces plausible text that may not be on the screen, or prose about the image — and both leave marks that a
# transcript does not: symbols where letters should be, cues in another script, one line repeated across many
# cues, replacement characters, sentences that describe the image. A track that fails is left alone and the
# audio is transcribed, as before 0.4.8.
_SCRIPT_FOR = {"ja": {"ja", "han"}, "zh": {"han"}, "yue": {"han"}, "ko": {"ko"}, "th": {"th"}, "el": {"greek"},
               "ru": {"cyrillic"}, "uk": {"cyrillic"}, "bg": {"cyrillic"}}
_PROSE = _re.compile(r"^\s*(?:the (?:text|image|subtitle|caption)|this (?:image|subtitle)|here is|here's|i can see|it (?:says|reads)|"
                     r"the words|there is no text|no text)", _re.I)
_COMMON_PUNCT = set(" \t\n.,;:!?'\"-–—…()[]「」『』、。！？・“”‘’«»¿¡/&+♪" + "\u3000")      # = % | # etc. are symbol salad


def assess(cues: list[OcrCue], lang: str) -> tuple[bool, dict, str]:
    """(usable, measurements, reason). Thresholds in config.OCR_GATE_*; the measurements are kept with the
    track's record so a rejection can be read later."""
    from collections import Counter
    from .lid import script_of
    texts = [c.text for c in cues if c.text.strip()]
    n = len(texts)
    if n < 10:
        return False, {"cues": n}, f"only {n} cues"
    expected = _SCRIPT_FOR.get(lang, {"latin"})
    scripts = [script_of(t) for t in texts]
    judged = [s for s in scripts if s]                       # a cue of only digits or symbols has no script
    script_match = sum(1 for s in judged if s in expected) / max(1, len(judged))
    chars = "".join(texts)
    symbols = sum(1 for ch in chars if not (ch.isalnum() or ch.isspace() or ch in _COMMON_PUNCT or
                                            0x300 <= ord(ch) <= 0x36F or 0xE31 <= ord(ch) <= 0xE4E or 0x3099 <= ord(ch) <= 0x309A))
    symbol_share = symbols / max(1, len(chars))
    replacement = chars.count("\ufffd") / max(1, len(chars))
    repeat = max(Counter(texts).values()) / n
    prose = sum(1 for t in texts if _PROSE.match(t)) / n
    # junk lines (0.5.0.5): tesseract's Thai salad passed the checks above — its noise is Thai digits and letters, in
    # script and alphanumeric — but it arrives as extra lines that are mostly digits and symbols ("4ส4๐ '" beside a
    # real line), and dialogue almost never has a line like that
    def junk_line(line: str) -> bool:
        body = [ch for ch in line if not ch.isspace()]
        if len(body) < 2:
            return False
        bad = sum(1 for ch in body if ch.isdigit() or not (ch.isalnum() or ch in _COMMON_PUNCT or
                                                              0x300 <= ord(ch) <= 0x36F or 0xE31 <= ord(ch) <= 0xE4E or 0x3099 <= ord(ch) <= 0x309A))
        return bad / len(body) >= 0.4
    junk = sum(1 for t in texts if any(junk_line(l) for l in t.splitlines())) / n
    m = {"cues": n, "script_match": round(script_match, 3), "symbol_share": round(symbol_share, 3),
         "replacement": round(replacement, 4), "repeat": round(repeat, 3), "prose": round(prose, 3), "junk_lines": round(junk, 3)}
    if script_match < config.OCR_GATE_SCRIPT:
        return False, m, f"{100 * (1 - script_match):.0f}% of cues are not in the {config.LANG_NAMES.get(lang, lang)} script"
    if symbol_share > config.OCR_GATE_SYMBOLS:
        return False, m, f"{100 * symbol_share:.0f}% of characters are symbols, not letters"
    if replacement > config.OCR_GATE_REPLACEMENT:
        return False, m, "replacement characters in the text"
    if repeat > config.OCR_GATE_REPEAT:
        return False, m, f"one line repeated across {100 * repeat:.0f}% of cues"
    if prose > config.OCR_GATE_PROSE:
        return False, m, f"{100 * prose:.0f}% of cues describe the image instead of transcribing it"
    if junk > config.OCR_GATE_JUNK:
        return False, m, f"{100 * junk:.0f}% of cues carry a line of digits and symbols"
    return True, m, "ok"


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


def ocr_track_cached(video: Path, s_index: int, lang: str, engine: str | None = None, progress=None) -> tuple[Path, int, bool]:
    """OCR a track once: (srt path, cue count, was it cached). The cache is keyed by file, track, engine and
    OCR_VERSION; the planner and the bench both go through here. `engine` None = the one measured best for the
    script (engine_for)."""
    engine = engine or engine_for(lang)
    if engine is None:
        raise RuntimeError(f"no OCR engine can read {config.LANG_NAMES.get(lang, lang)} bitmap subtitles here")
    out = cached_srt_for(video, s_index, engine)
    if out.is_file() and out.stat().st_size > 0:
        n = sum(1 for l in out.read_text(encoding="utf-8", errors="replace").splitlines() if "-->" in l)
        return out, n, True
    cues = ocr_track(video, s_index, lang, engine, progress)
    return out, write_srt(cues, out), False
