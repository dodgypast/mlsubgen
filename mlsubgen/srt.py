"""Subtitle typesetting (line breaks, durations, reading speed) for any target language, and the SRT writer/reader."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import config
from .segment import Cue
from .work import atomic_write_text

_BREAKS = [" — ", "; ", ", ", " but ", " and ", " so ", " because ", " which ", " that ", " when ", " if "]


@dataclass
class SrtCue:
    start: float
    end: float
    text: str


_LEADING = set("\u0e40\u0e41\u0e42\u0e43\u0e44"      # Thai เ แ โ ใ ไ — written before the consonant they belong to
               "\u0ec0\u0ec1\u0ec2\u0ec3\u0ec4")     # Lao ເ ແ ໂ ໃ ໄ — the same convention


def _combining(ch: str) -> bool:
    """A mark that attaches to the character before it (Thai, Lao, Khmer, Burmese, Devanagari … vowels and tones)."""
    return unicodedata.category(ch) in ("Mn", "Mc")


def _safe_cut(text: str, k: int) -> int:
    """Move a cut index back until it does not split a cluster: a combining mark from its base character, or a
    leading vowel from the consonant that follows it. Script-agnostic, by Unicode category (2026-10-01; it used
    to know Thai's marks only)."""
    while 0 < k < len(text) and (_combining(text[k]) or text[k - 1] in _LEADING):
        k -= 1
    return k or len(text)


def _slices(text: str, step: int) -> list[str]:
    out, i = [], 0
    while i < len(text):
        j = len(text) if i + step >= len(text) else _safe_cut(text, i + step)
        if j <= i:
            j = min(len(text), i + step)
        out.append(text[i:j]); i = j
    return out


def wrap_lines(text: str, max_chars: int = config.SRT_MAX_LINE_CHARS, max_lines: int = config.SRT_MAX_LINES) -> list[str] | None:
    """Balanced wrap into ≤ max_lines lines of ≤ max_chars; None if it cannot fit."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    if any(len(w) > max_chars for w in words):
        if " " not in text and len(text) <= max_chars * max_lines:     # Thai / CJK: no word spaces — split evenly
            n = -(-len(text) // max_chars)
            lines = _slices(text, -(-len(text) // n))
            return lines if len(lines) <= max_lines and all(len(l) <= max_chars for l in lines) else None
        return None
    # try 2 lines with the most balanced split at a word boundary, preferring a break after punctuation
    best: list[str] | None = None
    best_score = None
    for k in range(1, len(words)):
        a = " ".join(words[:k]); b = " ".join(words[k:])
        if len(a) > max_chars or len(b) > max_chars:
            continue
        score = abs(len(a) - len(b))
        if a[-1] in ",.;:!?—" :
            score -= 6
        if a.split(" ")[-1].lower() in {"the", "a", "an", "to", "of", "and", "or", "in", "on", "at", "for", "with"}:
            score += 8
        if best_score is None or score < best_score:
            best, best_score = [a, b], score
    if best and max_lines >= 2:
        return best
    return None


def split_text(text: str) -> tuple[str, str] | None:
    """Split an over-long translation into two cue texts at the best natural break near the middle."""
    n = len(text)
    if n < 20:
        return None
    mid = n / 2
    best = None
    for br in _BREAKS:
        for m in re.finditer(re.escape(br), text):
            pos = m.end()
            d = abs(pos - mid)
            if best is None or d < best[0]:
                best = (d, pos, br)
    if best is None or best[0] > n * 0.35:
        # fall back to the nearest space to the middle
        spaces = [m.start() for m in re.finditer(" ", text)]
        if not spaces:
            return None
        pos = min(spaces, key=lambda p: abs(p - mid)) + 1
        return text[:pos].strip(), text[pos:].strip()
    pos = best[1]
    a, b = text[:pos].strip(), text[pos:].strip()
    if a.endswith(("—",)):
        a = a[:-1].strip()
    return (a, b) if a and b else None


def fit_parts(text: str, depth: int = 0, max_chars: int = config.SRT_MAX_LINE_CHARS) -> list[str] | None:
    """Split a translation into pieces that each fit two lines; None when no natural split exists."""
    if wrap_lines(text, max_chars) is not None:
        return [text]
    if depth >= 3:
        return None
    parts = split_text(text)
    if not parts:
        return None
    left = fit_parts(parts[0], depth + 1, max_chars)
    right = fit_parts(parts[1], depth + 1, max_chars)
    if left is None or right is None:
        return None
    return left + right


def greedy_lines(text: str, max_chars: int = config.SRT_MAX_LINE_CHARS) -> list[str]:
    if " " not in text:                                                 # no word spaces: fixed-width, cluster-safe slices
        return _slices(text, max_chars)
    lines, cur = [], ""
    for w in text.split(" "):
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def typeset(cues: list[Cue], min_dur: float = config.SRT_MIN_DUR, max_dur: float = config.SRT_MAX_DUR,
            gap: float = config.SRT_MIN_GAP, max_cps: float | None = None, lang: str = "en") -> list[SrtCue]:
    max_chars = config.SRT_MAX_LINE_CHARS_BY_LANG.get(lang, config.SRT_MAX_LINE_CHARS)
    if max_cps is None:
        max_cps = config.SRT_MAX_CPS_BY_LANG.get(lang, config.SRT_MAX_CPS)
    # 1. text → one cue, or several when it cannot be wrapped into two lines
    items: list[SrtCue] = []
    for c in cues:
        text = (c.en or "").strip()
        if not text:
            continue
        parts = fit_parts(text, max_chars=max_chars)
        if parts is None:
            # last resort: more than two lines rather than losing text
            items.append(SrtCue(c.start, c.end, "\n".join(greedy_lines(text, max_chars))))
            continue
        if len(parts) == 1:
            items.append(SrtCue(c.start, c.end, "\n".join(wrap_lines(parts[0], max_chars) or [parts[0]])))
            continue
        total = sum(len(p) for p in parts)
        span = c.end - c.start - gap * (len(parts) - 1)
        t = c.start
        for p in parts:
            d = max(0.3, span * len(p) / total)
            items.append(SrtCue(t, t + d, "\n".join(wrap_lines(p, max_chars) or [p])))
            t += d + gap

    # 2. timing rules, in order
    for i, s in enumerate(items):
        nxt = items[i + 1].start if i + 1 < len(items) else float("inf")
        s.start = max(0.0, s.start)
        # minimum duration and reading speed both may extend the end into the following gap
        need = max(min_dur, len(s.text.replace("\n", " ")) / max_cps)
        if s.end - s.start < need:
            s.end = min(nxt - gap, s.start + need)
        if s.end - s.start > max_dur:
            s.end = s.start + max_dur
        if s.end <= s.start:
            s.end = s.start + 0.5
    for i in range(1, len(items)):
        if items[i].start < items[i - 1].end + gap:
            items[i].start = items[i - 1].end + gap
            if items[i].end <= items[i].start:
                items[i].end = items[i].start + min_dur
    return items


def ts(sec: float) -> str:
    sec = max(0.0, sec)
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, cues: list[SrtCue]) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{ts(c.start)} --> {ts(c.end)}", c.text, ""]
    atomic_write_text(path, "\n".join(lines) + "\n")


_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def read_srt(path: Path) -> list[SrtCue]:
    out: list[SrtCue] = []
    block: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines() + [""]:
        if line.strip():
            block.append(line.rstrip("\r"))
            continue
        if block:
            times = next((l for l in block if "-->" in l), None)
            if times:
                m = _TS.search(times)
                if m:
                    g = [int(x) for x in m.groups()]
                    a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                    b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                    txt = "\n".join(l for l in block[block.index(times) + 1:])
                    out.append(SrtCue(a, b, re.sub(r"<[^>]+>", "", txt)))
            block = []
    return out
