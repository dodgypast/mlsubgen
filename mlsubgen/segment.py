"""Word timestamps → source-language cues (the units that get translated one-to-one)."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from . import config
from .asr import Word

SENT_END = "。！？!?…"
CLAUSE = "、,，"
PUNCT_ONLY = re.compile(r"^[\s。、！？!?…,，.]+$")
_JA_SPACE = re.compile(r"(?<=[぀-ヿ一-鿿])\s+(?=[぀-ヿ一-鿿])")  # spaces between Japanese characters
NO_SPACE_LANGS = {"ja", "zh", "yue", "th"}                       # scripts written without word spaces


@dataclass
class Cue:
    idx: int
    start: float
    end: float
    ja: str                  # the SOURCE text, whatever its language (field name kept for the work-file format)
    en: str = ""             # the TARGET text of the pass in progress (translated, or copied through)
    speech: float = 1.0      # VAD speech ratio under the cue
    flags: list[str] | None = None
    lang: str = "ja"         # the source language of this cue

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Cue":
        return Cue(int(d["idx"]), float(d["start"]), float(d["end"]), d.get("ja", ""), d.get("en", ""),
                   float(d.get("speech", 1.0)), d.get("flags"), d.get("lang", "ja"))


def join_words(parts: list[str], lang: str = "ja") -> str:
    if lang in NO_SPACE_LANGS:
        text = "".join(parts)
        text = _JA_SPACE.sub("", text)
    else:
        text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def join_ja(parts: list[str]) -> str:
    return join_words(parts, "ja")


def build_cues(words: list[Word], max_sec: float = config.CUE_MAX_SEC, max_chars: int = config.CUE_MAX_CHARS_JA,
               gap_split: float = config.CUE_GAP_SPLIT_SEC) -> list[Cue]:
    """Sentence ends, pauses, overflow and language changes all close a cue."""
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        lang = cur[0].lang
        text = join_words([w.text for w in cur], lang)
        if text and not PUNCT_ONLY.match(text):
            cues.append(Cue(len(cues), cur[0].start, cur[-1].end, text, lang=lang))
        elif text and cues:
            cues[-1].ja += text   # stray punctuation → attach to the previous cue
        cur = []

    def split_at_clause() -> None:
        """When a cue overflows mid-sentence, split at the last clause mark if both halves are worth keeping."""
        nonlocal cur
        for k in range(len(cur) - 2, 2, -1):
            if cur[k].text and cur[k].text[-1] in CLAUSE:
                head, tail = cur[:k + 1], cur[k + 1:]
                if sum(len(w.text) for w in tail) >= 4:
                    cur = head
                    flush()
                    cur = tail
                    return
        flush()

    for w in words:
        if not w.text.strip():
            continue
        if PUNCT_ONLY.match(w.text):
            if cur:
                cur[-1] = Word(cur[-1].text + w.text.strip(), cur[-1].start, max(cur[-1].end, w.end), cur[-1].lang)
                if w.text.strip()[-1] in SENT_END:
                    flush()
            elif cues:
                cues[-1].ja += w.text.strip()
            continue
        if cur:
            gap = w.start - cur[-1].end
            dur = w.end - cur[0].start
            chars = sum(len(x.text) for x in cur)
            limit = max_chars if cur[0].lang in NO_SPACE_LANGS else max_chars * 2   # Latin text is roomier per char
            if w.lang != cur[0].lang or gap >= gap_split or dur > max_sec:
                flush()
            elif chars + len(w.text) > limit:
                split_at_clause()
        cur.append(w)
        if w.text[-1] in SENT_END:
            flush()
    flush()
    for i, c in enumerate(cues):
        c.idx = i
    return cues


def normalise_timing(cues: list[Cue], min_sec: float = config.CUE_MIN_SEC, lead: float = 0.12, tail: float = 0.25) -> list[Cue]:
    """Give each cue a little lead-in and tail, enforce a minimum duration, never overlap the neighbour."""
    for i, c in enumerate(cues):
        prev_end = cues[i - 1].end if i > 0 else 0.0
        next_start = cues[i + 1].start if i + 1 < len(cues) else float("inf")
        c.start = max(prev_end + 0.05, c.start - lead)
        c.end = min(next_start - 0.05, c.end + tail)
        if c.end - c.start < min_sec:
            c.end = min(next_start - 0.05, c.start + min_sec)
        if c.end <= c.start:
            c.end = c.start + 0.5
    return cues
