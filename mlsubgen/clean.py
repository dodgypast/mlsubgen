"""Hallucination and loop filters. A cue the VAD saw speech under is trusted. A cue the VAD did NOT see speech
under is suspect — it is either a quiet, short line the VAD missed (keep) or the decoder inventing text over
music or silence (drop) — and is judged by what it is: filler-only, blacklisted, impossibly fast, too long for
its evidence, or without sound under it in the audio."""
from __future__ import annotations

import re

import numpy as np

from . import config
from .audio import SR
from .segment import Cue
from .vad import Span, speech_ratio

_LOOP = re.compile(r"(.{1,12}?)\1{3,}")       # the same 1–12 chars repeated 4+ times in a row
_PATTERNS = [re.compile(p) for p in config.HALLUCINATION_PATTERNS]
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)
_HAN = re.compile(r"[一-鿿]")
# vocalisation, not words: the kana of moans and grunts (not い/う/え/お — those make はい, いい, うん, ええ),
# small kana, prolongation marks, CJK interjections, Latin grunts
_FILLER_CHARS = set("あぁぃぅぇぉっんーはふむ嗯啊哦呃唔哈呀嘿哼嘛哎唉噢喔ー～〜…")
_FILLER_WORDS = {"hmm", "mmm", "mm", "ah", "ahh", "aah", "oh", "ooh", "uh", "um", "huh", "ha", "haha"}


def collapse_loops(text: str) -> tuple[str, bool]:
    new = _LOOP.sub(lambda m: m.group(1) * 2, text)
    return new, new != text


def wordlike(text: str) -> bool:
    """Two or more characters with at least one that is not a moan/grunt: 痛い, はい, スカートだけ yes; ああ, んんっ,
    a lone し no."""
    if re.search(r"[A-Za-zÀ-ɏ]", text):
        return any(w.lower() not in _FILLER_WORDS for w in re.findall(r"[A-Za-zÀ-ɏ]{2,}", text))
    stripped = _PUNCT.sub("", text)
    if len(stripped) == 1:
        return bool(_HAN.match(stripped))              # 何, 誰, 奥 are words; a lone が, て, し is a fragment
    if len(re.sub(r"(.)\1+", r"\1", stripped)) < len(stripped) * 0.5:
        return False                                   # mostly repeated characters: ああああうううう
    core = [ch for ch in stripped if ch not in _FILLER_CHARS]
    return len(stripped) >= 2 and len(core) >= 1


class Loudness:
    """Energy of the audio under a cue, in dB above the file's noise floor (the 5th percentile of 100 ms frames)."""

    def __init__(self, audio: np.ndarray):
        frame = SR // 10
        n = len(audio) // frame
        frames = audio[:n * frame].reshape(n, frame).astype(np.float32)
        self.rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        self.floor = float(np.percentile(self.rms, 5)) if n else 1e-6
        self.frame = frame

    def db_above_floor(self, start: float, end: float) -> float:
        a, b = int(start * SR / self.frame), max(int(start * SR / self.frame) + 1, int(end * SR / self.frame))
        seg = self.rms[a:b]
        if len(seg) == 0:
            return 0.0
        return float(20 * np.log10(max(float(np.percentile(seg, 80)), 1e-9) / max(self.floor, 1e-9)))


def filter_cues(cues: list[Cue], spans: list[Span], audio: np.ndarray | None = None) -> tuple[list[Cue], dict]:
    kept: list[Cue] = []
    stats = {"dropped_no_speech": 0, "kept_quiet": 0, "dropped_blacklist": 0, "loops_collapsed": 0, "dropped_repeat": 0}
    loud = Loudness(audio) if audio is not None and len(audio) else None
    prev_text = None
    repeat_run = 0
    for c in cues:
        c.speech = round(speech_ratio(spans, c.start, c.end), 3)
        text, looped = collapse_loops(c.ja)
        if looped:
            stats["loops_collapsed"] += 1
            c.ja = text
            c.flags = (c.flags or []) + ["loop"]
        if c.speech < config.MIN_SPEECH_RATIO:
            dur = max(c.end - c.start, 0.01)
            nchars = len(_PUNCT.sub("", c.ja))
            too_fast = nchars > 8 and nchars / dur > config.QUIET_MAX_CPS      # a paragraph in a second; a 50 ms うん is timing, not fraud
            db = loud.db_above_floor(c.start, max(c.end, c.start + 0.3)) if loud else None
            speechlike = (wordlike(c.ja) and not any(p.search(c.ja) for p in _PATTERNS)
                          and not too_fast and dur <= config.QUIET_MAX_SEC
                          and (db is None or db >= config.QUIET_MIN_DB))
            if not speechlike:
                stats["dropped_no_speech"] += 1
                continue
            stats["kept_quiet"] += 1
            c.flags = (c.flags or []) + ["quiet"]
        elif c.speech < config.BLACKLIST_SPEECH_RATIO and any(p.search(c.ja) for p in _PATTERNS):
            stats["dropped_blacklist"] += 1
            continue
        if prev_text is not None and c.ja == prev_text:
            repeat_run += 1
            if repeat_run >= 2:          # third identical cue in a row: the decoder is stuck
                stats["dropped_repeat"] += 1
                continue
        else:
            repeat_run = 0
        prev_text = c.ja
        kept.append(c)
    for i, c in enumerate(kept):
        c.idx = i
    return kept, stats
