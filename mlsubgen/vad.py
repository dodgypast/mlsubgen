"""Silero VAD: speech regions, language-aware chunking for the ASR, and the speech-ratio gate for hallucination
filtering. A Span may carry the language the LID assigned to it; chunks never mix languages."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .audio import SR


@dataclass
class Span:
    start: float
    end: float
    lang: str | None = None      # set by the LID; None = not decided / not applicable

    @property
    def dur(self) -> float:
        return self.end - self.start


def speech_spans(audio: np.ndarray) -> list[Span]:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(audio), model, sampling_rate=SR,
        threshold=config.VAD_THRESHOLD,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS,
        return_seconds=True,
    )
    return [Span(float(t["start"]), float(t["end"])) for t in ts]


def make_chunks(spans: list[Span], total_dur: float,
                max_len: float = config.CHUNK_MAX_SEC,
                break_silence: float = config.CHUNK_BREAK_SILENCE_SEC,
                pad: float = config.CHUNK_PAD_SEC,
                min_len: float = config.CHUNK_MIN_SEC, max_gap: float = config.CHUNK_MAX_GAP_SEC) -> list[Span]:
    """Group consecutive speech spans into ASR chunks of at most `max_len` seconds. A chunk closes on a language
    change (always), on a silence of `break_silence` once the chunk is `min_len` long (so an isolated one-second
    span joins its neighbour instead of becoming a chunk of its own — unless the gap exceeds `max_gap`, because
    the decoder must not be fed minutes of silence), and when it would exceed `max_len`.
    Each chunk carries the language of its spans."""
    chunks: list[Span] = []
    cur: Span | None = None
    for s in spans:
        # a single span longer than max_len: hard-split it
        pieces = [s]
        if s.dur > max_len:
            pieces = []
            t = s.start
            while t < s.end:
                pieces.append(Span(t, min(s.end, t + max_len), s.lang))
                t += max_len
        for p in pieces:
            if cur is None:
                cur = Span(p.start, p.end, p.lang)
                continue
            gap = p.start - cur.end
            lang_change = p.lang is not None and cur.lang is not None and p.lang != cur.lang
            if lang_change or (p.end - cur.start) > max_len or (gap >= break_silence and (cur.dur >= min_len or gap >= max_gap)):
                chunks.append(cur)
                cur = Span(p.start, p.end, p.lang)
            else:
                cur.end = p.end
    if cur is not None:
        chunks.append(cur)
    # pad the edges a little so no phoneme is cut, without overlapping neighbours
    out: list[Span] = []
    for c in chunks:
        a = max(0.0, c.start - pad)
        b = min(total_dur, c.end + pad)
        if out and a < out[-1].end:
            a = out[-1].end
        out.append(Span(a, b, c.lang))
    return out


def cover_chunks(audio: np.ndarray, spans: list[Span], total_dur: float, max_len: float = config.CHUNK_MAX_SEC,
                 quiet_db: float = config.COVER_QUIET_DB, min_quiet: float = config.COVER_MIN_QUIET_SEC,
                 dominant: str | None = None, pad: float = config.CHUNK_PAD_SEC) -> list[Span]:
    """Chunks that cover the WHOLE timeline, so the decoders hear everything: only stretches that sit at the file's
    noise floor (within `quiet_db`) for at least `min_quiet` seconds are skipped. The VAD is advisory — it misses
    dialogue over music wholesale, so it must never decide what gets decoded — but a VAD gap is a preferred cut
    point, and the language of a chunk comes from the LID-labelled spans it overlaps (else `dominant`)."""
    frame = SR // 10
    n = len(audio) // frame
    if n == 0:
        return []
    frames = audio[:n * frame].reshape(n, frame).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    floor = float(np.percentile(rms, 5))
    quiet = rms < floor * 10 ** (quiet_db / 20)
    # active stretches: runs of frames that are not part of a quiet run of at least min_quiet
    min_q = int(min_quiet * 10)
    active: list[list[float]] = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= min_q or not active:
                i = j
                continue
            active[-1][1] = j / 10.0       # a short dip: stays inside the stretch
            i = j
            continue
        j = i
        while j < n and not quiet[j]:
            j += 1
        if active and i / 10.0 - active[-1][1] < 0.05:
            active[-1][1] = j / 10.0
        else:
            active.append([i / 10.0, j / 10.0])
        i = j
    gaps = sorted((a.end, b.start) for a, b in zip(spans, spans[1:]) if b.start - a.end >= 0.3)

    def cut_point(a: float, b: float) -> float:
        """Where to cut a stretch that is longer than max_len: a VAD gap in the last third of the window if there is
        one, else the quietest frame there."""
        lo, hi = a + max_len * 0.6, a + max_len
        cand = [(g0 + g1) / 2 for g0, g1 in gaps if lo <= (g0 + g1) / 2 <= hi]
        if cand:
            return max(cand)
        f0, f1 = int(lo * 10), min(n, int(hi * 10))
        if f1 <= f0:
            return hi
        return (f0 + int(np.argmin(rms[f0:f1]))) / 10.0

    chunks: list[Span] = []
    for a, b in active:
        if b - a < 0.5:
            continue
        while b - a > max_len:
            c = cut_point(a, b)
            chunks.append(Span(a, c))
            a = c
        chunks.append(Span(a, b))
    out: list[Span] = []
    for c in chunks:
        s, e = max(0.0, c.start - pad), min(total_dur, c.end + pad)
        if out and s < out[-1].end:
            s = out[-1].end
        if e <= s:
            continue
        seconds: dict[str, float] = {}
        for sp in spans:
            if sp.lang and sp.end > s and sp.start < e:
                seconds[sp.lang] = seconds.get(sp.lang, 0.0) + min(sp.end, e) - max(sp.start, s)
        lang = max(seconds, key=seconds.get) if seconds else dominant
        out.append(Span(s, e, lang))
    return out


def speech_ratio(spans: list[Span], start: float, end: float) -> float:
    """Fraction of [start, end] that the VAD marked as speech."""
    if end <= start:
        return 0.0
    covered = 0.0
    for s in spans:
        if s.end <= start:
            continue
        if s.start >= end:
            break
        covered += min(s.end, end) - max(s.start, start)
    return max(0.0, min(1.0, covered / (end - start)))
