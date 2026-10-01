"""ffmpeg extraction to 16 kHz mono PCM, and loading it as float32."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

SR = 16000


UNDECODABLE = ("Error submitting packet to decoder", "Header missing")


def extract_wav(video: Path, audio_index: int, out_wav: Path, start: float | None = None, end: float | None = None,
                expected: float | None = None) -> tuple[int, bool]:
    """ffmpeg → 16 kHz mono PCM.

    Undecodable packets (damaged MP3 frames in old AVIs, joins) are skipped by the decoder; without compensation every
    later sample lands early, so cues drift. `aresample=async=1` fills those gaps with silence from the container
    timestamps (and trims overlaps), keeping wav time == container time. If that makes the wav absurdly longer than
    `expected` (the track's duration), the timestamps themselves are broken and the plain extraction is used instead.
    Returns (undecodable packet count, gaps_filled)."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    base = ["ffmpeg", "-v", "error", "-y", "-nostdin"]
    if start is not None:
        base += ["-ss", f"{start:.3f}"]
    if end is not None:
        base += ["-to", f"{end:.3f}"]
    base += ["-i", str(video), "-map", f"0:a:{audio_index}", "-vn", "-sn", "-dn"]
    # ffmpeg writes to a .part name; the wav appears under its real name only when complete, so a reboot or kill
    # mid-extraction leaves nothing a resumed run would mistake for a finished file
    part = out_wav.with_name(out_wav.name + ".part.wav")
    tail = ["-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(part)]
    try:
        bad = _ffmpeg(video, base + ["-af", "aresample=async=1:first_pts=0"] + tail)
        filled = True
        if expected and wav_seconds(part) > expected * 1.02 + 2.0:
            bad = _ffmpeg(video, base + tail)
            filled = False
        os.replace(part, out_wav)
    finally:
        part.unlink(missing_ok=True)
    return bad, filled


def _ffmpeg(video: Path, cmd: list[str]) -> int:
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        tail = "\n".join(res.stderr.strip().splitlines()[-15:])
        raise SystemExit(f"{video.name}: ffmpeg failed (exit {res.returncode})\n{tail}")
    lines = res.stderr.splitlines()
    # ffmpeg ≥ 6 prints "Error submitting packet…" per skipped packet (plus the decoder's own "Header missing");
    # older builds print only the latter.
    return (sum(1 for l in lines if UNDECODABLE[0] in l)
            or sum(1 for l in lines if UNDECODABLE[1] in l))


def wav_seconds(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def slice_audio(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(start * SR))
    b = min(len(audio), int(end * SR))
    return audio[a:b]


def fmt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_ts(text: str) -> float:
    """'1:23:45.6' / '23:45' / '45' / '600s' → seconds."""
    t = text.strip().lower().rstrip("s")
    parts = t.split(":")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total
