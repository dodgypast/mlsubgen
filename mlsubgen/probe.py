"""ffprobe: find the Japanese audio track and the duration."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

JA_TAGS = {"jpn", "ja", "jp", "japanese"}
EN_TAGS = {"eng", "en", "english"}
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "ttml"}   # convertible to .srt


@dataclass
class AudioTrack:
    index: int          # index among audio streams (0-based) — what `-map 0:a:N` takes
    stream_index: int   # absolute stream index in the container
    language: str
    title: str
    codec: str
    channels: int
    sample_rate: int
    is_default: bool
    duration: float = 0.0     # the stream's own duration (0 if the container does not say)
    start_time: float = 0.0


@dataclass
class SubTrack:
    index: int          # index among subtitle streams — what `-map 0:s:N` takes
    stream_index: int
    language: str
    title: str
    codec: str
    is_text: bool       # text (convertible to srt) vs bitmap (pgs/vobsup — unusable without OCR)
    forced: bool
    is_default: bool
    sniffed: bool = False   # the language came from the track's own text, not from a tag (0.4.7)


_SNIFF_CACHE: dict[tuple[str, float, int], str | None] = {}


def language_of_text(text: str, min_hits: int = 10) -> str | None:
    """The language of a subtitle track from its words: script first (kana, Thai, Hangul, Han, Greek), then the
    function words for Latin-script languages. None when the text does not say — too little, or a Latin-script
    text with no clear winner. Cyrillic is left undecided (Russian, Ukrainian, Bulgarian share it)."""
    from .lid import script_of, text_language
    body = " ".join(l for l in text.splitlines() if l.strip() and not l.strip().isdigit() and "-->" not in l)
    if len(body) < 200:
        return None
    s = script_of(body)
    if s in ("ja", "th", "ko"):
        return s
    if s == "han":
        return "zh"
    if s == "greek":
        return "el"
    if s == "latin":
        lang, n = text_language(body)
        return lang if lang and n >= min_hits else None
    return None


def sniff_subtitle_language(path: Path, track: SubTrack, seconds: int = 1200) -> str | None:
    """What language an untagged text track is in, read from its first `seconds` of cues (2026-10-02: an ASS track
    tagged `und` with no title was invisible to the planner, and a run would have transcribed an episode that
    already had English subtitles). Reading only the opening minutes keeps it cheap on a large file."""
    key = (str(path), path.stat().st_mtime, track.index)
    if key in _SNIFF_CACHE:
        return _SNIFF_CACHE[key]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-t", str(seconds), "-i", str(path), "-map", f"0:s:{track.index}",
           "-c:s", "srt", "-f", "srt", "pipe:1"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=120)
        code = language_of_text(res.stdout) if res.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        code = None
    _SNIFF_CACHE[key] = code
    return code


@dataclass
class ProbeResult:
    duration: float
    tracks: list[AudioTrack]
    chosen: AudioTrack | None
    reason: str
    subs: list[SubTrack] = field(default_factory=list)


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        tail = " ".join(res.stderr.strip().splitlines()[-2:]) or "no output"
        raise SystemExit(f"{path.name}: ffprobe cannot read it — {tail}")
    try:
        return json.loads(res.stdout)
    except ValueError:
        raise SystemExit(f"{path.name}: ffprobe returned no stream information")


def probe(path: Path, prefer_index: int | None = None) -> ProbeResult:
    data = ffprobe(path)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    tracks: list[AudioTrack] = []
    subs: list[SubTrack] = []
    a = 0
    for s in data.get("streams", []):
        if s.get("codec_type") == "subtitle":
            tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
            disp = s.get("disposition") or {}
            codec = s.get("codec_name", "")
            subs.append(SubTrack(index=len(subs), stream_index=int(s.get("index", 0)),
                                 language=(tags.get("language") or "und").lower(), title=tags.get("title", "") or "",
                                 codec=codec, is_text=codec in TEXT_SUB_CODECS,
                                 forced=bool(disp.get("forced")), is_default=bool(disp.get("default"))))
            continue
        if s.get("codec_type") != "audio":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        tracks.append(AudioTrack(
            index=a, stream_index=int(s.get("index", a)),
            language=(tags.get("language") or "und").lower(),
            title=tags.get("title", "") or "",
            codec=s.get("codec_name", ""), channels=int(s.get("channels") or 0),
            sample_rate=int(s.get("sample_rate") or 0),
            is_default=bool((s.get("disposition") or {}).get("default")),
            duration=float(s.get("duration") or 0.0), start_time=float(s.get("start_time") or 0.0),
        ))
        a += 1
    # untagged text tracks: read the language from their own words (0.4.7) — a tag-less track used to be invisible
    for t in subs:
        if t.is_text and t.language in ("", "und") and not t.title:
            code = sniff_subtitle_language(path, t)
            if code:
                t.language, t.sniffed = code, True
    if not tracks:
        return ProbeResult(duration, [], None, "no audio streams", subs)
    if prefer_index is not None:
        for t in tracks:
            if t.index == prefer_index:
                return ProbeResult(duration, tracks, t, f"--audio-track {prefer_index}", subs)
        return ProbeResult(duration, tracks, None, f"audio track {prefer_index} does not exist", subs)
    for t in tracks:
        if t.language in JA_TAGS:
            return ProbeResult(duration, tracks, t, f"language tag '{t.language}'", subs)
    for t in tracks:
        if "日本" in t.title or "japan" in t.title.lower():
            return ProbeResult(duration, tracks, t, f"title '{t.title}'", subs)
    if len(tracks) == 1:
        return ProbeResult(duration, tracks, tracks[0], "only audio track (language untagged)", subs)
    for t in tracks:
        if t.is_default:
            return ProbeResult(duration, tracks, t, "default track (no Japanese tag found — check with --list-tracks)", subs)
    return ProbeResult(duration, tracks, tracks[0], "first track (no Japanese tag found — check with --list-tracks)", subs)


def describe_tracks(res: ProbeResult) -> str:
    lines = []
    for t in res.tracks:
        mark = "→" if res.chosen and t.index == res.chosen.index else " "
        lines.append(f" {mark} a:{t.index}  lang={t.language:<4} {t.codec:<6} {t.channels}ch {t.sample_rate} Hz"
                     f"{'  ' + str(round(t.duration / 60, 1)) + ' min' if t.duration else ''}"
                     f"{'  default' if t.is_default else ''}{'  ' + t.title if t.title else ''}")
    for t in res.subs:
        kind = "text" if t.is_text else "bitmap (needs OCR — ignored)"
        if not t.is_text:
            kind = "bitmap (PGS — read by OCR)" if t.codec == "hdmv_pgs_subtitle" else f"bitmap ({t.codec} — not PGS, not read)"
        lines.append(f"   s:{t.index}  lang={t.language:<4} {t.codec:<10} {kind}"
                     f"{' (language read from the text)' if t.sniffed else ''}"
                     f"{'  forced' if t.forced else ''}{'  default' if t.is_default else ''}{'  ' + t.title if t.title else ''}")
    return "\n".join(lines)
