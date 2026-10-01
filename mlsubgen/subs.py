"""Embedded subtitle tracks. A text track in a target language satisfies that target — it is left alone, no
sidecar is written. A non-forced, non-signs text track in any of the 45 languages we name is a better transcript
than any ASR, so it replaces the VAD/ASR stages and goes straight to translation of the targets that are missing:
the spoken language's track first (the audio tag), then Japanese, English and the rest. Bitmap tracks (PGS, VobSub)
would need OCR and are ignored."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import config
from .probe import EN_TAGS, JA_TAGS, ProbeResult, SubTrack
from .segment import Cue
from .srt import SrtCue, read_srt, write_srt
from .work import temp_beside

MIN_CUES = 10                         # fewer than this is not a dialogue track
PARTIAL_RE = re.compile(r"sign|song|forced|commentary|karaoke|lyric|caption only", re.I)
# container language tags (ISO 639-2 B/T, 639-1, names) → our codes — every one of config.LANG_NAMES (2026-10-01;
# the 17 languages added in 0.3.3 were missing, so their tracks were neither "already done" nor a transcript)
_ISO3 = {"en": "eng", "ja": "jpn", "th": "tha", "zh": "zho chi cmn", "ko": "kor", "yue": "yue", "de": "ger deu",
         "fr": "fre fra", "es": "spa", "it": "ita", "pt": "por", "ru": "rus", "id": "ind", "vi": "vie", "tr": "tur",
         "hi": "hin", "ar": "ara", "nl": "dut nld", "pl": "pol", "cs": "cze ces", "sv": "swe", "da": "dan", "fi": "fin",
         "no": "nor nob nno", "hu": "hun", "ro": "rum ron", "el": "gre ell", "uk": "ukr", "ms": "may msa", "tl": "tgl fil",
         "fa": "per fas", "he": "heb", "bn": "ben", "ta": "tam", "km": "khm", "lo": "lao", "my": "bur mya", "ca": "cat",
         "bg": "bul", "hr": "hrv scr", "sk": "slo slk", "sl": "slv", "lt": "lit", "lv": "lav", "et": "est"}
LANG_TAGS: dict[str, set[str]] = {code: {code, config.LANG_NAMES[code].lower(), *(_ISO3.get(code, "").split())}
                                  for code in config.LANG_NAMES}
LANG_TAGS["ja"] |= JA_TAGS
LANG_TAGS["en"] |= EN_TAGS
# a track titled in the language counts too: the English name, the language's own name, and a few common extras
_TITLE_EXTRA = {"zh": ("中文", "简体", "繁體", "繁体"), "id": ("bahasa",), "tl": ("tagalog", "filipino"), "fa": ("farsi", "persian"),
                "my": ("myanmar", "burmese"), "no": ("bokmål", "nynorsk")}
TITLE_WORDS: dict[str, tuple[str, ...]] = {
    code: tuple(dict.fromkeys((config.LANG_NAMES[code].lower(), config.NATIVE_NAMES.get(code, "").lower(), *_TITLE_EXTRA.get(code, ()))))
    for code in config.LANG_NAMES}
TITLE_WORDS["id"] = ("indonesian", "bahasa indonesia", "bahasa")      # "bahasa" alone is Indonesian by convention; Malay has its own name
TITLE_WORDS["ms"] = ("malay", "bahasa melayu")


def code_for_tag(tag: str | None) -> str | None:
    """A container language tag (jpn, ja, japanese, …) → our code, or None for und / unknown."""
    tag = (tag or "").strip().lower()
    if not tag or tag == "und":
        return None
    for code, tags in LANG_TAGS.items():
        if tag in tags:
            return code
    return None

_ASS_TAG = re.compile(r"\{[^}]*\}")                           # {\an8}{\i1}… override tags that survive conversion
_SPEAKER = re.compile(r"^[（(][^）)]{1,12}[)）]")                # （田中）こんにちは — closed-caption speaker labels
_SFX_ONLY = re.compile(r"^(?:[（(][^）)]*[)）]|[\[［][^\]］]*[\]］]|【[^】]*】|[♪♬〜～・…\s]+)$")   # （拍手）, ［音楽］, ♪～


def pick(subs: list[SubTrack], lang: str) -> SubTrack | None:
    """The first usable text track in `lang`: tagged or titled so, not forced/signs-only."""
    tags = LANG_TAGS.get(lang, {lang})
    words = TITLE_WORDS.get(lang, ())
    for t in subs:
        if not t.is_text or t.forced or PARTIAL_RE.search(t.title):
            continue
        if t.language in tags or any(w in t.title.lower() for w in words):
            return t
    return None


_BRACES = re.compile(r"\{[^}]*\}")                                  # ASS override tags and {comments} — never rendered
_SKIP_STYLE = re.compile(r"romaji|roman|kanji|karaoke|\bkara\b|sign|title|logo|credit|staff|typeset|note|lyric", re.I)
_TOP_OVERRIDE = re.compile(r"\\an[4-9]\b|\\pos\(|\\move\(")          # positioned, middle or top: signs, titles, karaoke
_DECOR = re.compile(r"\\(?:1?c&H|fn|fs\d|t\(|fad|fade|bord|be\d|blur|fsc|fr[xyz]?\d|clip|shad)")   # typesetting tags


def decorated(body: str) -> bool:
    """Typeset text (a logo, a sign built letter by letter): three or more override blocks carrying colour/font/
    animation tags. Dialogue carries at most an italic or a position."""
    blocks = _BRACES.findall(body)
    return sum(1 for b in blocks if _DECOR.search(b)) >= 3
_ASS_TIME = re.compile(r"^(\d+):(\d\d):(\d\d)[.:](\d\d)$")


def _ass_seconds(t: str) -> float | None:
    m = _ASS_TIME.match(t.strip())
    if not m:
        return None
    h, mi, se, cs = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + se + cs / 100


def ass_to_cues(text: str) -> list[SrtCue]:
    """Dialogue events of an ASS/SSA script as plain cues. Dropped: styles that are karaoke/romaji/signs/credits by
    name, events with a scroll/banner effect, and positioned or top-aligned lines while dialogue is on screen (a
    sign that stands alone is kept — it is probably a translated on-screen text). Override tags and {comments}
    are stripped, \\N becomes a line break, and karaoke fragments repeating the same line are merged."""
    section = None
    styles: dict[str, dict] = {}
    style_fmt: list[str] | None = None
    fmt: list[str] | None = None
    top_values = {"7", "8", "9"}
    events: list[tuple[float, float, str, bool]] = []
    for raw in text.splitlines():
        line = raw.strip("\ufeff\r ")
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section == "v4 styles":
                top_values = {"5", "6", "7"}          # SSA v4 alignment numbering differs from ASS v4+
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        if section in ("v4+ styles", "v4 styles"):
            if key == "format":
                style_fmt = [x.strip().lower() for x in rest.split(",")]
            elif key == "style" and style_fmt:
                vals = [x.strip() for x in rest.split(",", len(style_fmt) - 1)]
                styles[vals[0]] = dict(zip(style_fmt, vals))
        elif section == "events":
            if key == "format":
                fmt = [x.strip().lower() for x in rest.split(",")]
            elif key == "dialogue" and fmt:
                vals = rest.split(",", len(fmt) - 1)
                rec = {k: (v if k == "text" else v.strip()) for k, v in zip(fmt, vals)}
                if rec.get("effect", "").strip():
                    continue                            # scroll / banner / karaoke effect lines
                style = rec.get("style", "").lstrip("*")
                if _SKIP_STYLE.search(style):
                    continue
                st, en = _ass_seconds(rec.get("start", "")), _ass_seconds(rec.get("end", ""))
                body = rec.get("text", "")
                if st is None or en is None or en <= st:
                    continue
                top = (bool(_TOP_OVERRIDE.search(body)) or styles.get(style, {}).get("alignment", "") in top_values
                       or decorated(body))
                clean = _BRACES.sub("", body).replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
                clean = "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in clean.split("\n"))
                clean = re.sub(r"\n{2,}", "\n", clean).strip()
                if clean:
                    events.append((st, en, clean, top))
    events.sort(key=lambda e: (e[0], e[1]))
    # positioned lines only survive when nothing else is on screen at the time
    plain = [e for e in events if not e[3]]
    kept: list[tuple[float, float, str]] = []
    for st, en, txt, top in events:
        if top and any(p[0] < en and p[1] > st for p in plain):
            continue
        kept.append((st, en, txt))
    out: list[SrtCue] = []
    for st, en, txt in kept:
        if out and out[-1].text == txt and st <= out[-1].end + 0.05:      # karaoke fragments of one line
            out[-1].end = max(out[-1].end, en)
            continue
        out.append(SrtCue(st, en, txt))
    return out


def _run_ffmpeg(video: Path, s_index: int, codec: str, fmt: str, tmp: Path) -> None:
    cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(video), "-map", f"0:s:{s_index}", "-c:s", codec,
           "-f", fmt, str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = "\n".join(res.stderr.strip().splitlines()[-5:])
        raise SystemExit(f"{video.name}: ffmpeg could not extract subtitle track s:{s_index}\n{tail}")


def extract_track(video: Path, s_index: int, out: Path, codec: str | None = None) -> int:
    """Subtitle stream s:N → a clean .srt at `out`; returns the cue count. ASS/SSA tracks are read as ASS (styles
    decide what is dialogue; tags, comments and karaoke go); other text tracks are converted by ffmpeg and
    stripped of tags. Written atomically (temp beside `out`, renamed when complete)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_beside(out)
    try:
        if (codec or "").lower() in ("ass", "ssa"):
            _run_ffmpeg(video, s_index, "copy", "ass", tmp)
            cues = ass_to_cues(tmp.read_text(encoding="utf-8-sig", errors="replace"))
        else:
            _run_ffmpeg(video, s_index, "srt", "srt", tmp)
            cues = []
            for c in read_srt(tmp):
                txt = "\n".join(l.strip() for l in _BRACES.sub("", c.text).splitlines() if l.strip())
                if txt and c.end > c.start:
                    cues.append(SrtCue(c.start, c.end, txt))
    finally:
        tmp.unlink(missing_ok=True)
    write_srt(out, cues)
    return len(cues)


# Which embedded track becomes the transcript, when it is not in a target language: the spoken language first
# (the audio track's tag — the only transcript that is not already a translation), then, for untagged audio, the
# old order: Japanese (a Japanese text track almost always means Japanese audio), English (the pivot the
# translators do best from), Chinese, Korean, Thai, then every other language we name.
SOURCE_ORDER = ("ja", "en", "zh", "ko", "th") + tuple(l for l in config.LANG_NAMES if l not in ("ja", "en", "zh", "ko", "th"))


def source_order(spoken: str | None) -> tuple[str, ...]:
    return ((spoken,) if spoken else ()) + tuple(l for l in SOURCE_ORDER if l != spoken)


def plan_embedded(subs: list[SubTrack], targets: list[str], mode: str,
                  spoken: str | None = None) -> tuple[list[str], tuple[SubTrack, str] | None]:
    """(targets already embedded as text tracks, the text track to use as the transcript + its language).
    Any non-forced, non-signs text track in one of our 45 languages qualifies as a transcript; `spoken` (the audio
    track's language, when tagged) is preferred. mode auto: both; mode ja: embedded target tracks are ignored (we
    make our own translation) but a source track still replaces the ASR; mode ignore: nothing."""
    if mode == "ignore" or not subs:
        return [], None
    satisfied = [t for t in targets if mode == "auto" and pick(subs, t)]
    remaining = [t for t in targets if t not in satisfied]
    if not remaining:
        return satisfied, None
    for lang in source_order(spoken):
        if lang in remaining:
            continue
        t = pick(subs, lang)
        if t:
            return satisfied, (t, lang)
    return satisfied, None


def plan_sources(video: Path, subs: list[SubTrack], targets: list[str], mode: str,
                 spoken: str | None = None) -> tuple[list[str], tuple[object, str] | None]:
    """plan_embedded, then sidecar files: a <video>.<lang>.srt beside the video in a source language (ours from an
    earlier run, or anyone's) is a timed transcript too — the missing targets are translated from it instead of
    transcribing the audio again. The source is (SubTrack, lang) or (Path, lang); mode ignore = audio only."""
    satisfied, source = plan_embedded(subs, targets, mode, spoken)
    if source is not None or mode == "ignore":
        return satisfied, source
    remaining = [t for t in targets if t not in satisfied]
    if not remaining:
        return satisfied, None
    for lang in source_order(spoken):
        if lang in remaining:
            continue                                     # being (re)made now — not a source
        side = video.with_name(f"{video.stem}.{lang}.srt")
        if side.is_file():
            return satisfied, (side, lang)
    return satisfied, None


def clean_ja_line(line: str) -> str:
    line = _ASS_TAG.sub("", line).strip().replace("＜", "").replace("＞", "")
    if not line or _SFX_ONLY.match(line):
        return ""
    return _SPEAKER.sub("", line).strip()


def cues_from_track(srt_path: Path, lang: str = "ja") -> list[Cue]:
    """Subtitle file → source cues with the track's own timings. Display lines are joined without spaces for
    Japanese/Chinese/Thai and with spaces otherwise; CC speaker labels and sound-effect-only lines go."""
    joiner = "" if lang in ("ja", "zh", "yue", "th", "km", "lo", "my") else " "    # scripts without word spaces
    cues: list[Cue] = []
    for c in read_srt(srt_path):
        parts = [clean_ja_line(line) for line in c.text.splitlines()]
        text = joiner.join(p for p in parts if p).strip()
        if text and c.end > c.start:
            cues.append(Cue(len(cues), c.start, c.end, text, lang=lang))
    return cues


def describe(pr: ProbeResult) -> str:
    return ", ".join(f"s:{t.index} {t.language} {t.codec}{' forced' if t.forced else ''}{' ' + repr(t.title) if t.title else ''}"
                     for t in pr.subs)
