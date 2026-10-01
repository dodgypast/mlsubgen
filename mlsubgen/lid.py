"""Language identification: decide the language, THEN decode.

The old check took 30 s of the longest speech chunk and trusted Qwen's bare label — in sparse-speech material
that is 240 s of mostly non-verbal audio, and 21 Japanese films were skipped as Chinese or English. Here the
language is a property of each stretch of speech:

  1. VAD spans are glued into detection windows of ~10 s of actual SPEECH (spliced, gaps removed; ≤ 30 s audio).
  2. Each window is judged by two detectors: faster-whisper's language detector, which returns a probability
     (a window with no real words gives a flat distribution → low probability → "uncertain", not "Chinese"), and
     Qwen in auto mode, whose label and text give a second vote — the text's script is evidence on its own
     (kana → Japanese, Thai block → Thai, Hangul → Korean).
  3. Confident windows anchor; uncertain windows inherit the nearest confident neighbour, and a switch to a
     second language needs LID_SWITCH_MIN_WINDOWS consecutive confident windows — a stray phrase does not
     fragment the file, a real bilingual scene gets its own label.
  4. First pass samples LID_SAMPLE_WINDOWS windows spread over the file; if a second language shows up in the
     sample, every window is judged so the switch points are exact.

The result labels every VAD span; chunks are then built so each has one language, and each is decoded with
that language forced (see asr.py). "unknown" is a state, not a guess.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np

from . import config
from .audio import SR, slice_audio
from .vad import Span

_KANA = re.compile(r"[ぁ-ゖァ-ヺー]")
_HAN = re.compile(r"[一-鿿]")
_THAI = re.compile(r"[฀-๿]")
_HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_GREEK = re.compile(r"[Ͱ-Ͽ]")
_ARABIC = re.compile(r"[؀-ۿ]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_NAME_TO_CODE = {v.lower(): k for k, v in config.LANG_NAMES.items()}
_NAME_TO_CODE.update({"mandarin": "zh", "cantonese": "yue", "unknown": None, "none": None, "": None})


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def code_from_name(name) -> str | None:
    """'Japanese' → 'ja'; already a code → itself; unknown → None."""
    if not name:
        return None
    s = str(name).strip().lower()
    if s in _NAME_TO_CODE:
        return _NAME_TO_CODE[s]
    if s in config.LANG_NAMES:
        return s
    return s if len(s) in (2, 3) and s.isalpha() else None


def script_of(text: str) -> str:
    """The dominant script of a text: ja | th | ko | han | latin | cyrillic | greek | arabic | devanagari | none."""
    counts = {"ja": len(_KANA.findall(text)), "han": len(_HAN.findall(text)), "th": len(_THAI.findall(text)),
              "ko": len(_HANGUL.findall(text)), "latin": len(_LATIN.findall(text)),
              "cyrillic": len(_CYRILLIC.findall(text)), "greek": len(_GREEK.findall(text)),
              "arabic": len(_ARABIC.findall(text)), "devanagari": len(_DEVANAGARI.findall(text))}
    if counts["ja"] > 0 and counts["ja"] + counts["han"] >= counts["latin"]:
        return "ja"                       # any kana with Han makes it Japanese, not Chinese
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "none"


SCRIPT_OF_LANG = {"ja": "ja", "zh": "han", "yue": "han", "th": "th", "ko": "ko", "ru": "cyrillic", "uk": "cyrillic",
                  "el": "greek", "ar": "arabic", "hi": "devanagari"}


def script_matches(text: str, lang: str) -> bool:
    """Does the decoded text look like this language? Latin-script languages cannot be told apart by script, so any
    Latin text passes for them; Chinese passes for Han text WITHOUT kana."""
    s = script_of(text)
    want = SCRIPT_OF_LANG.get(lang, "latin")
    if want == "han":
        return s == "han"
    if want == "latin":
        return s in ("latin", "none")
    if lang == "ja":
        return s in ("ja", "han")       # a short all-kanji line is still Japanese
    return s == want


def script_vote(text: str) -> tuple[str | None, float]:
    """What the script of Qwen's auto-decode says about the language, and how strongly."""
    s = script_of(text)
    if s == "ja":
        return "ja", 0.7
    if s in ("th", "ko"):
        return s, 0.7
    if s == "han":
        return "zh", 0.3                # Han only: Chinese OR Japanese-without-kana; a weak vote
    if s == "greek":
        return "el", 0.7
    if s == "cyrillic":
        return None, 0.0                # Russian, Ukrainian, Bulgarian… share it: words, yes; a language, no
    if s == "latin":
        return "en", 0.3                # any Latin-script language; weak
    return None, 0.0


# characters that are vocalisation, not words: CJK interjections, kana fillers, prolongation marks
_NON_LEXICAL = set("嗯啊哦呃唔哈呀嘿哼嘛哎唉噢喔んあぁいぃうぅえぇおぉっーはぁふぅむぅ・…～〜ー")
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def lexical(text: str) -> bool:
    """Does the text contain words, as opposed to moans, fillers and punctuation? Agreement between detectors on a
    window with no words is worth nothing — Qwen decodes non-verbal vocalisation as 嗯/啊 and calls it Chinese."""
    if script_of(text) == "latin":
        return sum(1 for w in re.findall(r"[A-Za-zÀ-ɏ]{3,}", text) if w.lower() not in ("hmm", "mmm", "ahh", "ooh", "aah")) >= 2
    core = [ch for ch in _PUNCT.sub("", text) if ch not in _NON_LEXICAL]
    return len(core) >= 4


def decide(w_lang: str | None, w_prob: float, q_lang: str | None, text: str) -> tuple[str | None, float, float]:
    """Combine the votes: (language, score, margin). Confident when the score reaches LID_CONFIDENT_SCORE and beats
    the runner-up by LID_CONFIDENT_MARGIN. Without lexical content in Qwen's decode, Qwen's vote is nearly worthless
    and the script says nothing — only a very sure whisper can carry the window then."""
    scores: dict[str, float] = {}
    if w_lang:
        scores[w_lang] = scores.get(w_lang, 0.0) + max(0.0, min(1.0, w_prob))
    words = lexical(text)
    if q_lang:
        scores[q_lang] = scores.get(q_lang, 0.0) + (0.5 if words else 0.1)
    s_lang, s_w = script_vote(text) if words else (None, 0.0)
    if s_lang:
        scores[s_lang] = scores.get(s_lang, 0.0) + s_w
    if not scores:
        return None, 0.0, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, score = ranked[0]
    margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return best, round(score, 2), round(margin, 2)


def is_confident(score: float, margin: float) -> bool:
    return score >= config.LID_CONFIDENT_SCORE and margin >= config.LID_CONFIDENT_MARGIN


@dataclass
class Window:
    start: float
    end: float
    speech: float
    spans: list[int]                       # indices into the span list
    whisper: tuple[str | None, float] = (None, 0.0)
    qwen: tuple[str | None, str] = (None, "")
    lang: str | None = None
    score: float = 0.0
    margin: float = 0.0
    confident: bool = False
    judged: bool = False

    def to_dict(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2), "speech": round(self.speech, 1),
                "whisper": [self.whisper[0], round(self.whisper[1], 3)], "qwen": [self.qwen[0], self.qwen[1][:60]],
                "lang": self.lang, "score": self.score, "margin": self.margin, "confident": self.confident,
                "judged": self.judged}


@dataclass
class LidResult:
    version: int
    windows: list[Window]
    spans: list[Span]                      # every VAD span, now labelled
    dominant: str | None
    seconds: dict[str, float]              # speech seconds per language (after smoothing)
    confident_windows: int
    uncertain_windows: int
    forced: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_speech(self) -> float:
        return sum(s.dur for s in self.spans)

    def summary(self) -> str:
        if self.forced:
            return f"forced {self.dominant}"
        total = self.total_speech or 1.0
        parts = ", ".join(f"{lang} {sec / total * 100:.0f}%" for lang, sec in
                          sorted(self.seconds.items(), key=lambda kv: kv[1], reverse=True))
        return (f"{self.dominant or 'unknown'} ({parts}) — {self.confident_windows} confident / "
                f"{self.uncertain_windows} uncertain windows")

    def to_dict(self) -> dict:
        return {"version": self.version, "dominant": self.dominant, "forced": self.forced,
                "seconds": {k: round(v, 1) for k, v in self.seconds.items()},
                "confident_windows": self.confident_windows, "uncertain_windows": self.uncertain_windows,
                "spans": [[round(s.start, 3), round(s.end, 3), s.lang] for s in self.spans],
                "windows": [w.to_dict() for w in self.windows], "notes": self.notes, "summary": self.summary()}

    @staticmethod
    def from_dict(d: dict) -> "LidResult":
        spans = [Span(float(a), float(b), lang) for a, b, lang in d.get("spans", [])]
        wins = [Window(w["start"], w["end"], w["speech"], [], (w["whisper"][0], w["whisper"][1]),
                       (w["qwen"][0], w["qwen"][1]), w["lang"], w["score"], w["margin"], w["confident"],
                       w.get("judged", True)) for w in d.get("windows", [])]
        return LidResult(d["version"], wins, spans, d.get("dominant"), d.get("seconds", {}),
                         d.get("confident_windows", 0), d.get("uncertain_windows", 0), d.get("forced", False),
                         d.get("notes", []))


def build_windows(spans: list[Span], min_speech: float = config.LID_WINDOW_SPEECH_SEC,
                  max_audio: float = config.LID_WINDOW_MAX_AUDIO_SEC) -> list[Window]:
    """Consecutive spans until `min_speech` seconds of speech (or `max_audio` seconds of audio) — every span belongs
    to exactly one window."""
    out: list[Window] = []
    cur: list[int] = []
    speech = 0.0
    for i, s in enumerate(spans):
        if cur and (speech >= min_speech or s.end - spans[cur[0]].start > max_audio):
            out.append(Window(spans[cur[0]].start, spans[cur[-1]].end, speech, cur))
            cur, speech = [], 0.0
        cur.append(i)
        speech += s.dur
    if cur:
        if out and speech < min_speech / 3:          # a tiny tail joins the previous window
            out[-1].spans += cur; out[-1].end = spans[cur[-1]].end; out[-1].speech += speech
        else:
            out.append(Window(spans[cur[0]].start, spans[cur[-1]].end, speech, cur))
    return out


def splice(audio: np.ndarray, spans: list[Span], max_sec: float = config.LID_WINDOW_MAX_AUDIO_SEC) -> np.ndarray:
    """The speech of these spans concatenated (gaps removed), capped at `max_sec`."""
    parts = []
    have = 0
    for s in spans:
        piece = slice_audio(audio, s.start, s.end)
        room = int(max_sec * SR) - have
        if room <= 0:
            break
        piece = piece[:room]
        parts.append(piece)
        have += len(piece)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def sample_indices(n: int, k: int) -> list[int]:
    """k indices spread evenly over range(n) (all of them when n ≤ k)."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def smooth(langs: list[str | None], confident: list[bool], min_run: int = config.LID_SWITCH_MIN_WINDOWS) -> list[str | None]:
    """Uncertain windows inherit the nearest confident neighbour; a confident run shorter than `min_run` of a language
    other than the dominant one is absorbed (it is a stray phrase, not a scene)."""
    n = len(langs)
    if n == 0:
        return []
    conf_langs = [l for l, c in zip(langs, confident) if c and l]
    if not conf_langs:
        return [None] * n
    dominant = max(set(conf_langs), key=conf_langs.count)
    out: list[str | None] = list(langs)
    # runs of confident windows in a non-dominant language must be long enough
    i = 0
    while i < n:
        if confident[i] and out[i] and out[i] != dominant:
            j = i
            while j < n and confident[j] and out[j] == out[i]:
                j += 1
            if j - i < min_run:
                for k in range(i, j):
                    out[k] = dominant
            i = j
        else:
            i += 1
    # uncertain windows: nearest confident neighbour (ties → the earlier one)
    anchors = [i for i in range(n) if confident[i] and out[i]]
    for i in range(n):
        if not (confident[i] and out[i]):
            nearest = min(anchors, key=lambda a: (abs(a - i), a))
            out[i] = out[nearest]
    return out


def identify(audio: np.ndarray, spans: list[Span], engines, whisper_only: bool = False,
             sample: int = config.LID_SAMPLE_WINDOWS) -> LidResult:
    """Label every span. `engines` provides `.whisper` (may be None) and `.qwen`, each with identify(piece)."""
    windows = build_windows(spans)
    if not windows:
        return LidResult(config.LID_VERSION, [], spans, None, {}, 0, 0, notes=["no speech"])

    def judge(w: Window) -> None:
        piece = splice(audio, [spans[i] for i in w.spans])
        wl, wp = (None, 0.0)
        ql, qt = (None, "")
        wh = engines.whisper
        if wh is not None:
            wl, wp = wh.identify(piece)
        if not whisper_only:
            ql, qt = engines.qwen.identify(piece)
        w.whisper, w.qwen = (wl, wp), (ql, qt)
        w.lang, w.score, w.margin = decide(wl, wp, ql, qt)
        w.confident = is_confident(w.score, w.margin)
        w.judged = True

    picked = sample_indices(len(windows), sample)
    for i in picked:
        judge(windows[i])
    langs_seen = {windows[i].lang for i in picked if windows[i].confident and windows[i].lang}
    notes = []
    if len(langs_seen) > 1 and len(picked) < len(windows):
        notes.append(f"{len(langs_seen)} languages in the {len(picked)}-window sample — judging all {len(windows)}")
        for w in windows:
            if not w.judged:
                judge(w)
    judged = [w for w in windows if w.judged]
    labels = smooth([w.lang for w in judged], [w.confident for w in judged])
    # every window gets a label: unjudged ones (monolingual sample) take the nearest judged neighbour's
    label_at = {id(w): l for w, l in zip(judged, labels)}
    final: list[str | None] = []
    last = None
    for w in windows:
        if id(w) in label_at:
            last = label_at[id(w)]
        final.append(last)
    for i, l in enumerate(final):                  # leading unjudged windows take the first label
        if l is None and last is not None:
            final[i] = next((x for x in final[i:] if x), last)
    seconds: dict[str, float] = {}
    for w, l in zip(windows, final):
        for i in w.spans:
            spans[i].lang = l
        if l:
            seconds[l] = seconds.get(l, 0.0) + w.speech
    dominant = max(seconds, key=seconds.get) if seconds else None
    conf = sum(1 for w in judged if w.confident)
    return LidResult(config.LID_VERSION, windows, spans, dominant, seconds, conf, len(judged) - conf, notes=notes)


def forced(spans: list[Span], lang: str) -> LidResult:
    """--source LANG: every span is that language, no detector consulted."""
    for s in spans:
        s.lang = lang
    return LidResult(config.LID_VERSION, [], spans, lang, {lang: sum(s.dur for s in spans)}, 0, 0, forced=True)
