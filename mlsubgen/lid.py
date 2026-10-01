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
  5. (v3, 2026-10-01) Where two neighbouring windows disagree, the switch is placed at the exact span: whisper is
     asked a two-way question (A or B?) about every span of the two windows and the single best split is taken.
     The same version made function words count for Latin-script languages (the script alone cannot tell
     English from Italian) and let a strongly evidenced single window of a third language survive the smoothing.
     Measured with `mlsubgen lidbench` against forced subtitle tracks: v2 reproduced about half the language
     changes within 3 s on Babel, Inglourious Basterds and Only God Forgives, with the window edge as the cause.

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


# Function words that are frequent in one Latin-script language and rare in the others (LID v3, 2026-10-01). The
# script of a decode says nothing between English, German, French, Italian or Spanish; these words do. Shared
# words (a, de, la, no, me, in, se…) are deliberately absent: a hit must point one way.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {k: frozenset(v.split()) for k, v in {
    "en": "the and you that this with have what your they about there which would their were from just when because",
    "es": "el los las que por como pero está usted muy más también nosotros ellos tiene hay donde aquí porque una",
    "fr": "les des est vous nous une pas pour dans avec sur qui c'est très aussi même votre notre cette sont",
    "de": "und ist nicht ich sie das wir ein eine auch der die dem mit sich aber noch nur schon haben wird",
    "it": "che della delle sono perché anche questo questa come gli nel dalla essere molto quando c'è lei una",
    "pt": "não você com uma isso ele ela nós eles está muito também mas são tem porque quando aqui",
    "nl": "het een niet van dat ik hij zij wij jullie maar ook nog wel zijn heeft worden deze omdat waarom",
    "pl": "nie się jest co jak ale już tylko tego jego bardzo może czy tak jeszcze będzie przez",
    "tr": "bir ve bu için ama değil çok daha onlar yok mı nasıl şimdi burada",
    "id": "yang dan tidak ini itu dengan untuk saya kamu kita mereka adalah akan sudah bisa ada apa karena",
    "ms": "yang dan tidak ini itu dengan untuk saya awak kita mereka adalah akan sudah boleh ada apa kerana",
    "tl": "ang ng mga sa ako ikaw siya kami tayo hindi ito iyan iyon kasi lang naman ba",
    "vi": "và của không có tôi bạn anh chị chúng được này đó như rồi đã sẽ",
    "ca": "els una que amb per com però està molt també nosaltres aquest aquesta són hi",
    "ro": "și este sunt pentru dar foarte acest această noi voi lor când unde",
    "sv": "och är inte jag det att han hon också bara har kan ska ni när",
    "da": "og er ikke jeg det hun også bare har kan skal når hvor",
    "no": "og er ikke jeg det hun også bare har kan skal når hvor",
    "fi": "ja ei minä sinä hän mutta myös vain olen olet ovat kun missä",
    "hu": "és nem az egy hogy vagyok vagy ők csak még már nagyon mert",
    "cs": "ale jsem jsi jsme jste jsou tak jak taky jen už ještě proč protože",
    "sk": "ale som sme ste sú tak ako tiež len už ešte prečo pretože",
    "hr": "ali sam smo ste su tako kako također samo već još zašto jer",
    "sl": "ampak sem smo ste so tako kako tudi samo že še zakaj ker",
    "lt": "yra bet aš tu mes jūs jie taip kaip pat tik jau dar kodėl nes",
    "lv": "bet es tu mēs jūs viņi tā kā arī tikai jau vēl kāpēc jo",
    "et": "ja ei aga mina sina meie teie nemad nii kuidas ka ainult juba veel miks sest",
}.items()}
_WORD = re.compile(r"[A-Za-zÀ-ɏ']+")


def text_language(text: str) -> tuple[str | None, int]:
    """Which Latin-script language the words point to: (language, hits) when one language has at least three
    distinct function-word hits and clearly beats the runner-up; (None, 0) otherwise."""
    words = {w.lower() for w in _WORD.findall(text.replace("’", "'"))}
    if len(words) < 3:
        return None, 0
    hits = {lang: len(words & fw) for lang, fw in _FUNCTION_WORDS.items()}
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    best, n = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if n >= 3 and n >= 1.5 * second:
        return best, n
    return None, 0


def script_vote(text: str) -> tuple[str | None, float]:
    """What the script — and, for Latin script, the function words — of Qwen's auto-decode says about the
    language, and how strongly."""
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
        lang, n = text_language(text)
        if lang:
            return lang, 0.5            # the words say which Latin-script language it is
        return "en", 0.15               # Latin script and nothing decisive: a faint lean, no more
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


def smooth(langs: list[str | None], confident: list[bool], min_run: int = config.LID_SWITCH_MIN_WINDOWS,
           margins: list[float] | None = None) -> list[str | None]:
    """Uncertain windows inherit the nearest confident neighbour; a confident run shorter than `min_run` of a language
    other than the dominant one is absorbed (it is a stray phrase, not a scene) — unless every window of the run
    is strongly evidenced (margin ≥ LID_STRONG_MARGIN: both detectors agreeing on real words). A strongly evidenced
    switch is never blocked (LID v3: a single ten-second exchange in a third language used to vanish)."""
    n = len(langs)
    if n == 0:
        return []
    conf_langs = [l for l, c in zip(langs, confident) if c and l]
    if not conf_langs:
        return [None] * n
    dominant = max(set(conf_langs), key=conf_langs.count)
    out: list[str | None] = list(langs)
    margins = margins or [0.0] * n
    # runs of confident windows in a non-dominant language must be long enough — or strong
    i = 0
    while i < n:
        if confident[i] and out[i] and out[i] != dominant:
            j = i
            while j < n and confident[j] and out[j] == out[i]:
                j += 1
            strong = all(margins[k] >= config.LID_STRONG_MARGIN for k in range(i, j))
            if j - i < min_run and not strong:
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
    labels = smooth([w.lang for w in judged], [w.confident for w in judged], margins=[w.margin for w in judged])
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
    for w, l in zip(windows, final):
        for i in w.spans:
            spans[i].lang = l
    moved = 0
    if engines.whisper is not None and not whisper_only:
        moved = refine_boundaries(audio, spans, windows, final, engines.whisper)
        if moved:
            notes.append(f"{moved} language switch(es) placed at the exact span by the boundary refinement")
    seconds: dict[str, float] = {}
    for s in spans:
        if s.lang:
            seconds[s.lang] = seconds.get(s.lang, 0.0) + s.dur
    dominant = max(seconds, key=seconds.get) if seconds else None
    conf = sum(1 for w in judged if w.confident)
    return LidResult(config.LID_VERSION, windows, spans, dominant, seconds, conf, len(judged) - conf, notes=notes)


def refine_boundaries(audio: np.ndarray, spans: list[Span], windows: list[Window], final: list[str | None],
                      whisper, min_span: float = 0.6) -> int:
    """Where two neighbouring windows disagree, the switch happened somewhere inside them — not at the window edge.
    Every span of the two windows gets whisper's probability for just the two candidate languages (a two-way
    question a short span can answer), and the one split point that best explains the sequence (all A, then all B)
    is taken. The windows themselves keep their labels; the spans carry the refined ones. Returns how many
    switches were moved off a window edge (LID v3, 2026-10-01: the baseline reproduced only half the language
    changes within 3 s, and the ten-second window was the reason)."""
    moved = 0
    for i in range(len(windows) - 1):
        a, b = final[i], final[i + 1]
        if not a or not b or a == b:
            continue
        idx = windows[i].spans + windows[i + 1].spans
        pa: list[float] = []
        pb: list[float] = []
        for k in idx:
            s = spans[k]
            if s.dur < min_span:
                pa.append(0.5); pb.append(0.5)             # too short to ask: no opinion
                continue
            probs = whisper.language_probs(slice_audio(audio, s.start, s.end))
            x, y = probs.get(a, 0.0), probs.get(b, 0.0)
            tot = x + y
            pa.append(x / tot if tot > 0 else 0.5)
            pb.append(y / tot if tot > 0 else 0.5)
        n = len(idx)
        left = len(windows[i].spans)
        # spans of the left window already given another language by the previous pair (A B A: the switch into
        # this window was placed by that pass) stay as they are: the split can only fall after them
        lock = max((pos + 1 for pos in range(left) if spans[idx[pos]].lang != a), default=0)
        # split after position c: spans[:c] are A, spans[c:] are B; c = n: all A; c = 0: all B
        best_c, best_score = left, None
        for c in range(lock, n + 1):
            score = sum(pa[lock:c]) + sum(pb[c:])
            if best_score is None or score > best_score + 1e-9:
                best_c, best_score = c, score
        if best_c != left:
            moved += 1
        for pos, k in enumerate(idx):
            if pos >= lock:
                spans[k].lang = a if pos < best_c else b
    return moved


def forced(spans: list[Span], lang: str) -> LidResult:
    """--source LANG: every span is that language, no detector consulted."""
    for s in spans:
        s.lang = lang
    return LidResult(config.LID_VERSION, [], spans, lang, {lang: sum(s.dur for s in spans)}, 0, 0, forced=True)
