"""Dual-engine transcription: Qwen3-ASR (lowest Japanese character error rate) and whisper (robust on music and
overlap, better on kanji and proper nouns) both decode every chunk; where they disagree, the local LLM
reconciles them into one transcript. Timing of the merged text needs no model: each character inherits the time
of the engine it came from, the rest are interpolated between their neighbours."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from . import config
from .asr import Word

_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)
NO_SPACE = {"ja", "zh", "yue", "th"}


def norm(text: str) -> str:
    return _STRIP.sub("", text)


def agreement(a: str, b: str) -> float:
    """Character-level similarity of two transcripts, punctuation and spaces ignored."""
    na, nb = norm(a), norm(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


def decide(qwen_text: str, whisper_text: str) -> tuple[str, str]:
    """(verdict, text): 'agree' → Qwen's text (lower CER) stands; 'qwen'/'whisper' → one side had nothing to say;
    'llm' → the LLM must reconcile (text is empty)."""
    nq, nw = norm(qwen_text), norm(whisper_text)
    if len(nq) < config.MERGE_MIN_CHARS and len(nw) < config.MERGE_MIN_CHARS:
        return "agree", qwen_text if len(nq) >= len(nw) else whisper_text
    if len(nq) < config.MERGE_MIN_CHARS:
        return "whisper", whisper_text
    if len(nw) < config.MERGE_MIN_CHARS:
        return "qwen", qwen_text
    if agreement(qwen_text, whisper_text) >= config.MERGE_AGREE:
        return "agree", qwen_text
    return "llm", ""


MERGE_SYSTEM = """You reconcile two automatic transcriptions of the same short stretch of {lang} audio into one correct transcript.
The material is {genre}.
A comes from a model with the lowest character error rate; B from a model that is better at kanji and proper nouns and more robust to background music, but that sometimes invents text where there is none.

Rules:
- Output ONLY the reconciled {lang} transcript, in the original language, nothing else — no notes, no labels, no translation.
- Keep every utterance that either side heard, in order; a line present in only one side is real unless it is obviously an artefact (the same phrase looping many times, a stock phrase like "ご視聴ありがとうございました" over nothing).
- Sung lines (opening and ending songs, a character singing) are content: keep them, and prefer A's version of sung passages.
- Where both heard the same words differently, prefer A's hearing of the sounds and B's choice of kanji and names. Keep the tone and register — rude stays rude.
- Do not add, summarise, tidy or complete anything. Punctuate naturally at sentence ends."""


def merge_prompt(qwen_text: str, whisper_text: str, lang_name: str, genre: str, before: str) -> tuple[str, str]:
    system = MERGE_SYSTEM.format(lang=lang_name, genre=genre)
    user = ((f"PRECEDING TRANSCRIPT (context only, do not output):\n{before}\n\n" if before else "")
            + f"A:\n{qwen_text}\n\nB:\n{whisper_text}\n\nReconciled transcript:")
    return system, user


def clean_llm(text: str, lang: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()
    text = re.sub(r"^(?:reconciled transcript|transcript|統合|文字起こし)\s*[:：]\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] in "「\"“" and text[-1] in "」\"”":
        text = text[1:-1]
    if lang in NO_SPACE:
        text = re.sub(r"(?<=[぀-ヿ一-鿿])\s+(?=[぀-ヿ一-鿿])", "", text)
    return text.strip()


def char_timeline(words: list[Word]) -> list[tuple[str, float, float]]:
    out = []
    for w in words:
        chars = [c for c in w.text if not c.isspace()]
        if not chars:
            continue
        step = (w.end - w.start) / len(chars)
        for i, ch in enumerate(chars):
            out.append((ch, w.start + i * step, w.start + (i + 1) * step))
    return out


def transfer_timing(merged: str, primary: list[Word], secondary: list[Word], start: float, end: float,
                    lang: str = "ja") -> list[Word]:
    """Time the merged text from the two engines' timelines: characters matching the primary take its times, the
    rest try the secondary, what neither had is interpolated between the nearest timed neighbours (chunk bounds
    at the edges). Non-monotonic inheritances are discarded and interpolated instead."""
    chars = [c for c in merged if not c.isspace()]
    if not chars:
        return []
    times: list[tuple[float, float] | None] = [None] * len(chars)
    for tl in (char_timeline(primary), char_timeline(secondary)):
        if not tl:
            continue
        sm = SequenceMatcher(None, chars, [c for c, _, _ in tl], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for k in range(i2 - i1):
                if times[i1 + k] is None:
                    times[i1 + k] = (tl[j1 + k][1], tl[j1 + k][2])
    # enforce monotonic starts: an inherited time that goes backwards is not trusted
    last = start
    for i, t in enumerate(times):
        if t is None:
            continue
        if t[0] < last - 0.3 or t[0] > end + 0.3:
            times[i] = None
        else:
            last = max(last, t[0])
    # interpolate gaps
    n = len(chars)
    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else start
        right = times[j][0] if j < n else end
        if right <= left:
            right = left + 0.1 * (j - i)
        span = right - left
        for k in range(i, j):
            a = left + span * (k - i) / (j - i)
            times[k] = (a, left + span * (k - i + 1) / (j - i))
        i = j
    if lang in NO_SPACE:
        return [Word(c, round(t[0], 3), round(max(t[1], t[0] + 0.01), 3), lang) for c, t in zip(chars, times)]
    # spaced languages: one Word per whitespace-separated token
    out: list[Word] = []
    k = 0
    for tok in merged.split():
        m = len([c for c in tok if not c.isspace()])
        if m == 0:
            continue
        s, e = times[k][0], times[k + m - 1][1]
        out.append(Word(tok, round(s, 3), round(max(e, s + 0.01), 3), lang))
        k += m
    return out


def merge_chunks(dual: list[dict], client, lang: str, lang_name: str, genre: str, log) -> tuple[list[Word], dict]:
    """dual: per chunk {"start", "end", "qwen": {"text", "words"}, "whisper": {"text", "words"}, "retimed"}.
    Returns the final words for the file and the merge statistics."""
    stats = {"agree": 0, "qwen": 0, "whisper": 0, "llm": 0, "llm_failed": 0}
    words: list[Word] = []
    before = ""
    for ch in dual:
        q, w = ch["qwen"], ch["whisper"]
        qwords = [Word(d["text"], d["start"], d["end"], lang) for d in q["words"]]
        wwords = [Word(d["text"], d["start"], d["end"], lang) for d in w["words"]]
        verdict, text = decide(q["text"], w["text"])
        if verdict == "llm":
            if client is None:
                verdict, text = "qwen", q["text"]
                stats["llm_failed"] += 1
            else:
                system, user = merge_prompt(q["text"], w["text"], lang_name, genre, before[-200:])
                try:
                    text = clean_llm(client.chat(system, user, max_tokens=1024), lang)
                except Exception as e:  # noqa: BLE001
                    log(f"[merge] ⚠ LLM failed on chunk {ch['start']:.0f}-{ch['end']:.0f}s ({e}) — keeping A")
                    text = q["text"]
                    stats["llm_failed"] += 1
                if not norm(text) or len(norm(text)) > 3 * max(len(norm(q["text"])), len(norm(w["text"]))):
                    text = q["text"]                     # empty or runaway answer: not a reconciliation
                    stats["llm_failed"] += 1
        stats[verdict] += 1
        # primary timeline: Qwen's aligner where it held, whisper's DTW where the aligner had to be repaired
        if ch.get("retimed", 0) > 0.3 * max(len(qwords), 1) or verdict == "whisper":
            primary, secondary = wwords, qwords
        else:
            primary, secondary = qwords, wwords
        got = transfer_timing(text, primary, secondary, ch["start"], ch["end"], lang)
        words += got
        before = text
    return words, stats
