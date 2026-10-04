"""Terminology (0.5.1, 2026-10-02): the names and recurring terms of a film, rendered once per target and fed to
every translation window through the glossary, so a character, a place or a ship is written the same way in the
last scene as in the first. The windows see twenty lines at a time and cannot know what an earlier window decided;
this pass decides once, up front.

Two steps, both cached in the work file:
  1. candidates — proper nouns and recurring special terms of the source text: a script-aware heuristic (katakana
     runs in Japanese, capitalised words in Latin scripts) plus one LLM pass over the transcript; kept when they
     occur at least TERMS_MIN_OCCURRENCES times, at most TERMS_MAX by frequency.
  2. renderings — one LLM call per target: each term as subtitles in that language would write it (standard
     transliteration for names, the established translation for titles). The user's own glossary always wins.
"""
from __future__ import annotations

import re
import sys
import time

from . import config
from .segment import Cue

_KATAKANA = re.compile(r"[ァ-ヴー]{2,}")
_CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-ZÀ-Ý][a-zà-ÿ]{2,}(?:\s[A-ZÀ-Ý][a-zà-ÿ]{2,})?")
# capitalised English words that are not names: interjections, pronouns, question words, days — the capitalised-word
# heuristic sees them after a dialogue dash or a question mark mid-cue (2026-10-04: "Yeah", "You", "What" were among
# the six most frequent "terms" of a film, crowding the cap)
_ENGLISH_STOP = {"Yeah", "Yes", "You", "What", "Why", "How", "Who", "When", "Where", "Okay", "Hey", "Oh", "Well", "Look",
                 "Come", "Wait", "Thanks", "Thank", "Sorry", "Please", "Right", "Sure", "Good", "Great", "God", "Jesus",
                 "Mom", "Dad", "Mommy", "Daddy", "Honey", "Baby", "Sir", "Miss", "Mister", "Doctor", "Hello", "Goodbye",
                 "Bye", "Hi", "Mister", "Mrs", "Mr", "Ms", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday", "Christmas", "Easter", "The", "And", "But", "Then",
                 # months are deliberately NOT here: April, May and June are names (April is a lead in Definitely, Maybe)
                 "Now", "Just", "Maybe", "Really", "Nothing", "Something", "Everything", "Nobody", "Somebody", "Everybody",
                 "Stop", "Go", "Get", "Let", "Listen", "Excuse", "Fine", "Love", "Dude", "Man", "Guys", "Hmm", "Uh", "Um"}
_COMMON_KATAKANA = {"ドア", "テレビ", "ママ", "パパ", "トイレ", "バス", "タクシー", "コーヒー", "ビール", "ゲーム", "アイス", "カメラ",
                    "ホテル", "レストラン", "ニュース", "メール", "パソコン", "スマホ", "ピザ", "ケーキ", "サラダ", "ジュース",
                    "ダメ", "チーズ", "ハチミツ", "ビデオ", "バイト", "デート", "プール", "ベッド", "ソファ", "ドライブ", "ペット",
                    "ラーメン", "カレー", "パン", "ミルク", "サッカー", "テスト", "ノート", "ペン", "ボール", "キス", "ドキドキ"}

EXTRACT_PROMPT = ("Below is dialogue from {genre}, in {language}. List the proper nouns in it — people's names, "
                  "nicknames, places, organisations, ships, products, titles of works — and any invented or recurring "
                  "special terms. One per line, exactly as written in the text, nothing else: no explanations, no "
                  "numbering, no translations.\n\n{text}")
# Every recurring term is rendered, names or not: 0.5.0.2 asked the model to drop "ordinary words" and it dropped
# the beetle and the oak the episode is about — the two terms the pass had demonstrably made consistent — while
# keeping "police". Recurrence is the criterion; the model's idea of a proper noun is not.
RENDER_PROMPT = ("These terms come from {genre} in {source}. For each, give the form that {target} subtitles would use: "
                 "the standard {target} transliteration or spelling for names, the established {target} translation for "
                 "titles and organisations, and the usual {target} word for anything else. A character's name is rendered "
                 "as the character is called in {target} dialogue, never as the title of the work the character is from. "
                 "Be consistent and conventional. Answer with one line per term in the form\nterm<TAB>rendering\nand "
                 "nothing else.\n\n{terms}")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def katakana_runs(texts: list[str]) -> dict[str, int]:
    """Every run of katakana in the text with its count — common words included (the fragment check needs them)."""
    counts: dict[str, int] = {}
    for t in texts:
        for m in _KATAKANA.findall(t):
            counts[m] = counts.get(m, 0) + 1
    return counts


def heuristic_candidates(texts: list[str], lang: str) -> dict[str, int]:
    """Likely names by script alone, with counts."""
    counts: dict[str, int] = {}
    if lang == "ja":
        for m, n in katakana_runs(texts).items():
            if m not in _COMMON_KATAKANA and len(m) >= 2:
                counts[m] = n
    elif lang in ("en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "cs", "hu", "ro", "tr", "id", "ms", "tl", "vi", "ca", "hr", "sk", "sl", "lt", "lv", "et"):
        for t in texts:
            for m in _CAPITALISED.findall(". " + t):           # a line start is a sentence start: capitalised anyway
                if m in _ENGLISH_STOP or m.split()[0] in _ENGLISH_STOP:
                    continue
                counts[m] = counts.get(m, 0) + 1
    return counts


def count_in(texts: list[str], term: str) -> int:
    return sum(t.count(term) for t in texts)


def _chunks(texts: list[str], size: int) -> list[str]:
    out, cur = [], []
    n = 0
    for t in texts:
        cur.append(t); n += len(t) + 1
        if n >= size:
            out.append("\n".join(cur)); cur, n = [], 0
    if cur:
        out.append("\n".join(cur))
    return out


def llm_candidates(client, texts: list[str], lang: str, genre: str, max_calls: int = 4) -> list[str]:
    """One pass of the LLM over the transcript (in pieces of ~TERMS_CHUNK_CHARS), union of what it names."""
    found: list[str] = []
    pieces = _chunks(texts, config.TERMS_CHUNK_CHARS)
    if len(pieces) > max_calls:                              # a long film: sample evenly
        step = len(pieces) / max_calls
        pieces = [pieces[int(i * step)] for i in range(max_calls)]
    for piece in pieces:
        prompt = EXTRACT_PROMPT.format(genre=genre, language=config.LANG_NAMES.get(lang, lang), text=piece)
        try:
            out = client.chat(None, prompt, max_tokens=600)
        except Exception as e:                               # noqa: BLE001
            _log(f"[terms] ⚠ extraction call failed: {e}")
            continue
        for line in (out or "").splitlines():
            s = re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip().strip("\"'「」『』")
            if 1 < len(s) <= 30 and not re.search(r"[:：→\t]", s):
                found.append(s)
    return found


def select_terms(texts: list[str], lang: str, heuristic: dict[str, int], from_llm: list[str]) -> list[tuple[str, int]]:
    """Merge, count in the text, keep the recurring ones, cap by frequency. A candidate that is a fragment of a
    longer word in the text — kept or not — is dropped: 0.5.0.2 rendered ハチミ, cut from ハチミツ (honey), as a
    name, because the whole word was on the common list and so no longer there to fold the fragment into."""
    cand = dict(heuristic)
    for s in from_llm:
        if s not in cand:
            cand[s] = count_in(texts, s)
    longer = dict(cand)
    if lang == "ja":
        for m, n in katakana_runs(texts).items():
            longer[m] = max(longer.get(m, 0), n)
    keep = [(t, n) for t, n in cand.items() if n >= config.TERMS_MIN_OCCURRENCES]
    keep.sort(key=lambda tn: (-tn[1], tn[0]))
    out: list[tuple[str, int]] = []
    for t, n in keep:
        if any(t != o and t in o and n <= m for o, m in longer.items()):
            continue
        out.append((t, n))
    return out[:config.TERMS_MAX]


def render_terms(client, terms: list[str], source: str, target: str, genre: str) -> dict[str, str]:
    """term → rendering in `target`, one LLM call; lines that do not parse are dropped (the windows then decide)."""
    if not terms:
        return {}
    prompt = RENDER_PROMPT.format(genre=genre, source=config.LANG_NAMES.get(source, source),
                                  target=config.LANG_NAMES.get(target, target), terms="\n".join(terms))
    try:
        out = client.chat(None, prompt, max_tokens=60 + 24 * len(terms))
    except Exception as e:                                   # noqa: BLE001
        _log(f"[terms] ⚠ rendering call failed: {e}")
        return {}
    got: dict[str, str] = {}
    want = set(terms)
    for line in (out or "").splitlines():
        if "\t" in line:
            a, b = line.split("\t", 1)
        elif "→" in line:
            a, b = line.split("→", 1)
        elif " - " in line:
            a, b = line.split(" - ", 1)
        else:
            continue
        a, b = a.strip().strip("\"'「」"), b.strip().strip("\"'「」")
        if a in want and b and len(b) <= 60 and b not in ("-", "–", "—", "—"):
            got[a] = b
    return got


def build_glossary(auto: dict[str, str], user: dict | None) -> dict[str, str]:
    """The automatic renderings, overridden by anything the user gave."""
    out = dict(auto)
    for k, v in (user or {}).items():
        out[k] = v
    return out


def summarise(terms: list[tuple[str, int]], rendered: dict[str, str], k: int = 6) -> str:
    shown = [f"{t} ({n})→{rendered[t]}" if t in rendered else f"{t} ({n})" for t, n in terms[:k]]
    return ", ".join(shown) + (" …" if len(terms) > k else "")
