"""Register agreement (0.5.18, 2026-10-06) — the measurement chrF++ cannot make.

chrF++ counts character n-grams, and what the character work changes is a particle, a pronoun or a verb ending:
a few characters in a line, invisible to it (night one: every language within two points, off against on). This
score asks the question directly. For each language a small set of REGISTER FEATURES is read off a line with
regular expressions — the Thai politeness particle (ครับ / ค่ะ) and self-reference pronoun, the French tu / vous,
the German du / Sie, the Korean and Japanese polite ending, the Hebrew second-person gender, the Vietnamese
pronoun pair, the Slavic ty / vy. Generated and human cues are matched by time overlap; for every pair where the
human line shows a feature, the score counts whether the generated line shows the same value. Agreement per
feature, not per character: a line that says ค่ะ where the human said ครับ is one disagreement whatever the rest of
the words did. The regexes are deliberately plain and listed here so that a reader of the language can object.
"""
from __future__ import annotations

import re

_TH = r"(?:%s)"                  # Thai writes no spaces between words, so a pronoun is matched as a substring
_HE = r"(?<![\u05d0-\u05ea])(?:%s)(?![\u05d0-\u05ea])"
FEATURES: dict[str, dict[str, dict[str, str]]] = {
    "th": {"particle": {"m": r"ครับ|คร้าบ", "f": r"ค่ะ|คะ"},
           "self": {"phom": _TH % "ผม", "chan": _TH % "ฉัน|ดิฉัน", "nu": _TH % "หนู", "rao": _TH % "เรา", "ku": _TH % "กู|ข้า"},
           "address": {"khun": _TH % "คุณ", "thoe": _TH % "เธอ", "kae": _TH % "แก|มึง|นาย", "kin": _TH % "พ่อ|แม่|ลุง|ป้า|น้า|อา|ปู่|ย่า|ตา|ยาย"}},
    "fr": {"tv": {"tu": r"\b(?:tu|toi|te)\b|\bt'", "vous": r"\b(?:vous|votre|vos)\b"}},
    "de": {"tv": {"du": r"\b(?:du|dich|dir|dein|deine|deinen|deiner|deinem|deines)\b", "sie": r"\b(?:Sie|Ihnen|Ihr|Ihre|Ihren|Ihrer|Ihrem)\b"}},
    "es": {"tv": {"tu": r"\b(?:tú|te|ti|contigo)\b", "usted": r"\b(?:usted|ustedes)\b"}},
    "it": {"tv": {"tu": r"\b(?:tu|te|ti)\b", "lei": r"\b(?:Lei|Le)\b"}},
    "ko": {"polite": {"polite": r"(?:요|습니다|니다|세요|십시오|습니까)[.?!…\"'」]*$", "plain": r"(?:다|야|어|아|지|니|냐|네|자|군|라|걸)[.?!…\"'」]*$"}},
    "ja": {"polite": {"polite": r"(?:です|ます|ました|ません|でした|ましょう)[。？！…」]*$", "plain": r"(?:だ|だよ|だね|だろ|た|ない|る|う|よ|ね|ぞ|ぜ|か|さ|の|わ)[。？！…」]*$"},
           "self": {"boku": r"僕|ぼく", "ore": r"俺|おれ", "watashi": r"私|わたし", "atashi": r"あたし"}},
    "he": {"you": {"m": _HE % "אתה", "f": _HE % "את"}},
    "vi": {"pronoun": {"anh": r"\b[Aa]nh\b", "em": r"\b[Ee]m\b", "chi": r"\b[Cc]hị\b", "toi": r"\b[Tt]ôi\b", "cau": r"\b(?:[Cc]ậu|[Tt]ớ)\b",
                       "may": r"\b(?:[Mm]ày|[Tt]ao)\b", "con": r"\b[Cc]on\b", "bo": r"\b(?:[Bb]ố|[Mm]ẹ|[Bb]a|[Mm]á)\b", "ong": r"\b(?:[Ôô]ng|[Bb]à|[Cc]háu)\b"}},
    "cs": {"tv": {"ty": r"\b(?:ty|tě|tebe|ti|tobě|tvůj|tvoje|tvá|tvé)\b", "vy": r"\b(?:vy|vás|vám|váš|vaše)\b"}},
    "pl": {"tv": {"ty": r"\b(?:ty|ciebie|cię|tobie|ci|twój|twoja|twoje)\b", "vy": r"\b(?:pan|pani|państwo|pana|panu|panią)\b"}},
    "ru": {"tv": {"ty": r"\b(?:ты|тебя|тебе|тобой|твой|твоя|твоё|твои)\b", "vy": r"\b(?:вы|вас|вам|вами|ваш|ваша|ваше|ваши)\b"}},
    "uk": {"tv": {"ty": r"\b(?:ти|тебе|тобі|тобою|твій|твоя|твоє|твої)\b", "vy": r"\b(?:ви|вас|вам|вами|ваш|ваша|ваше|ваші)\b"}},
    "el": {"tv": {"esy": r"\b(?:εσύ|σε|σου|σένα)\b", "eseis": r"\b(?:εσείς|σας)\b"}},
    "hu": {"tv": {"te": r"\b(?:te|téged|neked|tied|tiéd)\b", "on": r"\b(?:ön|önt|önnek|maga|magát|magának)\b"}},
    "hr": {"tv": {"ti": r"\b(?:ti|tebe|tebi|tvoj|tvoja|tvoje)\b", "vi": r"\b(?:vi|vas|vama|vaš|vaša|vaše)\b"}},
}
_CASE_SENSITIVE = {("de", "tv"), ("it", "tv"), ("vi", "pronoun")}


def features_of(text: str, lang: str) -> dict[str, set[str]]:
    """The register features a line shows, per class: {"tv": {"tu"}, ...}. A class with both values on one line (a
    quoted exchange) keeps both; the comparison treats that as agreement with either."""
    spec = FEATURES.get(lang)
    if not spec:
        return {}
    flat = text.replace("\n", " ").strip()
    out: dict[str, set[str]] = {}
    for cls, values in spec.items():
        target = flat.split("/")[-1].strip() if (lang in ("ko", "ja") and cls == "polite") else flat
        flags = 0 if (lang, cls) in _CASE_SENSITIVE else re.IGNORECASE
        for val, pat in values.items():
            if re.search(pat, target, flags):
                out.setdefault(cls, set()).add(val)
    return out


def _text(c) -> str:
    t = getattr(c, "text", None)
    return t if t is not None else getattr(c, "ja", "")


def match_cues(ref, hyp, min_overlap: float = 0.2) -> list[tuple]:
    """(reference cue, generated cue) pairs by greatest time overlap."""
    hyp = sorted(hyp, key=lambda c: c.start)
    pairs = []
    for r in sorted(ref, key=lambda c: c.start):
        best, best_ov = None, 0.0
        for h in hyp:
            if h.start > r.end:
                break
            ov = min(h.end, r.end) - max(h.start, r.start)
            if ov > best_ov:
                best, best_ov = h, ov
        if best is not None and best_ov > min_overlap:
            pairs.append((r, best))
    return pairs


def register_agreement(ref_cues, hyp_cues, lang: str) -> dict:
    """Per feature class: how often the generated line shows the same value as the human line, over the pairs where
    the human line shows one; and how often it shows none at all. {"lang", "pairs", "classes": {cls: {...}}}."""
    pairs = match_cues(ref_cues, hyp_cues)
    classes: dict[str, dict] = {}
    for r, h in pairs:
        fr, fh = features_of(_text(r), lang), features_of(_text(h), lang)
        for cls, rv in fr.items():
            d = classes.setdefault(cls, {"n": 0, "agree": 0, "absent": 0})
            d["n"] += 1
            hv = fh.get(cls)
            if not hv:
                d["absent"] += 1
            elif rv & hv:
                d["agree"] += 1
    for d in classes.values():
        d["rate"] = round(100 * d["agree"] / d["n"], 1) if d["n"] else None
        d["absent_rate"] = round(100 * d["absent"] / d["n"], 1) if d["n"] else None
    return {"lang": lang, "pairs": len(pairs), "classes": classes}


def print_register(rows: list[dict], title: str = "") -> None:
    if title:
        print(title)
    print(f"   {'lang':<5} {'feature':<10} {'human':>6} {'agree':>6} {'rate%':>6} {'absent%':>8}   (pairs)")
    for r in rows:
        for cls, d in sorted(r["classes"].items()):
            print(f"   {r['lang']:<5} {cls:<10} {d['n']:>6} {d['agree']:>6} {str(d['rate']):>6} {str(d['absent_rate']):>8}   ({r['pairs']})")
