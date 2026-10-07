"""The character sheet (0.5.1, 2026-10-04): who speaks in a film, and how they address each other in the target
language — decided once, fed to every translation window.

Why: a window of twenty cues cannot know that the woman in this scene is the man's daughter, or that the speaker of
line 430 is male, so each window guesses. In a gendered language the guess shows in every verb and adjective (a
father answering in the feminine), in a language with a register system it shows in the pronouns and particles
(a ten-year-old calling her father vous, a Thai child switching between หนู and ผม), and the guesses differ from window
to window. Measured 2026-10-04 on one film in 44 languages: both translators misgendered speakers and the
translation-only one defaulted to formal address.

Two steps, cached in the work file like the terms:
  1. the sheet — a chat model reads the transcript and lists the recurring characters: name or label, gender,
     age group, role, and their relationships and relative status to each other; JSON.
  2. the rendering — per target, the same model turns the sheet into rules for that language: each character's
     grammatical gender for their own speech, how they refer to themselves, how they address each other
     character, speech level / particles. Plain lines, short.
The rendering goes into the prompt as CHARACTERS beside the GLOSSARY. Off with --register off, or for a clip too
short to have recurring characters. TranslateGemma's fixed prompt has no room for it (its formality is part of
why TranslateGemma is routed only where the 31B invents words).
"""
from __future__ import annotations

import json
import re
import sys

from . import config


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


FACTS_NOTE = ("\n\nKNOWN FACTS about this title, from public sources (priors — the dialogue below wins where it contradicts "
              "them; use them for names, who plays whom, relationships and genders that the dialogue leaves unsaid):\n{facts}\n")
SHEET_PROMPT = ("Below is the dialogue of {genre}, in {language}.{voices_note}{facts_note} List the recurring characters who speak "
                "(at most 8), one entry per person — if the same person is called both by a name and by a label "
                "(\"Dad\", \"the father\", \"Will\"), list them ONCE under the name and put the other forms in "
                "\"aliases\". For each give: \"name\", \"aliases\" (a list, may be empty), \"gender\" (\"m\", \"f\" or "
                "\"?\"), \"age\" (\"child\", \"teen\", \"adult\", \"elderly\" or \"unknown\"), \"role\" (a few words), "
                "\"evidence\" (one or two short quotes from the dialogue that establish who they are — a vocative, a "
                "self-description, another character's words), and \"relations\": a list of {{\"to\": <other "
                "character's name>, \"relation\": <e.g. \"father\", \"daughter\", \"wife\", \"boss\", \"employee\", "
                "\"friend\", \"stranger\">, \"status\": <\"higher\", \"equal\" or \"lower\" — the speaker's standing "
                "relative to that person>, \"evidence\": <a short quote>}}. State only what the dialogue itself shows "
                "or makes obvious; where it does not, write \"?\" for gender, \"unknown\" for age, and leave the "
                "relation out rather than guess — a wrong guess here would be applied consistently to the whole "
                "film.{voices_ask} Answer with JSON only, no prose: an object {{\"characters\": [...]{voices_field}}}."
                "\n\n{text}")
VOICES_NOTE = (" Lines start with a voice tag like [S2] from automatic speaker detection: the same tag is the same "
               "voice throughout, a different tag a different voice; the tags say nothing about who the voice is. A "
               "tag is never a name: a character whose name is not spoken gets a label like \"the father\" or \"a "
               "teacher\", and a voice you cannot place is simply left out of \"voices\".")
VOICES_ASK = (" Also say which character each voice tag belongs to, where the dialogue makes it clear (the voice that "
              "is called \"Dad\" by another voice, the voice that introduces itself), and quote the line that shows it; "
              "leave a tag out when unsure — a wrong voice would misgender a character in every line.")
VOICES_FIELD = ", \"voices\": {\"S1\": {\"character\": <character name>, \"evidence\": <a short quote from the dialogue>}, ...}"

RENDER_PROMPT = ("Here are the characters of {genre} (source {source}), as JSON:\n{sheet}\n{voices_line}\n"
                 "Write the rules a {target} subtitle translator must follow so that every character sounds right and "
                 "consistent in {target}. For each character, one line: how they refer to themselves (the pronoun or "
                 "form of self-reference {target} uses for a person of that gender, age and standing), how they address "
                 "each of the others they talk to (pronoun, kin term, title or name, and the politeness level or "
                 "particles/verb forms that go with it), and the grammatical gender to use for their own speech where "
                 "{target} marks it. Be specific to {target}; if {target} makes no such distinctions, say so in one "
                 "line and give only the genders. Where {target} distinguishes an informal and a formal address (tu/vous, "
                 "du/Sie, 반말/존댓말, plain/polite forms): friends, partners, spouses, siblings, parents and children, "
                 "and colleagues who know each other well take the INFORMAL form; strangers, a superior addressed by a "
                 "subordinate, officials, a court, a customer take the FORMAL; a character the sheet lists with a "
                 "relation \"friend\" or \"partner\" or \"colleague\" and status \"equal\" is informal unless the setting "
                 "is official. Where {target} addresses people by kin terms or role terms (Thai, Vietnamese, Burmese, "
                 "Khmer, Indonesian, Malay, Tagalog, Korean kin terms), choose each pair's OWN terms by relationship and "
                 "relative age, never a polite default: a teacher and a pupil take the school pair (Vietnamese thầy/cô "
                 "and em; Thai ครู and หนู or เธอ), a parent and a child the family pair (bố/mẹ and con; พ่อ/แม่ and "
                 "ลูก or หนู), an unrelated adult and a child the uncle/aunt pair (chú/cô and cháu; ลุง/น้า and หนู), "
                 "two adults who are peers, rivals, partners or friends the casual pair of their sexes (Vietnamese "
                 "cậu/tớ or anh/em by age, mày/tao when rough; Thai นาย/ฉัน between men, เธอ/ฉัน otherwise, แก/ข้า "
                 "when rough); keep the polite forms (tôi and anh/chị with ạ; คุณ with ผม/ดิฉัน and ครับ/ค่ะ) for "
                 "strangers, officials, customers and superiors. Two calibrations: Polish keeps pan / pani between adults "
                 "who are not family or close friends, including long-standing colleagues, far longer than French keeps "
                 "vous; Greek similarly keeps εσείς between adult acquaintances and in any professional setting. Short "
                 "lines, no explanations, no JSON.")


def build_sheet(client, texts: list[str], lang: str, genre: str, tagged: bool = False, facts: str = "") -> tuple[list[dict], dict[str, str]]:
    """The characters as JSON, from the transcript (sampled evenly when it is long), and — when the lines carry voice
    tags — the voices mapped to characters. `facts`: the web context as priors (0.5.8), when the lookup is on.
    Returns (sheet, voices)."""
    text = "\n".join(texts)
    if len(text) > config.CHARACTERS_CHARS:
        step = len(texts) / (config.CHARACTERS_CHARS / 60)
        picked, k = [], 0.0
        while int(k) < len(texts):
            picked.append(texts[int(k)]); k += max(1.0, step)
        text = "\n".join(picked)[:config.CHARACTERS_CHARS]
    prompt = SHEET_PROMPT.format(genre=genre, language=config.LANG_NAMES.get(lang, lang), text=text,
                                 voices_note=VOICES_NOTE if tagged else "", voices_ask=VOICES_ASK if tagged else "",
                                 voices_field=VOICES_FIELD if tagged else "",
                                 facts_note=FACTS_NOTE.format(facts=facts) if facts else "")
    data = None
    for attempt, p in enumerate((prompt, None)):
        if p is None:                                    # the retry (0.5.19): the same question without the facts block,
            if not facts:                                #  for a smaller model that choked on the longer prompt
                break
            p = SHEET_PROMPT.format(genre=genre, language=config.LANG_NAMES.get(lang, lang), text=text,
                                    voices_note=VOICES_NOTE if tagged else "", voices_ask=VOICES_ASK if tagged else "",
                                    voices_field=VOICES_FIELD if tagged else "", facts_note="")
            _log("[characters] the sheet did not parse with the facts block — asking again without it")
        try:
            out = client.chat(None, p, max_tokens=1600)
        except Exception as e:                           # noqa: BLE001
            _log(f"[characters] ⚠ sheet call failed: {e}")
            return [], {}
        m = re.search(r"\{.*\}|\[.*\]", out, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                break
            except json.JSONDecodeError:
                data = None
    if data is None:
        return [], {}
    chars = data.get("characters") if isinstance(data, dict) else data
    voices_raw = data.get("voices") if isinstance(data, dict) else {}
    sheet = []
    for d in chars if isinstance(chars, list) else []:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        if re.fullmatch(r"\[?S\d{1,2}\]?", str(d.get("name")).strip()):      # a voice tag is not a name (night one, 2026-10-05)
            continue
        sheet.append({"name": str(d.get("name"))[:40],
                      "aliases": [str(a)[:40] for a in (d.get("aliases") or []) if isinstance(a, str)][:6],
                      "gender": str(d.get("gender", "?"))[:1].lower(), "age": str(d.get("age", "unknown"))[:8],
                      "role": str(d.get("role", ""))[:60],
                      "evidence": [str(e)[:120] for e in (d.get("evidence") or []) if isinstance(e, str)][:3],
                      "relations": [{"to": str(r.get("to", ""))[:40], "relation": str(r.get("relation", ""))[:30],
                                     "status": str(r.get("status", "equal"))[:6], "evidence": str(r.get("evidence", ""))[:120]}
                                    for r in (d.get("relations") or []) if isinstance(r, dict)][:8]})
    names = {c["name"] for c in sheet} | {a for c in sheet for a in c["aliases"]}
    # evidence-gated (0.5.11): a voice maps to a character only with a quoted line behind it and a name the sheet
    # knows; a bare name, or an entry without evidence, is dropped — the translator then infers the speaker from the
    # dialogue for that voice, as it does without labels, rather than being told something unsupported
    voices: dict[str, dict] = {}
    for k, v in (voices_raw or {}).items() if isinstance(voices_raw, dict) else []:
        if not re.fullmatch(r"S\d{1,2}", str(k)):
            continue
        if isinstance(v, dict):
            name, ev = str(v.get("character", ""))[:40], str(v.get("evidence", "")).strip()[:160]
        else:
            name, ev = str(v)[:40], ""
        if name in names and ev:
            voices[str(k)] = {"character": name, "evidence": ev}
    return sheet[:8], voices


def render_sheet(client, sheet: list[dict], source: str, target: str, genre: str, voices: dict[str, str] | None = None) -> str:
    """The sheet as address rules in `target`; empty when the call fails or the sheet is empty. With voices, the
    rules start with the voice-to-character lines so the translator reads a tag as a person."""
    if not sheet:
        return ""
    slim = [{k: v for k, v in c.items() if k != "evidence"} for c in sheet]        # the quotes are for `why`, not the prompt
    for c in slim:
        c["relations"] = [{k: v for k, v in r.items() if k != "evidence"} for r in c.get("relations", [])]
    voices_line = ("Voice tags in the dialogue: " + "; ".join(f"[{k}] is {_who(v)}" for k, v in sorted(voices.items())) + ".\n") if voices else ""
    prompt = RENDER_PROMPT.format(genre=genre, source=config.LANG_NAMES.get(source, source),
                                  target=config.LANG_NAMES.get(target, target), sheet=json.dumps(slim, ensure_ascii=False),
                                  voices_line=voices_line)
    try:
        out = client.chat(None, prompt, max_tokens=900)
    except Exception as e:                               # noqa: BLE001
        _log(f"[characters] ⚠ rendering call failed: {e}")
        return ""
    lines = [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith("```")]
    head = [f"[{k}] is {_who(v)}" for k, v in sorted((voices or {}).items())]
    return "\n".join(head + lines)[:2800]


def _who(v) -> str:
    """The character behind a voice entry — the 0.5.11 dict form or the earlier bare name."""
    return v.get("character", "") if isinstance(v, dict) else str(v)


def summarise(sheet: list[dict], voices: dict | None = None) -> str:
    s = ", ".join(f"{c['name']} ({c['gender']}, {c['age']})" for c in sheet)
    if voices:
        s += " — voices: " + ", ".join(f"{k}={_who(v)}" for k, v in sorted(voices.items()))
    return s
