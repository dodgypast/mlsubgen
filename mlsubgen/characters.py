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


SHEET_PROMPT = ("Below is the dialogue of {genre}, in {language}. List the recurring characters who speak (at most 8). "
                "For each give: \"name\" (the name used in the dialogue, or a label like \"the father\" if no name is "
                "spoken), \"gender\" (\"m\", \"f\" or \"?\"), \"age\" (\"child\", \"teen\", \"adult\" or \"elderly\"), "
                "\"role\" (a few words), and \"relations\": a list of {{\"to\": <other character's name>, "
                "\"relation\": <e.g. \"father\", \"daughter\", \"wife\", \"boss\", \"employee\", \"friend\", "
                "\"stranger\">, \"status\": <\"higher\", \"equal\" or \"lower\" — the speaker's standing relative to "
                "that person>}}. State only what the dialogue itself shows or makes obvious; where it does not, write "
                "\"?\" for gender, \"unknown\" for age, and leave the relation out rather than guess — a wrong guess "
                "here would be applied consistently to the whole film. Answer with a JSON array only, no prose.\n\n{text}")

RENDER_PROMPT = ("Here are the characters of {genre} (source {source}), as JSON:\n{sheet}\n\n"
                 "Write the rules a {target} subtitle translator must follow so that every character sounds right and "
                 "consistent in {target}. For each character, one line: how they refer to themselves (the pronoun or "
                 "form of self-reference {target} uses for a person of that gender, age and standing), how they address "
                 "each of the others they talk to (pronoun, kin term, title or name, and the politeness level or "
                 "particles/verb forms that go with it), and the grammatical gender to use for their own speech where "
                 "{target} marks it. Be specific to {target}; if {target} makes no such distinctions, say so in one "
                 "line and give only the genders. Short lines, no explanations, no JSON.")


def build_sheet(client, texts: list[str], lang: str, genre: str) -> list[dict]:
    """The characters as JSON, from the transcript (sampled evenly when it is long)."""
    text = "\n".join(texts)
    if len(text) > config.CHARACTERS_CHARS:
        step = len(texts) / (config.CHARACTERS_CHARS / 60)
        picked, k = [], 0.0
        while int(k) < len(texts):
            picked.append(texts[int(k)]); k += max(1.0, step)
        text = "\n".join(picked)[:config.CHARACTERS_CHARS]
    prompt = SHEET_PROMPT.format(genre=genre, language=config.LANG_NAMES.get(lang, lang), text=text)
    try:
        out = client.chat(None, prompt, max_tokens=1200)
    except Exception as e:                               # noqa: BLE001
        _log(f"[characters] ⚠ sheet call failed: {e}")
        return []
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    sheet = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        sheet.append({"name": str(d.get("name"))[:40], "gender": str(d.get("gender", "?"))[:1].lower(),
                      "age": str(d.get("age", "adult"))[:8], "role": str(d.get("role", ""))[:60],
                      "relations": [{"to": str(r.get("to", ""))[:40], "relation": str(r.get("relation", ""))[:30],
                                     "status": str(r.get("status", "equal"))[:6]}
                                    for r in (d.get("relations") or []) if isinstance(r, dict)][:8]})
    return sheet[:8]


def render_sheet(client, sheet: list[dict], source: str, target: str, genre: str) -> str:
    """The sheet as address rules in `target`; empty when the call fails or the sheet is empty."""
    if not sheet:
        return ""
    prompt = RENDER_PROMPT.format(genre=genre, source=config.LANG_NAMES.get(source, source),
                                  target=config.LANG_NAMES.get(target, target), sheet=json.dumps(sheet, ensure_ascii=False))
    try:
        out = client.chat(None, prompt, max_tokens=900)
    except Exception as e:                               # noqa: BLE001
        _log(f"[characters] ⚠ rendering call failed: {e}")
        return ""
    lines = [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith("```")]
    return "\n".join(lines)[:2500]


def summarise(sheet: list[dict]) -> str:
    return ", ".join(f"{c['name']} ({c['gender']}, {c['age']})" for c in sheet)
