"""Series context (0.6.2, 2026-10-07): what one episode learnt, the next episode starts from.

Three Shin-chan episodes in a row named the family three different ways (Shin-chan / Hiroshi / Misae; no sheet at
all; しんのすけ / ミサエ / 父ちゃん), because every episode built its own character sheet from its own dialogue. A series
has the same characters every week; the sheet and the glossary should carry forward. This is local and needs no
network: the series is identified from the file name (lookup.identify), the carried state lives under
MLSUBGEN_HOME/context/series/<slug>.json, and it is offered to the next episode's sheet builder as KNOWN
CHARACTERS and to its terms pass as rendered names. The dialogue still wins where it contradicts the carried
state; a character seen in a later episode is added, never removed.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config


def _dir() -> Path:
    home = Path(getattr(config, "MLSUBGEN_HOME", None) or (Path.home() / "mlsubgen"))
    d = home / "context" / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def key_for(info: dict) -> str | None:
    """The series slug from lookup.identify()'s result, or None for a film / an unidentified file."""
    if info.get("kind") != "episode" or not info.get("series"):
        return None
    return re.sub(r"[^a-z0-9]+", "-", info["series"].lower()).strip("-")[:80] or None


def load(key: str) -> dict:
    p = _dir() / f"{key}.json"
    if not p.is_file():
        return {"series": key, "characters": [], "glossary": {}, "episodes": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {"series": key, "characters": [], "glossary": {}, "episodes": []}


def save(state: dict) -> None:
    p = _dir() / f"{state['series']}.json"
    p.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def facts_text(state: dict) -> str:
    """The carried characters as a block for the sheet builder's KNOWN FACTS."""
    chars = state.get("characters") or []
    if not chars:
        return ""
    lines = []
    for c in chars[:12]:
        bits = [c.get("gender", "?"), c.get("age", "unknown")]
        if c.get("role"):
            bits.append(c["role"])
        rel = "; ".join(f"{r.get('relation')} of {r.get('to')}" for r in (c.get("relations") or [])[:3] if r.get("to"))
        al = ", ".join(c.get("aliases") or [])
        lines.append(f"{c['name']} ({', '.join(bits)})" + (f"; {rel}" if rel else "") + (f"; also called {al}" if al else ""))
    return ("Recurring characters of this series, from earlier episodes. Use these names SPELT EXACTLY AS GIVEN here "
            "(Latin letters, not the source script) for anyone who is one of them, keep their genders, and add only "
            "what this episode shows:\n" + "\n".join(lines))


def merge_sheet(state: dict, sheet: list[dict], episode: str) -> dict:
    """Add this episode's characters to the carried state: a name already known keeps its first-seen gender, age
    and role and gains aliases and relations it lacked; a new name is appended."""
    by_name = {c["name"].lower(): c for c in state.get("characters", [])}
    alias_to = {a.lower(): c for c in state.get("characters", []) for a in (c.get("aliases") or [])}
    for s in sheet or []:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        # the name as the dialogue's own script writes it (0.6.2.3): an episode's sheet may call the boy しんのすけ
        # where an earlier one said Shin-chan; the source-script form is what both share, so it is matched and kept
        src_name = (s.get("source_name") or "").strip()
        known = by_name.get(name.lower()) or alias_to.get(name.lower()) or (alias_to.get(src_name.lower()) if src_name else None)
        if known is None:
            entry = {"name": name, "aliases": list(s.get("aliases") or []), "gender": s.get("gender", "?"), "age": s.get("age", "unknown"),
                     "role": s.get("role", ""), "relations": [{"to": r.get("to"), "relation": r.get("relation")} for r in (s.get("relations") or []) if r.get("to")]}
            if src_name and src_name.lower() != name.lower():
                entry["aliases"].append(src_name)
            state.setdefault("characters", []).append(entry)
            by_name[name.lower()] = entry
            for a in entry["aliases"]:
                alias_to[a.lower()] = entry
            continue
        for a in list(s.get("aliases") or []) + ([src_name] if src_name else []) + ([name] if name.lower() != known["name"].lower() else []):
            if a and a.lower() not in {x.lower() for x in known.get("aliases", [])} and a.lower() != known["name"].lower():
                known.setdefault("aliases", []).append(a)
                alias_to[a.lower()] = known
        if known.get("gender", "?") == "?" and s.get("gender", "?") != "?":
            known["gender"] = s["gender"]
        if known.get("age", "unknown") == "unknown" and s.get("age", "unknown") != "unknown":
            known["age"] = s["age"]
        have = {(r.get("to"), r.get("relation")) for r in known.get("relations", [])}
        for r in s.get("relations") or []:
            if r.get("to") and (r.get("to"), r.get("relation")) not in have:
                known.setdefault("relations", []).append({"to": r["to"], "relation": r.get("relation")})
    if episode and episode not in state.setdefault("episodes", []):
        state["episodes"].append(episode)
    state["updated"] = time.strftime("%Y-%m-%d %H:%M")
    return state


def merge_glossary(state: dict, target: str, rendered: dict[str, str]) -> dict:
    """Carry a target's rendered names forward: the first rendering of a name wins, so the spelling is the same in
    every episode."""
    g = state.setdefault("glossary", {}).setdefault(target, {})
    for k, v in (rendered or {}).items():
        if k and v and k not in g:
            g[k] = v
    return state


def glossary_for(state: dict, target: str) -> dict[str, str]:
    return dict((state.get("glossary") or {}).get(target) or {})
