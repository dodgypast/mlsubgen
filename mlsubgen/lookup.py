"""Media context from the web (0.5.8, 2026-10-05) — OPT-IN.

What is publicly known about a title is a third source of evidence beside the file and the audio: the cast and
who plays whom, the characters' relationships, the title in every language, the names a dub or a fan base
actually uses. The character sheet and the terms pass start from those facts instead of from a model's reading of
the dialogue, which is where the misgendering of a father in six languages came from. The dialogue still wins
where it contradicts the web: these are priors, with their sources kept.

Off unless `--web-context auto` (or MLSUBGEN_WEB_CONTEXT=auto). Then, in order, cheapest and most structured
first, each stopping when it has enough:
  1. Wikipedia (REST summary + the article's cast section; the interlanguage links are the localised titles)
  2. Wikidata (the item: instance of film / series, year — to confirm the identification)
  3. SearXNG at MLSUBGEN_SEARXNG_URL, the keyless engines (franchise wikis, episode pages)
  4. the Brave Search API with BRAVE_API_KEY, only when SearXNG returned too little, under a daily cap
Only the title and search terms leave the machine, through your own SearXNG where configured; never the media,
the audio or the subtitle text. Every outbound query is recorded in the work file and shown by `why`. Results
are cached under MLSUBGEN_HOME/context by title (series) and by episode, so a long series is looked up once.

Identification is the hard part: the title, year, season and episode are parsed from the file name and its
folders; a match that cannot be confirmed (no year, no episode, no Wikipedia hit of the right kind) gets a low
confidence and the context is NOT used — a wrong film would poison every later decision.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import config


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


UA = {"User-Agent": "mlsubgen/0.5 (https://github.com/dodgypast/mlsubgen; local subtitle tool)"}


def _home() -> Path:
    return Path(getattr(config, "HOME", None) or os.environ.get("MLSUBGEN_HOME") or (Path.home() / "mlsubgen"))


def enabled(job_setting: str | None) -> bool:
    return (job_setting or os.environ.get("MLSUBGEN_WEB_CONTEXT", "off")).lower() == "auto"
_RELEASE_TAGS = re.compile(r"\b(2160p|1080p|720p|480p|4k|uhd|bluray|blu-ray|bdrip|brrip|web-?dl|webrip|hdtv|amzn|nf|dsnp|hmax|atvp|"
                           r"ddp?[57]\.1|dd[57]\.1|aac|ac3|dts(-hd)?|truehd|atmos|x26[45]|h\.?26[45]|hevc|avc|10bit|hdr10\+?|hdr|dv|dovi|"
                           r"remux|hybrid|proper|repack|extended|unrated|dubbed|multi|sdr|ma|7\.1|5\.1)\b.*$", re.I)


def identify(video: Path) -> dict:
    """Title / year / series / season / episode from the file name and its folders, with a confidence."""
    name = video.stem
    info: dict = {"file": video.name}
    # [Group] Series - 0064 [720p]…   (fansub style)
    m = re.match(r"^\[[^\]]+\]\s*(.+?)\s*-\s*(\d{1,4})\b", name)
    if m:
        info.update(series=m.group(1).strip(), episode=int(m.group(2)), kind="episode")
    # Series.S01E02.Title… / Series S01E02
    m = re.search(r"^(.*?)[\s._-]+S(\d{1,2})E(\d{1,3})\b", name, re.I)
    if m and "series" not in info:
        info.update(series=re.sub(r"[._]+", " ", m.group(1)).strip(" -"), season=int(m.group(2)), episode=int(m.group(3)), kind="episode")
    # Title.2008.… / Title (2008) / Title 2008
    m = re.search(r"^(.*?)[\s._(-]+((?:19|20)\d{2})\b", name)
    if m and "series" not in info:
        info.update(title=re.sub(r"[._]+", " ", m.group(1)).strip(" -("), year=int(m.group(2)), kind="film")
    if "title" not in info and "series" not in info:
        clean = _RELEASE_TAGS.sub("", re.sub(r"[._]+", " ", name)).strip(" -")
        info.update(title=clean, kind="film")
    if info.get("kind") == "episode" and "series" in info and "season" not in info:
        # the season from a parent folder ("Season 2") when the name has only an absolute episode number
        for p in video.parents:
            m = re.match(r"season\s*(\d+)", p.name, re.I)
            if m:
                info["season_folder"] = int(m.group(1)); break
    for k in ("title", "series"):
        if k in info:
            info[k] = _RELEASE_TAGS.sub("", info[k]).strip(" -")
    info["confidence"] = 0.8 if ("year" in info or "episode" in info) else 0.4
    return info


def _get(url: str, timeout: float = 12.0, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _json(url: str, timeout: float = 12.0, headers: dict | None = None) -> dict:
    return json.loads(_get(url, timeout, headers).decode("utf-8", "replace"))


# ── Wikipedia / Wikidata ─────────────────────────────────────────────────────────────────────────────────────
def wikipedia(info: dict, targets: list[str], log: list[dict]) -> dict:
    """The article's summary, its cast section and its titles in the target languages."""
    name = info.get("title") or info.get("series") or ""
    if not name:
        return {}
    q = name + (f" {info['year']} film" if info.get("year") else " television series" if info.get("kind") == "episode" else " film")
    out: dict = {}
    try:
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": 3})
        log.append({"source": "wikipedia", "url": url})
        hits = _json(url)["query"]["search"]
        if not hits:
            return {}
        page = hits[0]["title"]
        out["page"] = page
        s = _json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(page.replace(" ", "_")))
        out["summary"] = (s.get("extract") or "")[:1200]
        out["description"] = s.get("description", "")
        # the cast section as plain text: "Actor as Character"
        sec = _json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "parse", "page": page, "prop": "sections", "format": "json"}))
        idx = next((x["index"] for x in sec["parse"]["sections"] if re.search(r"^cast|characters|voice cast", x["line"], re.I)), None)
        if idx:
            html = _json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "parse", "page": page, "section": idx, "prop": "text", "format": "json"}))["parse"]["text"]["*"]
            text = re.sub(r"<[^>]+>", "", html)
            cast = [m.strip() for m in re.findall(r"([^\n]{3,80}? as [^\n]{2,120})", text)]
            out["cast"] = cast[:25]
        # localised titles
        ll = _json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "titles": page, "prop": "langlinks", "lllimit": 200, "format": "json"}))
        pages = list(ll["query"]["pages"].values())
        titles = {}
        for x in (pages[0].get("langlinks") or []) if pages else []:
            code = {"zh": "zh", "zh-yue": "yue", "nb": "no", "nn": "no", "ms": "ms", "id": "id", "tl": "tl", "fil": "tl", "my": "my", "km": "km", "lo": "lo"}.get(x["lang"], x["lang"])
            if code in targets:
                titles[code] = re.sub(r"\s*\([^)]*\)\s*$", "", x["*"])      # "クレヨンしんちゃん (アニメ)" → the title alone
        out["titles"] = titles
    except Exception as e:                               # noqa: BLE001
        log.append({"source": "wikipedia", "error": str(e)[:120]})
    return out


# ── SearXNG and Brave ────────────────────────────────────────────────────────────────────────────────────────
def searxng(query: str, log: list[dict], n: int = 6) -> list[dict]:
    base = os.environ.get("MLSUBGEN_SEARXNG_URL", "").rstrip("/")
    if not base:
        return []
    url = f"{base}/search?" + urllib.parse.urlencode({"q": query, "format": "json", "categories": "general"})
    log.append({"source": "searxng", "query": query})
    try:
        res = _json(url).get("results", [])
    except Exception as e:                               # noqa: BLE001
        log.append({"source": "searxng", "error": str(e)[:120]}); return []
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:300], "engine": r.get("engine", "")}
            for r in res[:n]]


def _brave_budget_ok() -> bool:
    cap = int(os.environ.get("MLSUBGEN_BRAVE_DAILY_CAP", "100"))
    f = _home() / "context" / "brave-count.json"
    today = time.strftime("%Y-%m-%d")
    try:
        d = json.loads(f.read_text()) if f.is_file() else {}
    except Exception:                                    # noqa: BLE001
        d = {}
    used = d.get(today, 0)
    if used >= cap:
        return False
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({today: used + 1}))
    return True


def brave(query: str, log: list[dict], n: int = 6) -> list[dict]:
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key or not _brave_budget_ok():
        return []
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": n})
    log.append({"source": "brave", "query": query})
    try:
        res = _json(url, headers={"X-Subscription-Token": key, "Accept": "application/json"}).get("web", {}).get("results", [])
    except Exception as e:                               # noqa: BLE001
        log.append({"source": "brave", "error": str(e)[:120]}); return []
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("description") or "")[:300], "engine": "brave"} for r in res[:n]]


# ── the context ──────────────────────────────────────────────────────────────────────────────────────────────
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def media_context(video: Path, targets: list[str]) -> dict:
    """The cached or freshly gathered context for this file, with its identification, facts and provenance."""
    info = identify(video)
    name = info.get("title") or info.get("series") or ""
    if not name:
        return {"identity": info, "used": False, "why": "no title in the file name"}
    cache_dir = _home() / "context" / "media"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _slug(f"{name}-{info.get('year', '')}") if info.get("kind") == "film" else _slug(name)
    f = cache_dir / f"{key}.json"
    if f.is_file():
        try:
            ctx = json.loads(f.read_text(encoding="utf-8"))
            ctx["identity"] = info
            ctx["cached"] = True
            return ctx
        except Exception:                                # noqa: BLE001
            pass
    log: list[dict] = []
    wp = wikipedia(info, targets, log)
    used = bool(wp.get("page")) and info["confidence"] >= 0.6
    # the kind must match: a film name that hits a novel or a band is not this film
    desc = (wp.get("description") or "").lower()
    if used and desc and not re.search(r"film|movie|series|anime|television|tv|show|documentary", desc):
        used = False
    results: list[dict] = []
    if used and (not wp.get("cast") or len(wp.get("cast", [])) < 3):
        q = f"{name} {info.get('year', '')} characters cast relationships".strip()
        results = searxng(q, log)
        if len(results) < 3:
            results += brave(q, log)
    ctx = {"identity": info, "used": used, "wikipedia": wp, "results": results, "queries": log,
           "gathered": time.strftime("%Y-%m-%d %H:%M"), "cached": False,
           "why": "" if used else ("no Wikipedia match" if not wp.get("page") else f"match of the wrong kind ({desc})" if desc else "low identification confidence")}
    try:
        f.write_text(json.dumps(ctx, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return ctx


def facts_text(ctx: dict) -> str:
    """The context as a short block of facts for the character sheet's prompt (structured, no web prose)."""
    if not ctx or not ctx.get("used"):
        return ""
    wp = ctx.get("wikipedia") or {}
    parts = []
    if wp.get("page"):
        parts.append(f"Title: {wp['page']}" + (f" — {wp['description']}" if wp.get("description") else ""))
    if wp.get("summary"):
        parts.append("Synopsis: " + wp["summary"][:700])
    if wp.get("cast"):
        parts.append("Cast (actor as character): " + "; ".join(wp["cast"][:15]))
    for r in (ctx.get("results") or [])[:4]:
        if r.get("snippet"):
            parts.append(f"From {r['url']}: {r['snippet']}")
    return "\n".join(parts)[:2500]
