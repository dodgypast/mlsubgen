"""LLM translation with rolling context, from any source language to any target. Works against Ollama's native
API or any OpenAI-compatible server. Which model translates which language pair comes from TRANSLATE_ROUTES;
a cue already in the target language is copied through, not translated."""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config
from .config import Translator
from .lid import script_matches, script_of
from .segment import Cue

_LINE = re.compile(r"^\s*\[?(\d+)\]?\s*(?:[\t.．:：)）\-–—]|\s)\s*(.*?)\s*$")
_TN = re.compile(r"\s*\((?:TN|T/N|Note|Translator'?s? note)[^)]*\)\s*$", re.I)
_FILLERS_EN = re.compile(r"\b(?:um+|uh+|erm+|hmm+|mm+|ah+|uh-huh)\b[,.]?\s*", re.I)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def lang_name(code: str) -> str:
    return config.LANG_NAMES.get(code, code)


def route(src: str, tgt: str, force: str | None = None) -> str:
    """The translator preset for a language pair: -t forces one; else the most specific TRANSLATE_ROUTES entry."""
    if force:
        return force
    r = config.TRANSLATE_ROUTES
    for key in ((src, tgt), ("*", tgt), (src, "*"), ("*", "*")):
        if key in r:
            return r[key]
    raise KeyError(f"no translator route for {src}→{tgt}")


def copy_through(text: str, lang: str) -> str:
    """A cue already in the target language: cleaned, not translated (fillers out, spacing normalised)."""
    s = re.sub(r"\s+", " ", text).strip()
    if lang == "en":
        s = _FILLERS_EN.sub("", s).strip()
        s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    retries: int = 0
    fallbacks: int = 0
    aborts: int = 0          # server-side prediction aborts (repeat-limit loops) recovered by re-sampling/splitting

    def summary(self) -> str:
        tps = self.completion_tokens / self.seconds if self.seconds else 0.0
        return (f"{self.requests} requests, {self.prompt_tokens} prompt + {self.completion_tokens} completion tokens, "
                f"{self.seconds:.0f}s ({tps:.1f} tok/s), retries {self.retries}, per-line fallbacks {self.fallbacks}"
                f"{', aborts recovered ' + str(self.aborts) if self.aborts else ''}")


class PredictionAborted(RuntimeError):
    """The server killed the prediction itself (Ollama: 'prediction aborted, token repeat limit reached' — the
    model fell into a repetition loop). Retrying the identical request reproduces it; the caller must re-sample
    (different seed/temperature) or shrink the window instead."""


# extra sampling used when a window is re-tried after an abort: a fresh seed and a little more entropy shake
# the model out of the loop; the repeat penalty makes the loop itself less likely.
_NUDGE = {"temperature": 0.3, "repeat_penalty": 1.15, "repeat_last_n": 128}


class LLMClient:
    def __init__(self, tr: Translator, url: str | None = None, backend: str | None = None,
                 timeout: int = config.LLM_TIMEOUT_SEC):
        self.tr = tr
        self.backend = backend or tr.backend or config.LLM_BACKEND
        self.url = (url or tr.url or config.LLM_URL).rstrip("/")
        self.timeout = timeout
        self.usage = Usage()

    # ── transport ────────────────────────────────────────────────────────────────────────────────────
    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if "prediction aborted" in body or "repeat limit" in body:
                    self.usage.aborts += 1
                    raise PredictionAborted(f"HTTP {e.code} from {self.url}{path}: {body}") from None
                last = RuntimeError(f"HTTP {e.code} from {self.url}{path}: {body}")
                if e.code in (400, 404, 422):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = e
            self.usage.retries += 1
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last}")

    def available(self) -> tuple[bool, str]:
        try:
            if self.backend == "ollama":
                req = urllib.request.Request(self.url + "/api/tags")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    tags = json.loads(resp.read().decode("utf-8"))
                names = {m.get("name", "") for m in tags.get("models", [])}
                have = self.tr.model in names or (self.tr.model + ":latest") in names
                return (have, "ok" if have else f"model {self.tr.model!r} not pulled — run: ollama pull {self.tr.model}")
            req = urllib.request.Request(self.url + "/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            hint = "  (sudo systemctl start ollama ?)" if self.backend == "ollama" else ""
            return False, f"{self.backend} at {self.url} unreachable: {e}{hint}"

    def chat(self, system: str | None, user: str, max_tokens: int = 4096, nudge: bool = False) -> str:
        """One chat completion. nudge=True re-samples with a fresh seed, more temperature and a repeat penalty —
        used only to retry a window the server aborted for looping."""
        t0 = time.time()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        if self.backend == "ollama":
            options = {"temperature": self.tr.temperature, "num_ctx": self.tr.num_ctx, "num_predict": max_tokens,
                       **self.tr.extra_options}
            if nudge:
                options.update({**_NUDGE, "temperature": max(self.tr.temperature, 0.0) + _NUDGE["temperature"],
                                "seed": int(time.time() * 1000) % 2_000_000_000})
            payload = {
                "model": self.tr.model, "messages": messages, "stream": False, "think": bool(self.tr.think),
                "keep_alive": "10m",
                "options": options,
            }
            r = self._post("/api/chat", payload)
            content = (r.get("message") or {}).get("content", "") or ""
            self.usage.prompt_tokens += int(r.get("prompt_eval_count") or 0)
            self.usage.completion_tokens += int(r.get("eval_count") or 0)
        else:
            payload = {"model": self.tr.model, "messages": messages, "temperature": self.tr.temperature,
                       "max_tokens": max_tokens, "stream": False}
            if not self.tr.think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            r = self._post("/v1/chat/completions", payload)
            content = ((r.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
            u = r.get("usage") or {}
            self.usage.prompt_tokens += int(u.get("prompt_tokens") or 0)
            self.usage.completion_tokens += int(u.get("completion_tokens") or 0)
        self.usage.requests += 1
        self.usage.seconds += time.time() - t0
        # strip any leaked <think> blocks
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
        return content.strip()

    def unload(self) -> None:
        """Ask Ollama to free the model now (the next model needs the whole card)."""
        if self.backend != "ollama":
            return
        try:
            self._post("/api/generate", {"model": self.tr.model, "keep_alive": 0})
        except Exception:  # noqa: BLE001
            pass


class ClientPool:
    """One LLMClient per translator preset, and the discipline that only one model is loaded at a time: use(name)
    unloads the previous model before handing out the next. Swaps cost ~30 s each, so callers group work by
    route (stage 2 does: target by target, file by file)."""

    def __init__(self, url: str | None = None, backend: str | None = None, presets: dict | None = None):
        self.url, self.backend = url, backend
        self.presets = presets or {}
        self.clients: dict[str, LLMClient] = {}
        self.aliases: dict[str, str] = {}      # a route's model that is not available → the model standing in for it
        self.current: str | None = None
        self.swaps = 0

    def client(self, name: str) -> LLMClient:
        if name not in self.clients:
            tr = self.presets.get(name) or config.TRANSLATORS.get(name)
            if tr is None:
                raise SystemExit(f"unknown translator {name!r}; presets: {', '.join(config.TRANSLATORS)}")
            self.clients[name] = LLMClient(tr, self.url, self.backend)
        return self.clients[name]

    def use(self, name: str) -> LLMClient:
        name = self.aliases.get(name, name)
        if self.current and self.current != name:
            self.clients[self.current].unload()
            self.swaps += 1
            _log(f"[tl] switching translator {self.current} → {name}")
        self.current = name
        return self.client(name)

    def unload_all(self) -> None:
        if self.current:
            self.clients[self.current].unload()
        self.current = None


# ── prompts ───────────────────────────────────────────────────────────────────────────────────────────
GENERIC_SYSTEM = """You are a professional {src}-to-{tgt} subtitle translator. The material is {genre}.
You receive numbered {src} subtitle lines and must translate each one into natural, idiomatic {tgt} for subtitles.

Rules:
- Output exactly one line per input number, formatted as `<number><TAB><{tgt} translation>`, and nothing else: no headers, no notes, no {src}, no blank lines.
- Keep every number. Never merge, split, reorder or skip lines. A fragment stays a fragment; an unclear line gets your best reading, never "...".
- Subtitle {tgt}: concise (aim for at most 80 characters per line), plain everyday register, no bracketed explanations. Match the speaker's register — neutral for narration, conversational for interviews and chat.
- Names, places and technical terms: follow the GLOSSARY exactly; {names} Leave honorifics out unless they carry meaning.
- Numbers, dates and units in figures. Keep sentence-final punctuation minimal {punct}.
- The CONTEXT and FOLLOWING lines exist only so you know who is speaking and what is being discussed. Do not output them."""

TARGET_RULES = {
    "th": ("Thai register: polite particles (ครับ/ค่ะ/คะ) and ผม/ดิฉัน ONLY where the speaker is actually being polite or "
           "formal. Children, family, friends, lovers and rude or angry speakers talk without them, using casual pronouns "
           "(ฉัน/เรา/แก/นาย/เธอ or the person's name) and particles like นะ/สิ/ล่ะ/เหรอ/วะ as the tone demands. Keep each "
           "character's speech level consistent across the whole film. Never soften rudeness or crudeness."),
    "zh": "Write Simplified Chinese (简体字) as used in mainland subtitles; keep 。and 、as the punctuation.",
    "yue": "Write Cantonese in Traditional characters (繁體字) as Hong Kong subtitles do, in spoken Cantonese wording, not written Mandarin.",
    "ja": "Natural spoken Japanese for subtitles: polite or plain speech by who is talking to whom; keep 。and 、.",
    "ko": "Natural spoken Korean for subtitles: 반말/존댓말 by the speakers' relationship, consistent per character.",
    "pt": "Brazilian Portuguese unless the CONTEXT says the material is from Portugal.",
    "es": "Neutral Latin American Spanish unless the CONTEXT says the material is from Spain.",
    "de": "du or Sie by the speakers' relationship, consistent per character.",
    "fr": "tu or vous by the speakers' relationship, consistent per character.",
}

NAME_RULES = {
    ("ja", "en"): "romanise other Japanese names with Hepburn spelling and keep them consistent.",
    ("zh", "en"): "romanise other Chinese names in pinyin (no tone marks) and keep them consistent.",
    ("ko", "en"): "romanise other Korean names with Revised Romanization and keep them consistent.",
    ("th", "en"): "romanise other Thai names with RTGS spelling and keep them consistent.",
}


def name_rule(src: str, tgt: str) -> str:
    if (src, tgt) in NAME_RULES:
        return NAME_RULES[(src, tgt)]
    if tgt == "th":
        return "transliterate other names into Thai script and keep them consistent."
    return "keep other names consistent throughout."


def punct_rule(tgt: str) -> str:
    """CJK targets keep their own sentence punctuation; everything else drops the carried-over 。/！."""
    if tgt in _CJK_TARGETS:
        return "(a sentence-final 。 is fine; no ！ unless the line is shouted)"
    return "(no trailing 。 or ！ carried over)"


def system_prompt(src_code: str, tgt_code: str, genre: str) -> str:
    system = GENERIC_SYSTEM.format(src=lang_name(src_code), tgt=lang_name(tgt_code), genre=genre,
                                   names=name_rule(src_code, tgt_code), punct=punct_rule(tgt_code))
    if tgt_code in TARGET_RULES:
        system += "\n- " + TARGET_RULES[tgt_code]
    return system


TRANSLATEGEMMA_USER = ("You are a professional {src} to {tgt} translator. Your goal is to accurately convey the meaning "
                       "and nuances of the original {src} text while adhering to {tgt} grammar, vocabulary, and cultural "
                       "sensitivities. Produce only the {tgt} translation, without any additional explanations or commentary. "
                       "Keep the numbering: one translated line per numbered input line, in the same order. "
                       "Please translate the following {src} text into {tgt}:\n\n\n{text}")


SPEAKER_RULE = ("Some lines start with a speaker tag like [S2]. The same tag means the same voice throughout; a different "
                "tag is a different voice. The tags come from automatic speaker detection and can be wrong — treat them "
                "as a hint. They say ONLY that lines come from the same or a different speaker: nothing about who the "
                "speaker is — not their name, age, sex, status or relationship. Infer those from the dialogue itself, "
                "never from the tag. Keep each speaker's register, politeness level and pronouns consistent from line "
                "to line. Never output the tags.")
_SPEAKER_TAG = re.compile(r"^\s*\[S\d+\]\s*")


def spoken(c: Cue) -> str:
    """A cue's source text as the translator sees it: with its speaker tag when the speakers stage labelled it."""
    return f"[{c.speaker}] {c.ja}" if c.speaker else c.ja


def build_prompt(tr: Translator, window: list[Cue], before: list[Cue], after: list[Cue],
                 glossary: dict[str, str], genre: str, target: str) -> tuple[str | None, str]:
    src = lang_name(window[0].lang)
    tgt = lang_name(target)
    tagged = any(c.speaker for c in window + before + after)
    if tr.prompt_style == "translategemma":                      # fixed prompt: no room for tags
        numbered = "\n".join(f"{c.idx + 1}\t{c.ja}" for c in window)
        return None, TRANSLATEGEMMA_USER.format(src=src, tgt=tgt, text=numbered.replace("\t", ". "))
    numbered = "\n".join(f"{c.idx + 1}\t{spoken(c)}" for c in window)
    parts = []
    if glossary:
        parts.append(f"GLOSSARY ({src} → {tgt}):\n" + "\n".join(f"{k} → {v}" for k, v in glossary.items()))
    if before:
        parts.append("CONTEXT (already translated, do not output):\n" +
                     "\n".join(f"[{c.idx + 1}] {spoken(c)}  →  {c.en}" for c in before))
    parts.append("TRANSLATE THESE LINES:\n" + numbered)
    if after:
        parts.append("FOLLOWING LINES (context only, do not output):\n" + "\n".join(f"[{c.idx + 1}] {spoken(c)}" for c in after))
    system = system_prompt(window[0].lang, target, genre)
    if tagged:
        system += "\n- " + SPEAKER_RULE
    return system, "\n\n".join(parts)


def parse_numbered(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        s = m.group(2).strip()
        if n in out and out[n]:
            continue
        out[n] = s
    return out


_CJK_TARGETS = {"ja", "zh", "yue", "ko"}


def clean_en(s: str, target: str = "en") -> str:
    """Clean a translated line (the name is historical: it cleans any target language). CJK punctuation is only
    Westernised when the target is not itself CJK — a Japanese or Chinese subtitle keeps its 。 and 、."""
    s = s.strip()
    s = _SPEAKER_TAG.sub("", s)                   # an echoed speaker tag never reaches the subtitle
    s = _TN.sub("", s)
    if len(s) >= 2 and s[0] in "\"“'‘" and s[-1] in "\"”'’":
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    if target not in _CJK_TARGETS:
        s = s.replace("。", ".").replace("、", ",")
    return s.strip()


def has_japanese(s: str) -> bool:
    return script_of(s) in ("ja", "han")


def untranslated(text: str, src: str, tgt: str) -> bool:
    """The model echoed the source instead of translating: the line is in the source's script and not the target's.
    Two Latin-script languages cannot be told apart this way; then only an empty line counts."""
    if not text:
        return True
    if src == tgt:
        return False
    src_script = {"ja": "ja", "zh": "han", "yue": "han", "ko": "ko", "th": "th", "ru": "cyrillic", "uk": "cyrillic",
                  "el": "greek", "ar": "arabic", "hi": "devanagari"}.get(src, "latin")
    if src_script == "latin":
        return False
    return script_matches(text, src) and not script_matches(text, tgt)


def translate_cues(cues: list[Cue], client: LLMClient, glossary: dict[str, str] | None = None,
                   genre: str = "a documentary / interview programme",
                   window_size: int = config.WINDOW_CUES, before_n: int = config.CONTEXT_BEFORE,
                   after_n: int = config.LOOKAHEAD_AFTER, progress: bool = True, checkpoint=None,
                   target: str = "en", pool: ClientPool | None = None, force_model: str | None = None) -> list[Cue]:
    """Translate in windows into `target`. A window never mixes source languages; cues already in the target are
    copied through; the model for each window comes from the route of its language pair (via `pool`, else
    `client` handles everything). `checkpoint(cues)` runs after every window so an interrupted run loses at most
    one request; cues that already carry a translation (a resumed run) are skipped a whole window at a time and
    still serve as context."""
    glossary = glossary or {}
    total = len(cues)
    i = 0
    while i < total:
        c0 = cues[i]
        if c0.lang == target:
            j = i
            while j < total and cues[j].lang == target:
                if not cues[j].en:
                    cues[j].en = copy_through(cues[j].ja, target)
                    cues[j].flags = (cues[j].flags or []) + ["copied"]
                j += 1
            i = j
            continue
        window = [c0]
        for c in cues[i + 1:i + window_size]:
            if c.lang != c0.lang:
                break
            window.append(c)
        if all(c.en and not untranslated(c.en, c.lang, target) for c in window):
            i += len(window)
            continue
        cl = pool.use(route(c0.lang, target, force_model)) if pool is not None else client
        tr = cl.tr
        before = [c for c in cues[max(0, i - before_n):i] if c.en]
        after = cues[i + len(window):i + len(window) + after_n]
        system, user = build_prompt(tr, window, before, after, glossary, genre, target)
        got = _translate_window(cl, system, user, window, target)
        missing = [c for c in window if untranslated(c.en, c.lang, target)]
        if missing:
            cl.usage.fallbacks += len(missing)
            for c in missing:
                c.en = _translate_single(cl, c, before + [x for x in window if x.en and x is not c], glossary, genre, target)
        if progress:
            _log(f"[tl:{tr.name}] {c0.lang}→{target} {i + len(window)}/{total}  got {got}/{len(window)}"
                 f"{'  fallback ' + str(len(missing)) if missing else ''}")
        if checkpoint is not None:
            checkpoint(cues)
        i += len(window)
    return cues


def _translate_window(client: LLMClient, system: str | None, user: str, window: list[Cue], target: str,
                      nudge: bool = False) -> int:
    ids = {c.idx + 1: c for c in window}
    for attempt in range(2):
        try:
            text = client.chat(system, user, nudge=nudge)
        except PredictionAborted as e:
            # the model looped on this window. First: same window, re-sampled. Then: halves, recursively — a
            # smaller window changes the prompt enough to break most loops. A single cue that still loops is
            # left to the per-line fallback (which flags it 'untranslated'); the run never dies here.
            if not nudge:
                _log(f"[tl] prediction aborted ({e}); re-sampling window {window[0].idx + 1}–{window[-1].idx + 1}")
                return _translate_window(client, system, user, window, target, nudge=True)
            if len(window) > 1:
                mid = len(window) // 2
                _log(f"[tl] still looping; splitting window {window[0].idx + 1}–{window[-1].idx + 1} in two")
                hits = 0
                for part in (window[:mid], window[mid:]):
                    numbered = "\n".join(f"{c.idx + 1}\t{spoken(c)}" for c in part)
                    if "TRANSLATE THESE LINES:\n" in user:
                        puser = re.sub(r"TRANSLATE THESE LINES:\n.*?(?=\n\n|\Z)", "TRANSLATE THESE LINES:\n" + numbered,
                                       user, count=1, flags=re.S)
                    else:   # translategemma layout: the numbered text follows the last blank line
                        head = user.rsplit("\n\n\n", 1)[0]
                        puser = head + "\n\n\n" + numbered.replace("\t", ". ")
                    hits += _translate_window(client, system, puser, part, target, nudge=False)
                return hits
            _log(f"[tl] cue {window[0].idx + 1} loops the model even alone; leaving it to the per-line fallback")
            return 0
        parsed = parse_numbered(text)
        hits = 0
        for n, s in parsed.items():
            if n in ids and s:
                s = clean_en(s, target)
                if s and not untranslated(s, ids[n].lang, target):
                    ids[n].en = s
                    hits += 1
        if hits == len(window):
            return hits
        if attempt == 0:
            client.usage.retries += 1
            user = user + (f"\n\nYour previous answer covered {hits} of {len(window)} lines. Output ALL "
                           f"{len(window)} numbered lines ({window[0].idx + 1}–{window[-1].idx + 1}), one per line, "
                           f"as <number><TAB><{lang_name(target)}>.")
    return sum(1 for c in window if c.en)


def _translate_single(client: LLMClient, cue: Cue, context: list[Cue], glossary: dict[str, str], genre: str,
                      target: str) -> str:
    src, tgt = lang_name(cue.lang), lang_name(target)
    ctx = "\n".join(f"{spoken(c)} → {c.en}" for c in context[-6:] if c.en)
    gl = "\n".join(f"{k} → {v}" for k, v in glossary.items())
    if client.tr.prompt_style == "translategemma":
        system, user = None, TRANSLATEGEMMA_USER.format(src=src, tgt=tgt, text=cue.ja)
    else:
        system = system_prompt(cue.lang, target, genre)
        if cue.speaker or any(c.speaker for c in context):
            system += "\n- " + SPEAKER_RULE
        user = ((f"GLOSSARY:\n{gl}\n\n" if gl else "") + (f"CONTEXT (do not output):\n{ctx}\n\n" if ctx else "") +
                f"Translate this one subtitle line into {tgt}. Output only the {tgt} text, nothing else:\n{spoken(cue)}")
    for attempt in range(2):
        try:
            text = client.chat(system, user, max_tokens=256, nudge=(attempt == 1))
        except PredictionAborted:
            continue            # the second attempt re-samples; if that loops too, the cue is flagged below
        parsed = parse_numbered(text)
        cand = clean_en(parsed.get(cue.idx + 1) or parsed.get(1) or text.splitlines()[0] if text else "", target)
        if cand and not untranslated(cand, cue.lang, target):
            return cand
    cue.flags = (cue.flags or []) + ["untranslated"]
    return cue.en or ""
