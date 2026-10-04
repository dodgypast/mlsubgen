"""Defaults and translator presets. Everything here can be overridden from the command line."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
MLSUBGEN_HOME = Path(os.environ.get("MLSUBGEN_HOME", HOME / "mlsubgen"))
WORK_DIR = MLSUBGEN_HOME / "work"          # per-file transcript/translation cache; deleted once the .srt is written
BENCH_DIR = MLSUBGEN_HOME / "bench"
TMP_DIR = MLSUBGEN_HOME / "tmp"
LOG_DIR = MLSUBGEN_HOME / "logs"           # skipped.log: every file a run skipped, with the reason
DB_PATH = MLSUBGEN_HOME / "mlsubgen.db"       # the job queue (mlsubgen serve / jobs / cancel / retry)
WORKER_LOCK = MLSUBGEN_HOME / "worker.lock"
CODE_DIR = Path(__file__).resolve().parent.parent   # the checkout: subprocesses get it on PYTHONPATH (state may live elsewhere)
DETECT_VERSION = 2            # bump when the language check or the VAD skip logic changes: remembered skips are retried
JOB_MAX_FAILURES = 3          # a job that errors this many times is marked failed (interruptions do not count)
JOB_RETRY_DELAY_SEC = 300
JOB_READY_TIMEOUT_SEC = 1800  # how long a job may wait for its mounts / GPU / LLM server before it is failed
WORKER_POLL_SEC = 30
WEB_HOST = os.environ.get("MLSUBGEN_WEB_HOST", "0.0.0.0")   # `mlsubgen web` — LAN/Tailscale only, there is no login
WEB_PORT = int(os.environ.get("MLSUBGEN_WEB_PORT", "8790"))

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".ts", ".m2ts", ".webm", ".wmv", ".flv"}

# ── Languages ────────────────────────────────────────────────────────────────────────────────────────────
# ISO code → the name Qwen3-ASR wants. Qwen speaks 30; these are the ones we name. Whisper takes the code.
LANG_NAMES = {"ja": "Japanese", "en": "English", "zh": "Chinese (Simplified)", "yue": "Cantonese (Traditional)", "ko": "Korean", "th": "Thai",
              "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
              "id": "Indonesian", "vi": "Vietnamese", "tr": "Turkish", "hi": "Hindi", "ar": "Arabic", "nl": "Dutch",
              "pl": "Polish", "cs": "Czech", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
              "hu": "Hungarian", "ro": "Romanian", "el": "Greek", "uk": "Ukrainian",
              # 2026-10-01: more subtitle languages. Any code here is a target (the translator LLM writes it) and a
              # source (decoded by whisper unless its aligner language is in ALIGNER_LANGS). Codes are whisper's.
              "ms": "Malay", "tl": "Filipino", "fa": "Persian", "he": "Hebrew", "bn": "Bengali", "ta": "Tamil",
              "km": "Khmer", "lo": "Lao", "my": "Burmese", "ca": "Catalan", "bg": "Bulgarian", "hr": "Croatian",
              "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian"}
# How each language names itself — what the web form shows next to the English name
NATIVE_NAMES = {"ja": "日本語", "en": "English", "zh": "中文（简体）", "yue": "粵語（繁體）", "ko": "한국어", "th": "ไทย",
                "fr": "Français", "de": "Deutsch", "es": "Español", "it": "Italiano", "pt": "Português", "ru": "Русский",
                "id": "Bahasa Indonesia", "vi": "Tiếng Việt", "tr": "Türkçe", "hi": "हिन्दी", "ar": "العربية",
                "nl": "Nederlands", "pl": "Polski", "cs": "Čeština", "sv": "Svenska", "da": "Dansk", "fi": "Suomi",
                "no": "Norsk", "hu": "Magyar", "ro": "Română", "el": "Ελληνικά", "uk": "Українська",
                "ms": "Bahasa Melayu", "tl": "Filipino", "fa": "فارسی", "he": "עברית", "bn": "বাংলা", "ta": "தமிழ்",
                "km": "ខ្មែរ", "lo": "ລາວ", "my": "မြန်မာ", "ca": "Català", "bg": "Български", "hr": "Hrvatski",
                "sk": "Slovenčina", "sl": "Slovenščina", "lt": "Lietuvių", "lv": "Latviešu", "et": "Eesti"}
ALIGNER_LANGS = {"ja", "zh", "yue", "en", "ko", "fr", "de", "it", "pt", "ru", "es"}   # Qwen3-ForcedAligner — NOT Thai
# which ASR engine decodes a chunk, by its detected language ("*" = everything else); --asr forces one engine
ASR_ROUTES = {"*": "whisper", **{lang: "qwen" for lang in ALIGNER_LANGS}}
# a file whose dominant language is not in this set is skipped (reason logged); --source LANG forces it through.
# Every language we can name is a source: the detector's confidence is the gate, the layers below are language-agnostic
SOURCE_LANGS = set(LANG_NAMES)
# Targets withheld (2026-10-03): one episode translated into every language and read showed the translator's output
# in these to be below what can be shipped even with errors admitted — invented words (Lithuanian "xolops" for
# beetle, Estonian "märgikäigid"), broken grammar (Greek), Hungarian forms, and for Khmer, Lao and Burmese other
# scripts leaking into the text. They stay known languages (a subtitle track in them is still read as a source,
# detection and OCR still handle them) but are not offered as targets until a translator measured on them passes.
# 2026-10-04: Hungarian, Lithuanian, Latvian and Estonian return on TranslateGemma (routes above — dictionary-unknown
# words at a quarter to a tenth of the 31B's); Khmer, Lao and Burmese return on the 31B now that the translator's
# foreign-script guard sends a leaking line back (their faults were leaks, two reviewers found Burmese usable
# otherwise). Greek stays withheld: the choice there is between the 31B's grammar errors and TranslateGemma's
# gender slashes, and the character sheet has to settle that first.
UNSUPPORTED_TARGETS = {"el"}
WITHHELD_REASON = {"el": "the 31B's grammar errors vs TranslateGemma's gender slashes — returns once the character sheet is measured on it (2026-10-04)"}
# Per profile (2026-10-04): a smaller card runs a smaller translator, and a language that reads well on the 31B
# may not on the 26B or the 4B. Each profile withholds what has been measured below shippable on ITS translators;
# until a profile has its own measurement it inherits the full profile's set (the overnight batch of 2026-10-04 is
# the first such measurement). apply_profile() rebuilds TARGET_LANGS from this.
PROFILE_WITHHELD: dict[str, set[str]] = {
    "full": set(UNSUPPORTED_TARGETS), "16gb": set(UNSUPPORTED_TARGETS), "12gb": set(UNSUPPORTED_TARGETS),
    "8gb": set(UNSUPPORTED_TARGETS), "12gb-dense": set(UNSUPPORTED_TARGETS), "8gb-dense": set(UNSUPPORTED_TARGETS),
}
TARGET_LANGS = {c: n for c, n in LANG_NAMES.items() if c not in UNSUPPORTED_TARGETS}


def targets_for_profile(name: str) -> dict[str, str]:
    """The languages offered as targets on a profile — what `mlsubgen languages --profile X` and the web form show."""
    withheld = PROFILE_WITHHELD.get(name, UNSUPPORTED_TARGETS)
    return {c: n for c, n in LANG_NAMES.items() if c not in withheld}
# ── Default subtitle languages ───────────────────────────────────────────────────────────────────────────
# Precedence: --target on a run  >  settings.json (set from the CLI: `mlsubgen config targets en,th`, or the web
# form's "make these the default")  >  MLSUBGEN_TARGETS in the environment (the units / .env)  >  "en".
# Nothing in the pipeline needs English to be among them: every route, rule and file name is per target.
SETTINGS_PATH = MLSUBGEN_HOME / "settings.json"


def load_settings() -> dict:
    import json
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    import json
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


def _targets_from(settings: dict) -> tuple[str, str]:
    """(comma list, where it came from)."""
    saved = settings.get("targets")
    if isinstance(saved, list) and saved:
        return ",".join(str(t).strip().lower() for t in saved if str(t).strip()), "settings"
    env = os.environ.get("MLSUBGEN_TARGETS", "").strip()
    if env:
        return env, "environment"
    return "en", "built-in"


DEFAULT_TARGETS, DEFAULT_TARGETS_SOURCE = _targets_from(load_settings())


def set_default_targets(codes: list[str] | None) -> tuple[str, str]:
    """Persist the default subtitle languages (None / [] clears the saved value, so the environment or the built-in
    default applies again) and make them current in this process. Returns (comma list, source)."""
    global DEFAULT_TARGETS, DEFAULT_TARGETS_SOURCE
    data = load_settings()
    codes = [c.strip().lower() for c in (codes or []) if c.strip()]
    if codes:
        data["targets"] = list(dict.fromkeys(codes))
    else:
        data.pop("targets", None)
    save_settings(data)
    DEFAULT_TARGETS, DEFAULT_TARGETS_SOURCE = _targets_from(data)
    return DEFAULT_TARGETS, DEFAULT_TARGETS_SOURCE
# (source, target) → translator preset; the most specific entry wins, "*" matches anything; -t forces one model.
# Filled in from the hardware profile below (apply_profile) — see PROFILES.
TRANSLATE_ROUTES: dict[tuple[str, str], str] = {}

# ── Hardware profiles (2026-10-01; the 26B profiles 2026-10-03) ─────────────────────────────────────────────
# The ASR stage and the translation stage never share the GPU, so the card only has to hold the bigger of the two.
#   full  ≥ 20 GB   both ASR engines resident (≈ 10 GB); 27–31B translators (17–19 GB), fully on the card
#   16gb  15–20 GB  both engines resident, float16 (≈ 10 GB); the 26B MoE with 22 of 30 layers on the card
#   12gb  11–15 GB  both engines resident, whisper int8 (≈ 8 GB); the 26B with 14 layers on the card (24 GB RAM)
#   8gb   < 11 GB   one ASR engine at a time; the 26B with 6 layers on the card (32 GB RAM)
#   12gb-dense / 8gb-dense: gemma4:12b-it-qat / gemma4:e4b-it-qat, for machines without the RAM for the 26B's rest
# Picked from the GPU's memory at start (MLSUBGEN_PROFILE=auto), or forced: MLSUBGEN_PROFILE=12gb / --profile 12gb.
# The 26B mixture-of-experts (gemma4:26b, 30 layers, 18 GB, ~4B parameters active per token) measured 2026-10-03 on
# the Shin-chan clip against the human reference: chrF++ 12B 29.0 · 26B 30.2 · 31B 31.0, at 50 / 84 / 23 tok/s on a
# 24 GB card; as a vision OCR engine on Thai bitmaps 26B 76 % / 93.8 = the 31B's 77 % / 93.7 (the 12B: 65 % / 89.3).
# Ollama runs the layers that do not fit on the CPU, and because so few parameters are active it stays fast:
#   layers on the card   24 (13.8 GB)   16 (9.7 GB)   8 (5.6 GB)
#   tok/s                84             51            40
# So cards from 8 GB up can run the 26B — IF the machine has the system RAM for the rest of the model (the
# remainder is read from RAM every token). The profiles below take the 26B where the card AND the RAM allow it,
# and fall back to the dense small models otherwise. OLLAMA_NUM_GPU is the layer split, passed on every request.
PROFILES = {
    # Per-language routes on the full profile (2026-10-04, one film in 44 languages, two translators): TranslateGemma
    # 27B where the 31B invents words (Hungarian 3.5 % of words unknown to the dictionary → 0.4, Lithuanian 4.3 →
    # 0.6, Latvian 5.6 → 0.7, Estonian 3.0 → 0.5, Catalan 2.6 → 1.0; Finnish invented words on reading) and its
    # own fault — formal address by default, gender hedged with slashes — costs little. The 31B keeps every language
    # with a register system it handles (tu/vous, 반말, kin pronouns), and Greek, Czech and Slovenian until the
    # character sheet settles the gender question for both.
    "full": dict(min_vram_gb=20.0,
                 routes={("ja", "en"): "qwen3.8", ("*", "hu"): "translategemma", ("*", "lt"): "translategemma",
                         ("*", "lv"): "translategemma", ("*", "et"): "translategemma", ("*", "ca"): "translategemma",
                         ("*", "fi"): "translategemma", ("*", "*"): "gemma4"},
                 default="qwen3.8", whisper_compute="float16", asr_sequential=False, vlm="gemma4:31b-it-qat", num_gpu=None, min_ram_gb=16),
    "16gb": dict(min_vram_gb=15.0, routes={("*", "*"): "gemma4-26b"}, default="gemma4-26b",
                 whisper_compute="float16", asr_sequential=False, vlm="gemma4:26b", num_gpu=22, min_ram_gb=16,
                 fallback="12gb-dense"),
    "12gb": dict(min_vram_gb=11.0, routes={("*", "*"): "gemma4-26b"}, default="gemma4-26b",
                 whisper_compute="int8_float16", asr_sequential=False, vlm="gemma4:26b", num_gpu=14, min_ram_gb=24,
                 fallback="12gb-dense"),
    "8gb": dict(min_vram_gb=0.0, routes={("*", "*"): "gemma4-26b"}, default="gemma4-26b",
                whisper_compute="int8_float16", asr_sequential=True, vlm="gemma4:26b", num_gpu=6, min_ram_gb=32,
                fallback="8gb-dense"),
    # the dense small models, for machines without the RAM for the 26B's remainder
    "12gb-dense": dict(min_vram_gb=11.0, routes={("*", "*"): "gemma4-12b"}, default="gemma4-12b",
                       whisper_compute="int8_float16", asr_sequential=False, vlm="gemma4:12b-it-qat", num_gpu=None, min_ram_gb=0),
    "8gb-dense": dict(min_vram_gb=0.0, routes={("*", "*"): "gemma4-e4b"}, default="gemma4-e4b",
                      whisper_compute="int8_float16", asr_sequential=True, vlm="gemma4:e4b-it-qat", num_gpu=None, min_ram_gb=0),
}
OLLAMA_NUM_GPU: int | None = None   # layers of the translator / vision model kept on the card; None = all (set by the profile)
RAM_GB: float | None = None         # system memory, detected
PROFILE = "full"                    # the active profile (set by apply_profile at import, below)
DEFAULT_TRANSLATOR = "qwen3.8"      # the preset that reconciles the two ASR transcripts and stands in for a missing route
WHISPER_COMPUTE = "float16"         # faster-whisper compute type (int8_float16 halves its memory)
ASR_SEQUENTIAL = False              # True: never hold both ASR engines at once (reload per file instead)
VRAM_GB: float | None = None        # what was detected
# the vision model that reads bitmap subtitles when --engine vlm is asked for (0.4.8): the profile's Gemma 4 — the
# same model the profile translates with, so it costs no extra card space; MLSUBGEN_OCR_VLM overrides. The smaller
# ones read less well: measure with `mlsubgen ocrbench … --engine vlm` before relying on one
OCR_VLM_MODEL = os.environ.get("MLSUBGEN_OCR_VLM", "gemma4:31b-it-qat")


def detect_vram_gb() -> float | None:
    """The largest GPU's memory in GB via nvidia-smi, or None when there is no NVIDIA GPU to ask."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        sizes = [float(x) for x in out.split() if x.strip().replace(".", "").isdigit()]
        return max(sizes) / 1024.0 if sizes else None
    except Exception:  # noqa: BLE001
        return None


def pick_profile(name: str | None, vram_gb: float | None) -> str:
    """A profile name as given, or by memory ('auto'); no GPU information → 'full' (the historical defaults)."""
    if name and name != "auto":
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; profiles: {', '.join(PROFILES)}")
        return name
    if vram_gb is None:
        return "full"
    for prof in ("full", "16gb", "12gb", "8gb"):              # the -dense ones are never picked by memory: the RAM gate does that
        if vram_gb >= PROFILES[prof]["min_vram_gb"]:
            return prof
    return "8gb"


def apply_profile(name: str | None = None) -> str:
    """Make `name` (or MLSUBGEN_PROFILE / auto-detection) the active profile: routes, default translator, whisper
    compute type, sequential ASR. Mutates the module's values in place, so every `config.X` reader sees it."""
    global PROFILE, DEFAULT_TRANSLATOR, WHISPER_COMPUTE, ASR_SEQUENTIAL, VRAM_GB, OCR_VLM_MODEL, OLLAMA_NUM_GPU, RAM_GB
    want = name or os.environ.get("MLSUBGEN_PROFILE") or "auto"
    if want == "auto" and VRAM_GB is None:
        VRAM_GB = detect_vram_gb()
    if RAM_GB is None:
        RAM_GB = detect_ram_gb()
    PROFILE = pick_profile(want, VRAM_GB)
    p = PROFILES[PROFILE]
    # a profile that runs the 26B partly from system RAM needs that RAM; without it, its dense fallback
    if want == "auto" and p.get("fallback") and RAM_GB is not None and RAM_GB < p["min_ram_gb"]:
        PROFILE = p["fallback"]
        p = PROFILES[PROFILE]
    TRANSLATE_ROUTES.clear()
    TRANSLATE_ROUTES.update(p["routes"])
    DEFAULT_TRANSLATOR = p["default"]
    WHISPER_COMPUTE = p["whisper_compute"]
    ASR_SEQUENTIAL = p["asr_sequential"]
    OLLAMA_NUM_GPU = p.get("num_gpu")
    if os.environ.get("MLSUBGEN_NUM_GPU"):                   # measurement override: layers on the card for every Ollama request
        OLLAMA_NUM_GPU = int(os.environ["MLSUBGEN_NUM_GPU"])
    if not os.environ.get("MLSUBGEN_OCR_VLM"):
        OCR_VLM_MODEL = p["vlm"]
    # the languages offered as targets follow the profile (2026-10-04): PROFILE_WITHHELD is defined further down,
    # so the first call at import leaves the module defaults and the next call (a job's profile) applies them
    g = globals()
    if "PROFILE_WITHHELD" in g and "LANG_NAMES" in g:
        g["UNSUPPORTED_TARGETS"] = set(g["PROFILE_WITHHELD"].get(PROFILE, g["UNSUPPORTED_TARGETS"]))
        g["TARGET_LANGS"] = {c: n for c, n in g["LANG_NAMES"].items() if c not in g["UNSUPPORTED_TARGETS"]}
    return PROFILE


def detect_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


apply_profile()

# ── Language identification (LID) — decide the language, then decode ───────────────────────────────────────
LID_VERSION = 5                # v3 (2026-10-01): function-word evidence for Latin-script languages, strongly evidenced
                               # single windows survive the smoothing, switch points refined to the exact span. v4: the
                               # speaker-aware window and run rules (0.4.5). Cached older verdicts are redone; remembered
                               # skips are keyed by DETECT_VERSION and are not
LID_STRONG_MARGIN = 0.6        # a non-dominant run shorter than LID_SWITCH_MIN_WINDOWS survives when every window's margin is ≥ this
# speaker-aware detection (0.4.4, with --speakers): spans cut at speaker turns, a window is one voice, every voice
# sampled, and a voice's language history is evidence for its uncertain windows — never for its confident ones
LID_SPEAKER_SAMPLES = 2        # every voice gets at least this many of its windows judged in the first pass
LID_PRIOR_MIN_SPEECH = 8.0     # a voice needs this much confidently labelled speech before it has a prior
LID_PRIOR_STRONG = 0.5         # the prior's weight when one language is ≥ 90 % of the voice's confident speech …
LID_PRIOR_WEAK = 0.25          # … ≥ 65 %; below that a voice is bilingual and has no prior at all
LID_SAME_SPEAKER_REACH = 6     # an uncertain window inherits from a confident window of its own voice this many windows away
# Measured 2026-10-01 on Babel / Inglourious Basterds / Only God Forgives against their forced subtitle tracks
# (recall / switch recall / invented switches, plain v3 = 90/66/59, 97/57/24, 83/61/10):
#   close the window at every voice change, window-count run rule (0.4.4)  92/70/69  98/61/37  91/70/9   ← kept
#   + close only after 3 s of speech, seconds-based run rule (0.4.5)       87/61/56  98/58/31  83/54/9
#   + close at once, seconds-based run rule                                 80/57/60  91/57/51  78/54/10
#   + close after 1.5 s, seconds-based run rule                             84/60/55  95/58/39  83/54/9
# Merging short turns loses the boundaries that are the point; the seconds rule absorbs short TRUE foreign runs.
# Clustering on Babel (same rules): threshold 1.0 → 142 voices 92/70/69; 1.3 → 5 voices 92/67/63; 1.6 and 2.0 →
# 1 voice, gate rejects, = plain v3; --speakers 30 → 92/70/68. So the invented switches are not the cluster
# count's doing: they come with per-turn windows themselves, and the priors seldom engage because sherpa's
# clusters do not line up with language-consistent characters on a multilingual film. The threshold has a cliff
# between 1.0 and 1.3 on a feature film (142 → 5 voices); 1.0 keeps the turn boundaries, which are the useful part.
LID_SPEAKER_WINDOW_MIN_SPEECH = 0.0   # a change of voice closes the window at once
LID_SWITCH_MIN_SPEECH = 8.0           # the seconds-based run rule (experiments only; see lid.smooth)
LID_STRONG_MIN_SPEECH = 4.0
LID_WINDOW_SPEECH_SEC = 10.0   # a detection window = consecutive VAD spans until this much SPEECH (not audio)
LID_WINDOW_MAX_AUDIO_SEC = 30.0
LID_SAMPLE_WINDOWS = 24        # first pass: this many windows spread over the file; every window if a 2nd language shows
LID_CONFIDENT_SCORE = 1.0      # whisper prob (≤1) + Qwen agreement (0.5) + script evidence (0.7 / 0.3) must reach this
LID_CONFIDENT_MARGIN = 0.4     # … and beat the runner-up by this
LID_SWITCH_MIN_WINDOWS = 2     # a second language needs this many consecutive confident windows to count
LID_MIN_SPEECH_SEC = 6.0       # less speech than this in the whole file: "no usable speech"

# ── ASR ──────────────────────────────────────────────────────────────────────────────────────────────────
ASR_ENGINE = "dual"                          # dual (both engines, LLM-merged) | auto (per chunk, ASR_ROUTES) | qwen | whisper
ASR_VERSION = 5                              # part of the ASR cache key: bump when chunking, repair, fallback or merge changes
COVER_QUIET_DB = 6.0                         # timeline coverage: a stretch this close to the noise floor …
COVER_MIN_QUIET_SEC = 2.5                    # … for this long is silence and is skipped; everything else is decoded
MERGE_AGREE = 0.92                           # dual mode: chunks whose two transcripts agree at least this much need no LLM
MERGE_MIN_CHARS = 4                          # … and the LLM is only asked when both sides have at least this much text
ASR_RETRY_DENSITY = 3.0                      # a chunk with fewer characters than this per second of VAD speech in it
                                             # (given ≥ ASR_RETRY_MIN_SPEECH s of speech) came back thin: the other engine
                                             # decodes it again — an LLM decoder can emit nothing over music, a whisper
                                             # window can drop out; the two fail differently
ASR_RETRY_MIN_SPEECH = 3.0
ALIGN_SEC_PER_CHAR = 0.2                     # re-timing a collapsed run: this long per character when room allows
ALIGN_MAX_WORD_SEC = 3.0                     # a single word longer than this is an aligner artefact
ASR_MODEL_QWEN = "Qwen/Qwen3-ASR-1.7B"       # alt: neosophie/Qwen3-ASR-1.7B-JA (proper-noun fine-tune)
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
ASR_MODEL_WHISPER = "large-v3"                # the full model: turbo's 4-layer decoder is large-v2 quality, worse on Thai         # faster-whisper name; large-v3 is the slower/stronger sibling
ASR_LANGUAGE_QWEN = "Japanese"
ASR_LANGUAGE_WHISPER = "ja"

# ── Speakers (diarization, 0.4.0) — sherpa-onnx, on the CPU; opt-in with --speakers ─────────────────────────
SPEAKER_MODEL_DIR = MLSUBGEN_HOME / "models" / "speakers"       # fetched by `mlsubgen pull speakers` from GitHub
SPEAKER_SEGMENTATION_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SPEAKER_SEGMENTATION_FILE = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"   # pyannote segmentation-3.0, MIT
# The speaker embedding model decides how well turns cluster into voices. Any file from sherpa-onnx's
# speaker-recongition-models release works: MLSUBGEN_SPEAKER_EMBEDDING=<file> or --speaker-embedding <file>,
# then `mlsubgen pull speakers`. Measured 2026-10-02 on Babel with --speakers 30 — voice/language purity (the share
# of each voice's confident speech in its own main language; what the cue stage's "same voice" labels depend on),
# speaker priors engaged, and the detector's recall / switch recall / invented switches:
#   3dspeaker campplus zh_en advanced (27 MB)        purity 0.80  priors 16   92% / 65% / 64   ← the default
#   nemo titanet_large (97 MB)                        purity 0.76  priors  7   92% / 67% / 61
#   3dspeaker eres2net_base zh-cn (38 MB, old default) purity 0.72  priors  9   92% / 70% / 68
#   wespeaker voxceleb resnet34_LM (26 MB)            purity 0.68  priors  4   92% / 69% / 65
#   wespeaker voxceleb resnet293_LM (110 MB)          purity 0.63  priors  5   91% / 69% / 69
# The detector does not care which (its gains come from the turn boundaries, which are the segmentation model's);
# the clustering does, and the models trained on Chinese + English material cluster a film better than the
# VoxCeleb ones. Each model has its own clustering threshold scale, so compare them with --speakers N first.
SPEAKER_EMBEDDING_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
SPEAKER_EMBEDDING_FILE = os.environ.get("MLSUBGEN_SPEAKER_EMBEDDING", "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")
if not SPEAKER_EMBEDDING_FILE.endswith(".onnx"):
    SPEAKER_EMBEDDING_FILE += ".onnx"
SPEAKER_EMBEDDING_URL = SPEAKER_EMBEDDING_RELEASE + SPEAKER_EMBEDDING_FILE


def set_speaker_embedding(name: str) -> str:
    """Switch the embedding model for this process (--speaker-embedding). Returns the file name."""
    global SPEAKER_EMBEDDING_FILE, SPEAKER_EMBEDDING_URL
    name = name.strip()
    if not name.endswith(".onnx"):
        name += ".onnx"
    SPEAKER_EMBEDDING_FILE = name
    SPEAKER_EMBEDDING_URL = SPEAKER_EMBEDDING_RELEASE + name
    return name
SPEAKER_THRESHOLD = 1.0        # clustering threshold when the speaker count is not given: smaller = more speakers.
                               # 2026-10-01 sweep on a ten-minute anime clip: 0.5 → 80 clusters, 0.8 → 31, 1.0 → 14 with a
                               # plausible share per voice, 1.1 → 8; cue-boundary agreement was the same (48–49 %) throughout
SPEAKER_MIN_ON = 0.3           # a speaker turn shorter than this is dropped
SPEAKER_MIN_OFF = 0.5          # a gap shorter than this does not end a turn
SPEAKER_MIN_OVERLAP = 0.6      # a word takes a speaker only when that speaker covers this share of it (and twice the runner-up)
SPEAKER_MAX_AMBIGUOUS = 0.5    # labels are dropped for a file when more than this share of its words are ambiguous …
SPEAKER_MIN_CLUSTERS = 2       # … or when the diarizer found fewer speakers than this (nothing to tell apart)
SPEAKER_MAX_CLUSTER_RATIO = 0.25   # … or more clusters than this share of the turns (above 8): fragmentation, not speakers
SPEAKERS_VERSION = 1           # part of the cache: bump when the models or the assignment change

# ── Terminology (0.5.1): the film's names rendered once per target, fed to every translation window ─────────
TERMS_VERSION = 5              # part of the cache: bump when the extraction or the prompts change (4: a name is not a title;
                               #  5: English interjections and function words are never candidates)
                               # (2: the renderer dropped "ordinary words" — reversed in 3, it dropped the terms that mattered;
                               #  3: fragments of any word in the text are never terms)
TERMS_MIN_CUES = 40            # a clip shorter than this has no recurring names worth a pass
TERMS_MIN_OCCURRENCES = 2      # a candidate must occur this often in the transcript to be a term
TERMS_MAX = 60                 # the most frequent terms are kept; the glossary goes into every window's prompt
TERMS_CHUNK_CHARS = 6000       # transcript text per extraction call (up to four calls, spread over a long film)

# ── The character sheet (0.5.1): who speaks, and how they address each other in the target ──────────────────
CHARACTERS_VERSION = 1         # part of the cache: bump when the prompts change
CHARACTERS_MIN_CUES = 40       # a clip shorter than this has no recurring characters worth a sheet
CHARACTERS_CHARS = 14000       # transcript text the sheet call reads (sampled evenly beyond that)

# ── OCR of bitmap subtitle tracks (0.4.8) ────────────────────────────────────────────────────────────────────
OCR_WORKERS = 4                # tesseract processes at once (one per image; a 1,800-cue track is ~4 min on four cores)
OCR_VERSION = 2                # part of the OCR cache name: bump when the decoding, cleaning or engine changes
                               # (1 → 2: underscore-for-dash and bare-l-for-I rules, from the full-track measurement)
OCR_PREP = os.environ.get("MLSUBGEN_OCR_PREP", "binary")   # how the subtitle image is prepared for tesseract: binary | fill | gray | fill3x
# the OCR gate (0.5.0.4): an OCR'd track is used only when it reads like subtitles in the expected language —
# thresholds set from the measured tracks (see ocr.assess)
OCR_GATE_SCRIPT = 0.7          # at least this share of cues in the language's script
OCR_GATE_SYMBOLS = 0.12        # at most this share of characters that are symbols rather than letters, digits, punctuation
OCR_GATE_REPLACEMENT = 0.002   # at most this share of replacement characters (U+FFFD)
OCR_GATE_REPEAT = 0.2          # no single line on more than this share of cues
OCR_GATE_PROSE = 0.02          # at most this share of cues that describe the image ("The text says…")
OCR_GATE_JUNK = 0.15           # at most this share of cues carrying a line that is mostly digits and symbols (in-script salad)
# OCR_VLM_MODEL (the vision model for --engine vlm) is set per hardware profile by apply_profile(), above

# ── VAD / chunking ───────────────────────────────────────────────────────────────────────────────────────
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 300
VAD_MIN_SPEECH_MS = 200
VAD_SPEECH_PAD_MS = 150
CHUNK_MAX_SEC = 30.0           # the operating point every Qwen3-ASR integration uses and its benchmarks were run at:
                               # ≤ 30 s at VAD boundaries. Longer chunks make the decoder skim or go silent over music
                               # and the forced aligner collapse (half the words of a sparse film)
CHUNK_BREAK_SILENCE_SEC = 2.0  # a silence this long closes a chunk — once the chunk is at least CHUNK_MIN_SEC long
CHUNK_MIN_SEC = 6.0            # no 1-second orphan chunks: an isolated span joins its neighbour instead …
CHUNK_MAX_GAP_SEC = 30.0       # … unless the gap is this long (the decoder must not be fed minutes of silence)
CHUNK_PAD_SEC = 0.25

# ── Cue segmentation (Japanese side) ─────────────────────────────────────────────────────────────────────
CUE_MAX_SEC = 6.0
CUE_MAX_CHARS_JA = 34
CUE_GAP_SPLIT_SEC = 0.6
CUE_MIN_SEC = 0.9

# ── Subtitle typesetting (English side) ──────────────────────────────────────────────────────────────────
SRT_MAX_LINE_CHARS = 42
SRT_MAX_LINE_CHARS_BY_LANG = {"th": 50, "ja": 30, "zh": 30, "yue": 30, "ko": 34,   # narrower glyphs / no word spaces
                              "km": 50, "lo": 50, "my": 50}                         # no word spaces, like Thai
SRT_MAX_LINES = 2
SRT_MIN_DUR = 1.0
SRT_MAX_DUR = 7.0
SRT_MIN_GAP = 0.083          # 2 frames at 24 fps
SRT_MAX_CPS = 21.0           # reading speed ceiling; cues are stretched into the following gap when faster
SRT_MAX_CPS_BY_LANG = {"ja": 8.0, "zh": 9.0, "yue": 9.0, "ko": 12.0}   # a CJK character carries more than a letter

# ── Translation ──────────────────────────────────────────────────────────────────────────────────────────
LLM_BACKEND = "ollama"                       # ollama | openai   (openai = any /v1/chat/completions server)
LLM_URL = os.environ.get("MLSUBGEN_LLM_URL", "http://127.0.0.1:11434")
LLM_NUM_CTX = 8192
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT_SEC = 900
WINDOW_CUES = 20              # cues per request
CONTEXT_BEFORE = 8            # already-translated cue pairs shown before the window
LOOKAHEAD_AFTER = 3           # untranslated cues shown after the window (coherence only)


@dataclass
class Translator:
    name: str
    model: str
    backend: str = "ollama"
    url: str = ""
    prompt_style: str = "generic"   # generic | translategemma
    think: bool = False
    note: str = ""
    num_ctx: int = LLM_NUM_CTX
    temperature: float = LLM_TEMPERATURE
    extra_options: dict = field(default_factory=dict)


TRANSLATORS: dict[str, Translator] = {
    # Newest dense Qwen (Aug 2026). Reasoning model — thinking is switched off for translation.
    "qwen3.8": Translator("qwen3.8", "qwen3.8:27b", note="Qwen3.8-27B q4_K_M, 18 GB, 256K ctx, Apache-2.0", think=False),
    # Gemma 4 31B, quantisation-aware-trained 4-bit (19 GB) — dense, slower, strong multilingual.
    "gemma4": Translator("gemma4", "gemma4:31b-it-qat", note="Gemma 4 31B QAT, 19 GB, Apache-2.0", think=False),
    # Google's translation-specialised Gemma 3 (55 languages). Fixed prompt format; no instructions beyond 'translate'.
    "translategemma": Translator("translategemma", "translategemma:27b", prompt_style="translategemma",
                                 note="TranslateGemma 27B q4_K_M, 17 GB — translation-only model; routed for six languages on the full profile"),
    # the smaller TranslateGemmas (2026-10-04): candidates for the routed languages on the small profiles — unmeasured
    # until the overnight batch; nothing routes to them yet
    "translategemma-12b": Translator("translategemma-12b", "translategemma:12b", prompt_style="translategemma",
                                     note="TranslateGemma 12B, 8.1 GB — candidate for the 16gb/12gb profiles' routed languages (unmeasured)"),
    "translategemma-4b": Translator("translategemma-4b", "translategemma:4b", prompt_style="translategemma",
                                    note="TranslateGemma 4B, 3.3 GB — candidate for the 8gb profile's routed languages (unmeasured)"),
    # Fast MoE fallback: JP-TL-Bench LT 9.56 (above GPT-4o), ~3B active params.
    "qwen3-30b": Translator("qwen3-30b", "qwen3:30b-a3b-instruct-2507-q4_K_M", note="Qwen3-30B-A3B-Instruct-2507, ~18 GB, fast", think=False),
    # Smaller cards (the 12gb / 8gb profiles). Dense Gemma 4 12B at 4-bit QAT; the E4B edge model; the 26B-A4B MoE
    # for 16 GB cards (Ollama offloads part of it to the CPU below that).
    "gemma4-12b": Translator("gemma4-12b", "gemma4:12b-it-qat", note="Gemma 4 12B QAT, 7.2 GB — the 12gb profile", think=False),
    "gemma4-e4b": Translator("gemma4-e4b", "gemma4:e4b-it-qat", note="Gemma 4 E4B QAT, 6.1 GB — the 8gb profile", think=False),
    "gemma4-26b": Translator("gemma4-26b", "gemma4:26b", note="Gemma 4 26B-A4B MoE, 18 GB, ~4B active — the 16gb/12gb/8gb profiles, split across card and RAM", think=False),
}
# DEFAULT_TRANSLATOR is set by apply_profile() above (qwen3.8 on the full profile)

# Japanese phrases Whisper-family and Qwen models emit over music/silence (the YouTube tail). Only applied when the
# VAD says there was little or no speech under the cue.
HALLUCINATION_PATTERNS = [
    r"ご視聴ありがとうございました",
    r"ご視聴いただき",
    r"チャンネル登録",
    r"字幕(は|：|:)",
    r"最後までご覧",
    r"おやすみなさい[。！]?$",
    r"^ありがとうございました[。！]?$",
    r"^(ん+|あ+|え+|う+|は+)[。、]?$",
]
MIN_SPEECH_RATIO = 0.15       # below this VAD coverage a cue is suspect: kept only if it looks and sounds like speech
QUIET_MAX_CPS = 20.0          # … a suspect cue faster than this (chars/s) is a hallucination
QUIET_MAX_SEC = 8.0           # … or longer than this
QUIET_MIN_DB = 6.0            # … or with less than this above the file's noise floor under it
BLACKLIST_SPEECH_RATIO = 0.5  # a blacklisted phrase survives only if the VAD saw real speech under it
