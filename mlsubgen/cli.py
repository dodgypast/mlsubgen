"""Command line: run | bench | tracks | models | clean | selftest | help — the queue: serve | jobs | log | cancel | retry — web"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import __version__, config, jobs, work, worker
from .audio import parse_ts
from .config import TRANSLATORS, Translator


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────
def find_videos(paths: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p).expanduser().resolve()   # absolute: a name starting with '-' can't be mistaken for an option
        if path.is_dir():
            it = path.rglob("*") if recursive else path.iterdir()
            out += sorted(x for x in it if x.is_file() and x.suffix.lower() in config.VIDEO_EXTS)
        elif path.is_file():
            out.append(path)
        else:
            _log(f"skip (not found): {path}")
    return out


def load_glossary(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    gl: dict[str, str] = {}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            ja, en = line.split("\t", 1)
        elif "=" in line:
            ja, en = line.split("=", 1)
        else:
            continue
        gl[ja.strip()] = en.strip()
    return gl


def resolve_translator(name: str | None, model: str | None, backend: str | None, url: str | None,
                       prompt_style: str | None, num_ctx: int | None, temperature: float | None) -> Translator:
    base = TRANSLATORS.get(name or config.DEFAULT_TRANSLATOR)
    if base is None:
        if not model:
            raise SystemExit(f"unknown translator {name!r}; presets: {', '.join(TRANSLATORS)} — or pass --model")
        base = Translator(name or model, model)
    tr = replace(base)
    if model:
        tr = replace(tr, model=model, name=name or model)
    if backend:
        tr = replace(tr, backend=backend)
    if url:
        tr = replace(tr, url=url)
    if prompt_style:
        tr = replace(tr, prompt_style=prompt_style)
    if num_ctx:
        tr = replace(tr, num_ctx=num_ctx)
    if temperature is not None:
        tr = replace(tr, temperature=temperature)
    return tr


def record_skip(video: Path, reason: str) -> None:
    """Append to ~/mlsubgen/logs/skipped.log — the one thing a run leaves behind besides the .srt files."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_DIR / "skipped.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{video.resolve()}\t{reason}\n")


def clip_arg(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    a, b = text.split("-", 1)
    return parse_ts(a), parse_ts(b)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default=None, choices=["auto", *config.PROFILES],
                   help="hardware profile: full (>= 20 GB GPU), 12gb, 8gb — translator routes, whisper precision and "
                        "whether both ASR engines may be resident (default: MLSUBGEN_PROFILE, else by the GPU's memory)")
    p.add_argument("--speakers", default="labels", metavar="off|labels|auto|N",
                   help="speaker diarization (CPU, `mlsubgen pull speakers` first). labels (default, 0.5.10) = when the transcript is "
                        "a text track, voices from the audio label its cues so the character sheet knows who speaks; the audio path "
                        "is untouched. auto / N = also on the audio path: a speaker change closes a cue, detection follows voices, "
                        "the translator is told which lines share a voice (never written to the subtitles). off = no diarization")
    p.add_argument("--speaker-threshold", type=float, default=None,
                   help=f"clustering threshold for --speakers auto (default {config.SPEAKER_THRESHOLD}; smaller = more speakers)")
    p.add_argument("--speaker-embedding", default=None, metavar="FILE",
                   help="the speaker embedding model (a file from sherpa-onnx's speaker-recongition-models release; "
                        "`mlsubgen pull speakers` fetches it). See config.py for the known ones")
    p.add_argument("--ocr", default="auto", choices=["auto", "off"],
                   help="bitmap (PGS) subtitle tracks (0.4.8): auto = read them through OCR (tesseract + the language's pack) — "
                        "a bitmap track in a target language becomes that target's .srt, one in the spoken language becomes the "
                        "transcript; off = ignore them as before")
    p.add_argument("--terms", default="auto", choices=["auto", "off"],
                   help="terminology pass (0.5.1): auto = the film's names and recurring terms are rendered once per target and "
                        "fed to every translation window (your --glossary still wins); off = windows decide on their own")
    p.add_argument("--register", default="auto", choices=["auto", "off"],
                   help="the character sheet (0.5.1): auto = who speaks, their gender and how they address each other is worked out "
                        "once per film and given to every window in the target's terms (pronouns, kin terms, politeness, grammatical "
                        "gender); off = each window guesses")
    p.add_argument("--cross-evidence", default="off", choices=["auto", "off"], dest="cross_evidence",
                   help="cross-track evidence (0.5.15): auto = when a target needs gender or register, the film's own human tracks "
                        "in languages that mark it (Hebrew for gender, French for tu/vous, Korean for politeness) are shown to the "
                        "translator line by line as evidence, never output. Default off until it measures a gain")
    p.add_argument("--web-context", default=None, choices=["auto", "off"], dest="web_context",
                   help="OPT-IN (0.5.8): auto = look the title up (Wikipedia, then SearXNG at MLSUBGEN_SEARXNG_URL, then the Brave API "
                        "with BRAVE_API_KEY) and give the character sheet the cast, relationships and localised titles as priors; "
                        "only the title and search terms leave the machine. Default off (MLSUBGEN_WEB_CONTEXT)")
    p.add_argument("--asr", default=config.ASR_ENGINE, choices=["dual", "auto", "qwen", "whisper"],
                   help="ASR engine: dual = both engines decode every chunk and the LLM reconciles them (default); "
                        "auto = one engine per chunk by its language (Qwen where its aligner covers the language, "
                        "whisper elsewhere, e.g. Thai); qwen / whisper force one engine for everything")
    p.add_argument("--asr-model", default=None, help="Qwen3-ASR model id (default Qwen/Qwen3-ASR-1.7B)")
    p.add_argument("--whisper-model", default=None, help=f"faster-whisper model (default {config.ASR_MODEL_WHISPER})")
    p.add_argument("--audio-track", type=int, default=None, help="audio track index (a:N) instead of the auto-picked one (mlsubgen tracks FILE)")
    p.add_argument("--source", default=None, metavar="LANG",
                   help="skip the language detector: the audio is this language (a code from `mlsubgen languages`)")
    p.add_argument("--assume-ja", action="store_true", help="same as --source ja")
    p.add_argument("--context", default="", help="what the programme is about, names, terms — biases ASR and translation")
    p.add_argument("--context-file", default=None, help="file with the same, one paragraph")
    p.add_argument("--glossary", default=None, help="TSV file: source<TAB>target (names, terms)")
    p.add_argument("--genre", default="a documentary / interview programme",
                   help="register hint for the translator, e.g. \"a slapstick family anime\" (default: a documentary / interview programme)")
    p.add_argument("--work-dir", default=str(config.WORK_DIR))
    p.add_argument("--backend", default=None, choices=["ollama", "openai"])
    p.add_argument("--url", default=None, help="LLM server URL (default http://127.0.0.1:11434)")
    p.add_argument("--num-ctx", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--window", type=int, default=config.WINDOW_CUES, help="cues per translation request")


def parse_targets(text: str | None, warn: bool = True, allow_withheld: bool = False) -> list[str]:
    """allow_withheld: the benches measure withheld languages — that is how one earns its way back — so they may
    translate into them; a run may not."""
    out = []
    for t in (text or config.DEFAULT_TARGETS).replace(";", ",").split(","):
        t = t.strip().lower()
        if t and t not in out:
            out.append(t)
    if not out:
        raise SystemExit("no target language (--target en,th)")
    allow_withheld = allow_withheld or os.environ.get("MLSUBGEN_ALLOW_WITHHELD") == "1"   # measurement runs only
    withheld = [t for t in out if t in config.UNSUPPORTED_TARGETS and not allow_withheld]
    if withheld:
        raise SystemExit(f"{', '.join(config.LANG_NAMES[t] for t in withheld)}: not offered as a subtitle language yet — the "
                         f"translator's output in it was measured as not good enough to ship (2026-10-03); it returns when a "
                         f"translator passes on it. `mlsubgen languages` lists what is offered")
    unknown = [t for t in out if t not in config.LANG_NAMES]
    if unknown and warn:
        # not an error: the translator is simply asked for the code as written, and typesetting uses the defaults
        _log(f"[targets] unknown language code(s) {', '.join(unknown)} — the translator will be asked for them as written; "
             f"known codes: mlsubgen languages")
    return out


def check_speakers(a: argparse.Namespace) -> None:
    v = str(getattr(a, "speakers", "labels") or "labels").lower()
    if v not in ("off", "labels", "auto") and not v.isdigit():
        raise SystemExit("--speakers takes off, labels, auto or a number of speakers (e.g. --speakers 3)")
    a.speakers = v
    if getattr(a, "speaker_embedding", None):
        config.set_speaker_embedding(a.speaker_embedding)


def cmd_languages(a: argparse.Namespace) -> int:
    """The subtitle languages: every code is a target (the translator writes it) and a source (decoded by Qwen3-ASR
    where its aligner covers the language, by whisper elsewhere)."""
    if getattr(a, "profile", None) and a.profile != "auto":
        config.apply_profile(a.profile)                 # the offered languages and the routes follow the profile
    defaults = set(config.DEFAULT_TARGETS.split(","))
    from .translate import route as _route
    print(f"{'code':<5} {'language':<22} {'native':<18} {'ASR':<8} {'translator':<16} default")
    for code, name in sorted(config.TARGET_LANGS.items(), key=lambda kv: kv[1]):
        engine = "qwen" if code in config.ALIGNER_LANGS else "whisper"
        try:
            tl = _route("*", code) if code != "en" else _route("ja", "en") + "/" + _route("*", "en")
        except KeyError:
            tl = "?"
        print(f"{code:<5} {name:<22} {config.NATIVE_NAMES.get(code, ''):<18} {engine:<8} {tl:<16} {'yes' if code in defaults else ''}")
    print(f"\n{len(config.TARGET_LANGS)} languages · defaults: {config.DEFAULT_TARGETS} (MLSUBGEN_TARGETS, or --target per run)")
    print(f"translator: the route for this profile ({getattr(config, 'PROFILE', 'auto')}) — `-t NAME` forces one for every pair; the foreign-script guard "
          f"applies to every translation; `mlsubgen why FILE` says what a finished file actually got")
    print(f"withheld as targets (measured below shippable; still read as sources): "
          + ", ".join(f"{config.LANG_NAMES[c]} ({config.WITHHELD_REASON.get(c, 'measured below shippable')})"
                      for c in sorted(config.UNSUPPORTED_TARGETS, key=lambda c: config.LANG_NAMES[c])))
    return 0


HELP_TEXT = f"""\
mlsubgen {__version__} — subtitles for videos, entirely on your own machine.
The language of every stretch of speech is detected, each stretch is transcribed with that language forced (two
recognisers, reconciled by a local LLM), and every target language gets its own <video>.<lang>.srt beside the video.

USAGE
  mlsubgen [run] [PATH ...] [options]    subtitle every video in the paths (default: this folder, recursively)
  mlsubgen COMMAND [arguments]           one of the commands below;   mlsubgen help COMMAND   shows its options
  mlsubgen --version

QUICK START
  mlsubgen pull                          download the models: ASR (~8 GB) + the default translators (into Ollama)
  cd /folder/of/videos && mlsubgen       <video>.{config.DEFAULT_TARGETS.split(',')[0]}.srt for every video (default languages: {config.DEFAULT_TARGETS})
  mlsubgen --target en,th,de FOLDER      one .srt per language;   mlsubgen languages   lists the {len(config.TARGET_LANGS)} codes
  mlsubgen --source ja FOLDER            skip the language detector: the audio is Japanese
  mlsubgen --overwrite FILE              redo one file from scratch

COMMANDS
  subtitles   run        (default) subtitle the videos in the given files/folders
              scan       detect the languages only — no ASR, nothing written; one line per file
              bench      compare translators on one video or a clip:  mlsubgen bench VIDEO --clip 0:10:00-0:20:00
              lidbench   score the language detector on a multilingual film against its forced subtitle track
  models      pull       download models: mlsubgen pull | pull gemma4 | pull some/ollama:tag | pull asr | pull --all
              models     what is ready — translator presets in Ollama, ASR models in the Hugging Face cache
              languages  the {len(config.TARGET_LANGS)} subtitle languages: code, name, native name, which engine decodes it
  settings    config     the settings and where they come from;  config targets en,th  saves the default languages
              tracks     the audio and subtitle tracks of a file, and which audio track would be used
  queue       jobs       the queue of the worker service (--all for every job)
              log ID · pause ID · resume ID · cancel ID · retry ID · purge [--done] [ID ...]
              serve      the worker itself (normally started by mlsubgen-worker.service)
  web         web        the browser front end over the queue: http://<this host>:{config.WEB_PORT}  (no login — LAN only)
  care        clean      delete leftover work files and temp audio
              compare    word-level agreement between two ASR results of one file (needs --keep-work)
              selftest   exercise the text pipeline — no GPU needed

OPTIONS YOU WILL ACTUALLY USE (run)
  --target en,th         subtitle languages to write, one .srt each (default {config.DEFAULT_TARGETS})
  --source LANG          the audio IS this language — skips the detector (also rescues a file skipped as undetermined)
  --audio-track N        use audio track a:N instead of the auto-picked one (see: mlsubgen tracks FILE)
  -t NAME                one translator for every language pair: {', '.join(TRANSLATORS)}
                         (default, by pair: {', '.join(f'{a}→{b} {m}' for (a, b), m in config.TRANSLATE_ROUTES.items())})
  --model TAG            any Ollama model;   --backend openai --url http://host:port --model NAME   any OpenAI-compatible server
  --asr MODE             dual (default: both engines on every chunk, the LLM reconciles) | auto (one engine per
                         chunk by language) | qwen | whisper
  --context "..."        what the programme is about, names, terms — biases both ASR and translation
  --glossary FILE        source<TAB>target per line — pins names and terms;   --genre "..."   register hint
  --subs ja | ignore     embedded subtitles: by default an embedded target-language text track means that target is
                         done, and a spoken-language text track is the transcript (no ASR); ja = translate the
                         embedded source track yourself even if a target track exists; ignore = always transcribe
  --keep-source          also write the transcript in the spoken language as <video>.<lang>.srt
  --keep-work            keep the ASR cache after the .srt (re-translate later with -t ... --overwrite, no ASR)
  --speakers auto|N      speaker diarization (CPU; `mlsubgen pull speakers` once): a speaker change closes a cue and
                         the translator is told who is talking, so each character's register stays consistent;
                         the labels never appear in the subtitles. Off by default
  --no-recursive         stay in one folder      --dry-run   list what would be processed      --now   run here, not queued

HOW A RUN WORKS   (rounds of --batch files, default 10, so subtitles appear every round)
  1. embedded subtitles: a target-language text track → that target is done; a spoken-language text track → the transcript
  2. ffmpeg → 16 kHz audio → language detection per ~10 s of speech (whisper's probability, Qwen's decode and the
     script of the words all have to agree) → ≤ 30 s chunks of one language covering the whole timeline
  3. both engines decode every chunk with its language forced; the translator LLM reconciles where they disagree
  4. per target: cues already in the target are copied through, the rest translated in windows of 20 with context,
     glossary and register rules → typesetting → <video>.<target>.srt
  Interrupted? run it again: finished detection, decoded chunks and translated windows are reused.
  A skipped file is remembered (reason in {config.LOG_DIR / 'skipped.log'}); --source, --overwrite or --audio-track retries it.

THE QUEUE
  While mlsubgen-worker.service is up, `mlsubgen` in a folder queues the job and returns; the worker runs jobs one
  at a time, a reboot only pauses them, and `pause` holds a job across reboots until `resume`. `--now` runs in the
  foreground instead. Each job's log: mlsubgen log ID.

HARDWARE PROFILES   (picked from the GPU's memory; MLSUBGEN_PROFILE=... or --profile ... forces one; active: {config.PROFILE})
  full   20 GB and up   both ASR engines resident; 27–31B translators (qwen3.8:27b for ja→en, gemma4:31b for the rest)
  12gb   11–20 GB       whisper in int8; gemma4:12b-it-qat (7.2 GB) for every pair
  8gb    under 11 GB    one ASR engine on the card at a time; gemma4:e4b-it-qat (6.1 GB) for every pair
  The smaller translators are weaker (most of all for Japanese → English): mlsubgen bench shows by how much.

ENVIRONMENT   (a saved setting — mlsubgen config, or the web form's "make these the default" — beats the environment)
  MLSUBGEN_TARGETS       default subtitle languages (comma list)        MLSUBGEN_LLM_URL   translator server (Ollama)
  MLSUBGEN_MEDIA_ROOTS   folders the worker and the web picker may use (colon-separated)
  MLSUBGEN_HOME          state: work files, logs, the queue               MLSUBGEN_WEB_HOST / MLSUBGEN_WEB_PORT
  MLSUBGEN_PROFILE       auto (default) | full | 12gb | 8gb               HF_HUB_OFFLINE=1   never contact huggingface.co

MORE   mlsubgen help COMMAND  ·  README.md  ·  https://github.com/dodgypast/mlsubgen
"""


# ── run ──────────────────────────────────────────────────────────────────────────────────────────────────
class Prefetch:
    """Run fn(job) on a daemon thread; get() re-raises whatever it raised (SystemExit included)."""

    def __init__(self, fn, job):
        self.job = job
        self.result = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self.thread.start()

    def _run(self, fn) -> None:
        try:
            self.result = fn(self.job)
        except BaseException as e:      # noqa: BLE001 — SystemExit from ffmpeg/ffprobe must cross the thread
            self.error = e

    def get(self):
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.result

    def done(self) -> bool:
        return not self.thread.is_alive()


def cmd_run(a: argparse.Namespace) -> int:
    from .asr import Engines, words_from_dicts
    from .audio import extract_wav, load_wav
    from .pipeline import (Job, NotSupported, asr_key_for, check_source, cue_key_for, emit, labels_wanted, missing_targets,
                           speakers_cached, speakers_wanted, srt_path_for, stage_asr, stage_audio, stage_cues, stage_labels,
                           stage_lid, stage_merge, stage_speakers, stage_subs, stage_translate, usable_turns)
    from .probe import probe
    from .segment import Cue
    from .subs import code_for_tag, pick, plan_sources
    from .translate import ClientPool, route
    from .vad import cover_chunks

    if not a.now and not a.dry_run and (a.queue or worker.is_running()):
        return enqueue(a)
    if a.profile:
        config.apply_profile(a.profile)
    check_speakers(a)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    targets = parse_targets(a.target)
    source = a.source or ("ja" if a.assume_ja else None)
    videos = find_videos(a.paths or ["."], not a.no_recursive)
    if not videos:
        _log("no videos found"); return 1
    todo: list[tuple[Path, list[str]]] = []
    # The planning pass: one ffprobe per file that still lacks a target. On a big folder read off the NAS right after
    # a reboot (nothing cached) that is minutes, and a file that needs work prints nothing here — so report progress,
    # or a long stretch of to-do files looks exactly like a hang.
    scan_every, scan_secs = 25, 30.0
    n_videos, already, t_last = len(videos), 0, time.time()
    _log(f"[scan] {n_videos} video(s) found — checking which still need subtitles")
    for i, v in enumerate(videos, 1):
        if i > 1 and ((i - 1) % scan_every == 0 or time.time() - t_last >= scan_secs):
            _log(f"[scan] {i - 1}/{n_videos} checked · {len(todo)} to do · {already} already done")
            t_last = time.time()
        want = missing_targets(v, targets, a.overwrite, a.since)
        embedded: list[str] = []
        if want and a.subs == "auto":
            # a target that is embedded as a text track needs nothing — settle it here (one ffprobe) rather than
            # giving the file a batch slot to find that out; a file ffprobe cannot read is left to stage 1 to report
            try:
                pr = probe(v, a.audio_track)
                embedded = [t for t in want if pick(pr.subs, t)]
            except SystemExit:
                embedded = []
            want = [t for t in want if t not in embedded]
        if not want:
            have = [f"{t} embedded" if t in embedded else f"{t} exists" for t in targets]
            _log(f"skip (exists): {v.name} — {', '.join(have)}")
            already += 1
            continue
        long = [t for t in want if len(srt_path_for(v, t).name.encode("utf-8")) > 255]
        if long:
            _log(f"[skip] {v.name}: name too long for a .{long[0]}.srt sidecar — shorten the file name")
            record_skip(v, f"name too long for a .{long[0]}.srt sidecar (255-byte limit)")
            continue
        todo.append((v, want))
    _log(f"[scan] done: {n_videos} checked · {len(todo)} to do · {already} already done")
    if not todo:
        _log("nothing to do"); return 0
    context = a.context or (Path(a.context_file).expanduser().read_text(encoding="utf-8").strip() if a.context_file else "")
    glossary = load_glossary(a.glossary)
    if a.translator and a.translator not in TRANSLATORS and not a.model:
        raise SystemExit(f"unknown translator {a.translator!r}; presets: {', '.join(TRANSLATORS)} — or pass --model")
    presets = {}
    if a.model or a.backend or a.url or a.prompt_style or a.num_ctx or a.temperature is not None:
        for name in TRANSLATORS:
            presets[name] = resolve_translator(name, None, a.backend, a.url, a.prompt_style, a.num_ctx, a.temperature)
        if a.model:
            presets[a.translator or a.model] = resolve_translator(a.translator, a.model, a.backend, a.url, a.prompt_style,
                                                                  a.num_ctx, a.temperature)
    force_model = a.translator or (a.model if a.model else None)
    routes = (f"all pairs → {force_model}" if force_model else
              ", ".join(f"{x}→{y}: {m}" for (x, y), m in config.TRANSLATE_ROUTES.items()))
    _log(f"mlsubgen {__version__}: {len(todo)} file(s) · targets {','.join(targets)} · asr={a.asr} · translators {routes}"
         + f" · profile {config.PROFILE}" + (f" ({config.VRAM_GB:.0f} GB GPU)" if config.VRAM_GB else "")
         + (f" · source forced: {source}" if source else ""))
    if a.dry_run:
        for v, want in todo:
            _log(f"  would process {v}  → {', '.join(want)}")
        return 0

    jobs_ = [Job(v, None, a.audio_track, Path(a.work_dir).expanduser(), config.TMP_DIR, context, glossary, a.genre,
                 a.window, want, source, speakers=a.speakers, speaker_threshold=a.speaker_threshold, ocr=a.ocr, terms=a.terms,
                 register=a.register, web_context=a.web_context, cross_evidence=a.cross_evidence) for v, want in todo]
    engines = Engines(a.asr_model, a.whisper_model)
    pool = ClientPool(a.url, a.backend, presets)

    skipped: list[str] = []
    from_embedded = 0
    subtitled = 0

    def cleanup(job: Job, work_file: bool) -> None:
        """A run leaves nothing behind but the .srt: the wav goes after stage 1, the work file after the last target
        (or a stub on a skip). --keep-wav / --keep-work retain them; an interrupted run keeps the work files of the
        unfinished files so the next run resumes without redoing detection or ASR."""
        if not a.keep_wav:
            job.wav.unlink(missing_ok=True)
        if work_file and not a.keep_work:
            # the record outlives the run (0.5.13): the work file is condensed to its provenance — what `why` answers
            # from — rather than deleted; the ASR words, cues and translated text go, a few KB stay (the 2026-10-05
            # burn-in asked `why` about a finished file and was told it had never been processed)
            work.condense(job.work_file)

    def skip(job: Job, reason: str, remember: bool = True) -> None:
        if reason.startswith(job.video.name + ": "):
            reason = reason[len(job.video.name) + 2:]
        _log(f"[skip] {job.video.name}: {reason}")
        one_line = reason.splitlines()[0] if reason else "skipped"
        skipped.append(f"{job.video.name} ({one_line})")
        record_skip(job.video, one_line)
        cleanup(job, work_file=False)
        if remember:
            # The verdict stays in the work file, so a resumed or repeated run neither re-reads the file off the NAS
            # nor re-checks it. --source, --overwrite and --audio-track retry it, and so does a new config.DETECT_VERSION.
            data = work.load(job.work_file)
            data["video"] = str(job.video)
            data["skipped"] = {"reason": one_line, "when": time.strftime("%Y-%m-%d %H:%M"), "detect": config.DETECT_VERSION}
            for k in ("asr", "asr_partial", "cues", "translations"):
                data.pop(k, None)
            work.save(job.work_file, data)
        elif not a.keep_work:
            job.work_file.unlink(missing_ok=True)

    retry_skips = bool(source) or a.overwrite or a.audio_track is not None

    def remembered_skip(job: Job) -> dict | None:
        s = work.load(job.work_file).get("skipped")
        return s if (s and s.get("detect") == config.DETECT_VERSION and not retry_skips) else None

    def asr_cached(job: Job) -> bool:
        return bool((work.load(job.work_file).get("asr") or {}).get(asr_key_for(engines, context, a.asr)))

    # Background extraction of the next file's audio while the GPU works on the current one: reading a
    # multi-gigabyte file off the NAS is often the slowest part of stage 1 for talk-light files.
    # Daemon threads, not an executor, so Ctrl-C is not held up by a running ffmpeg (a stray wav in
    # ~/mlsubgen/tmp is the worst case; `mlsubgen clean` removes it).
    prefetch: dict[Path, Prefetch] = {}

    def _extract(job: Job):
        pr = probe(job.video, job.audio_track)
        if pr.chosen is None:
            return None
        use_ocr = a.ocr != "off"
        satisfied, source = plan_sources(job.video, pr.subs, job.targets, a.subs, code_for_tag(pr.chosen.language), ocr=use_ocr)
        from .subs import bitmap_targets as _bt
        satisfied = satisfied + list(_bt(pr.subs, job.targets, satisfied, ocr=use_ocr))
        if source is not None or len(satisfied) == len(job.targets):
            return None                                     # the text (or OCR) route needs no audio
        return extract_wav(job.video, pr.chosen.index, job.wav, None, None, pr.chosen.duration or pr.duration)

    def prefetch_after(i: int, batch: list[Job]) -> None:
        for nxt in batch[i + 1:i + 2]:
            if nxt.video not in prefetch and not asr_cached(nxt) and not remembered_skip(nxt):
                prefetch[nxt.video] = Prefetch(_extract, nxt)

    def merge_pass(pending: list) -> None:
        """Dual mode, after the engines have left the GPU: the LLM reconciles the two transcripts of every file
        of the round, then the cues are built (with the wav still present for the energy gate)."""
        client = None
        name = force_model or config.DEFAULT_TRANSLATOR
        try:
            ok, msg = pool.client(name).available()
            if ok:
                client = pool.use(name)
                _log(f"\n[merge] reconciling the two transcripts with {name} for {len(pending)} file(s)")
            else:
                _log(f"\n[merge] ⚠ {name}: {msg} — Qwen's transcript stands for this round, whisper's is kept in the work files")
        except SystemExit as e:
            _log(f"\n[merge] ⚠ {e} — Qwen's transcript stands for this round")
        try:
            for job, akey, spans in pending:
                data = work.load(job.work_file)
                _log(f"\n=== {job.video.name} ⇄ merge")
                words = stage_merge(job, data, akey, client)
                audio = load_wav(job.wav) if job.wav.exists() else None
                stage_cues(job, data, akey, words, spans, audio)
                cleanup(job, work_file=False)
        finally:
            for job, _, _ in pending:
                cleanup(job, work_file=False)

    def stage1(batch: list[Job]) -> dict[Path, str]:
        """Audio, VAD, language detection, ASR, cues for every file of the batch; returns video → cue key."""
        nonlocal from_embedded
        keys: dict[Path, str] = {}
        pending: list = []
        try:
            for i, job in enumerate(batch):
                prefetch_after(i, batch)
                data = work.reopen(work.load(job.work_file))      # a condensed record keeps its complete caches only
                _log(f"\n=== {job.video.name}")
                rem = remembered_skip(job)
                if rem:
                    _log(f"[skip] {job.video.name}: {rem['reason']} (remembered from {rem['when']}; "
                         f"--source, --overwrite or --audio-track retries it)")
                    skipped.append(f"{job.video.name} ({rem['reason']}, remembered)")
                    record_skip(job.video, rem["reason"])
                    continue
                data.pop("skipped", None)                 # a retry: the old verdict must not survive a later save
                # nothing can be written beside the video: say so now, before the card is used (0.5.14 — a read-only
                # folder, a wrong PUID/PGID in Docker, a share mounted read-only all look the same from here)
                if not os.access(job.video.parent, os.W_OK):
                    skip(job, "the folder is not writable (permissions, PUID/PGID, or a read-only mount) — the .srt could not be written beside the video", remember=False)
                    continue
                try:
                    try:
                        done_targets, key = stage_subs(job, data, a.subs, job.targets)
                        if done_targets:
                            from_embedded += len(done_targets)
                            job.targets = [t for t in job.targets if t not in done_targets]
                            if not job.targets:
                                _log("[subs] nothing to write: every target is already embedded")
                                # the work file goes when nothing was made here — but an OCR'd target IS something made
                                # here, and its record (track, engine, gate) is what `why` answers from (0.5.0.8)
                                cleanup(job, work_file=not (work.load(job.work_file).get("ocr_targets") if job.work_file.is_file() else False))
                                continue
                        if key:
                            keys[job.video] = key                 # cues are in the work file; stage 2 translates them
                            # speaker labels from the audio for a text track's cues (0.5.4): the diarizer alone, no
                            # transcription — so the character sheet applies to a known speaker, not a guessed one
                            if labels_wanted(job, key, data):
                                try:
                                    entry = prefetch.pop(job.video, None)
                                    stage_labels(job, data, key, prefetched=entry.get() if entry is not None else None)
                                except SystemExit as e:
                                    _log(f"[labels] ⚠ {e} — the cues stay untagged")
                                except Exception as e:                    # noqa: BLE001
                                    _log(f"[labels] ⚠ {type(e).__name__}: {e} — the cues stay untagged")
                                finally:
                                    cleanup(job, work_file=False)         # the wav
                            continue
                        pre = None
                        entry = prefetch.pop(job.video, None)
                        if entry is not None:
                            pre = entry.get()                     # an ffmpeg failure surfaces here as SystemExit
                        akey = asr_key_for(engines, context, a.asr)
                        cached = bool((data.get("asr") or {}).get(akey))
                        need_spk = speakers_wanted(job) and not speakers_cached(job, data)
                        pr, spans = stage_audio(job, data, need_wav=not cached or need_spk, prefetched=pre)
                    except SystemExit as e:
                        skip(job, str(e)); continue
                    speech = sum(sp.dur for sp in spans)
                    if not spans or speech < config.LID_MIN_SPEECH_SEC:
                        skip(job, "no speech at all" if not spans else f"only {speech:.0f}s of speech in the whole file"); continue
                    audio = None
                    turns = []
                    if speakers_wanted(job):                      # speakers (0.4.0): CPU, before the detector (0.4.4)
                        if need_spk:
                            audio = load_wav(job.wav)
                        stage_speakers(job, data, audio)
                        turns = usable_turns(data)
                    if not cached:
                        if audio is None:
                            audio = load_wav(job.wav)
                        try:
                            res = stage_lid(job, data, audio, spans, engines, turns=turns)
                            check_source(res)
                        except NotSupported as e:
                            skip(job, str(e)); continue
                        total_sec = len(audio) / 16000.0
                        chunks = cover_chunks(audio, res.spans, total_sec, dominant=res.dominant)
                        covered = sum(c.dur for c in chunks)
                        _log(f"[chunks] {len(chunks)} chunk(s) covering {covered / 60:.1f} of {total_sec / 60:.1f} min "
                             f"(silence skipped {(total_sec - covered) / 60:.1f} min; VAD flagged {sum(s.dur for s in spans) / 60:.1f} min): "
                             + ", ".join(f"{lang} {n}" for lang, n in sorted({c.lang: sum(1 for x in chunks if x.lang == c.lang) for c in chunks}.items())))
                        words = stage_asr(job, data, engines, audio, chunks, a.asr)
                    else:
                        words = words_from_dicts(data["asr"][akey]["words"])
                        _log(f"[asr] cached ({akey})")
                    keys[job.video] = cue_key_for(job, akey)      # cues (and translations) are a separate set with speakers on
                    if data["asr"][akey].get("merge") == "pending":
                        pending.append((job, akey, spans))       # cues after the merge; the wav stays for the energy gate
                        continue
                    stage_cues(job, data, akey, words, spans, audio)
                finally:
                    if not any(j is job for j, _, _ in pending):
                        cleanup(job, work_file=False)   # the wav, even when interrupted
        finally:
            engines.close()                         # free the GPU for the translator
        if pending:
            merge_pass(pending)
        return keys

    def stage2(batch: list[Job], keys: dict[Path, str]) -> bool:
        """Translate every file of the batch into each of its targets (target by target, so the translator swaps
        as rarely as possible), then write the .srt files."""
        nonlocal subtitled
        todo2 = [j for j in batch if j.video in keys]
        if not todo2:
            return True
        # only the translators the batch's cues actually need: a Japanese film with English targets needs the ja→en
        # model, not one for every language we could in principle detect; a cue already in the target needs none
        pairs: set[tuple[str, str]] = set()
        for j in todo2:
            langs = {d.get("lang", "ja") for d in ((work.load(j.work_file).get("cues") or {}).get(keys[j.video]) or {}).get("cues", [])}
            pairs |= {(src, t) for t in j.targets for src in langs if src != t}
        needed = {route(src, t, force_model) for src, t in pairs}
        if needed:
            _log("[tl] translators for this round: " + ", ".join(f"{s}→{t}: {route(s, t, force_model)}" for s, t in sorted(pairs)))
        for name in sorted(needed):
            ok, msg = pool.client(name).available()
            if ok:
                continue
            # a route's model is missing: the default translator stands in for it rather than failing the round
            fallback = config.DEFAULT_TRANSLATOR
            fb_ok, fb_msg = (pool.client(fallback).available() if fallback != name else (False, msg))
            if not fb_ok:
                _log(f"[tl] {name}: {msg}" + (f"; fallback {fallback}: {fb_msg}" if fallback != name else "")); return False
            _log(f"[tl] ⚠ {name}: {msg} — {fallback} stands in for its language pairs this round")
            pool.aliases[name] = fallback
        results: dict[tuple[Path, str], list[Cue]] = {}
        for target in sorted({t for j in todo2 for t in j.targets}):
            for job in todo2:
                if target not in job.targets:
                    continue
                data = work.load(job.work_file)
                key = keys[job.video]
                if key not in (data.get("cues") or {}):
                    skip(job, f"work file vanished since stage 1 ({job.work_file.name}) — re-run for this file", remember=False)
                    job.targets = []; continue
                cues = [Cue.from_dict(d) for d in data["cues"][key]["cues"]]
                if not cues:
                    skip(job, "no usable speech — no subtitle written"); job.targets = []; continue
                for cl in pool.clients.values():
                    cl.usage = type(cl.usage)()
                _log(f"\n=== {job.video.name} → {target}")
                results[(job.video, target)] = stage_translate(job, data, key, cues, target, pool, force=a.force_translate,
                                                               force_model=force_model)
        for job in todo2:
            wrote = 0
            for target in job.targets:
                done = results.get((job.video, target))
                if done is None:
                    continue
                out = srt_path_for(job.video, target)
                try:
                    n = emit(done, out, target)
                except OSError as e:                      # a name too long for the filesystem, a folder gone read-only mid-run
                    _log(f"[srt] ⚠ {out.name}: could not be written ({e.strerror or e}) — the translation stays in the work file")
                    skipped.append(f"{job.video.name} ({target}: {e.strerror or e})"); record_skip(job.video, f"{target}: {e.strerror or e}")
                    continue
                wrote += 1
                _log(f"[srt] {out.name}: {n} cues")
            if wrote:
                subtitled += 1
            if a.keep_source and job.targets and results.get((job.video, job.targets[0])) is not None:
                data = work.load(job.work_file)
                cues = [Cue.from_dict(d) for d in data["cues"][keys[job.video]]["cues"]]
                for src_lang in sorted({c.lang for c in cues}):
                    if src_lang in job.targets:
                        continue
                    part = [Cue(c.idx, c.start, c.end, c.ja, c.ja, lang=c.lang) for c in cues if c.lang == src_lang]
                    emit(part, srt_path_for(job.video, src_lang), src_lang)
            if job.targets:
                cleanup(job, work_file=True)
        pool.unload_all()                           # free the GPU for the next batch's ASR
        return True

    # Rounds of --batch files: ASR them all (one model load), then translate them all (one model load), so
    # subtitles appear every round instead of only after every file's ASR — on a big folder that is the
    # difference between the first .srt in an hour and in days. --batch 0 = one round for everything.
    batches = [jobs_[i:i + a.batch] for i in range(0, len(jobs_), a.batch)] if a.batch > 0 else [jobs_]
    try:
        for bi, batch in enumerate(batches, 1):
            if len(batches) > 1:
                _log(f"\n━━━ round {bi}/{len(batches)}: {len(batch)} file(s) ━━━")
            keys = stage1(batch)
            if bi < len(batches):
                prefetch_after(-1, batches[bi])    # next round's first file, while the translator runs
            if not stage2(batch, keys):
                return 2
    finally:
        for pf in prefetch.values():                # extracted but never used (interrupted run)
            if pf.done() and not a.keep_wav:
                pf.job.wav.unlink(missing_ok=True)
    _log(f"\ndone: {subtitled} file(s) subtitled" + (f", {from_embedded} target(s) already embedded" if from_embedded else "")
         + (f", {len(skipped)} skipped: " + "; ".join(skipped) + f"\n(skips are logged in {config.LOG_DIR / 'skipped.log'})"
            if skipped else ""))
    return 0


# ── scan / compare ───────────────────────────────────────────────────────────────────────────────────────
def cmd_scan(a: argparse.Namespace) -> int:
    """Language detection only: probe, extract, VAD, LID — no ASR, nothing written beside the videos. The verdict is
    saved in the work file, so a later run reuses it."""
    from .asr import Engines
    from .audio import load_wav
    from .pipeline import Job, stage_audio, stage_lid

    videos = find_videos(a.paths or ["."], not a.no_recursive)
    if not videos:
        _log("no videos found"); return 1
    if a.profile:
        config.apply_profile(a.profile)
    engines = Engines(a.asr_model, a.whisper_model)
    counts: dict[str, int] = {}
    low: list[str] = []
    rows: list[str] = []
    print(f"{'file':<60} {'dominant':<9} {'conf':>5} {'unc':>4}  speech   languages")
    try:
        for v in videos:
            job = Job(v, None, a.audio_track, Path(a.work_dir).expanduser(), config.TMP_DIR)
            data = work.load(job.work_file)
            name = v.name if len(v.name) <= 58 else v.name[:55] + "…"
            try:
                try:
                    pr, spans = stage_audio(job, data, need_wav=not (data.get("lid") or {}).get("version") == config.LID_VERSION)
                    if not spans:
                        row = f"{name:<60} {'—':<9} {'':>5} {'':>4}  no speech"
                        counts["no speech"] = counts.get("no speech", 0) + 1
                        print(row); rows.append(row); continue
                    audio = load_wav(job.wav) if job.wav.exists() else None
                    if audio is None and not ((data.get("lid") or {}).get("version") == config.LID_VERSION):
                        raise SystemExit("wav missing")
                    res = stage_lid(job, data, audio, spans, engines)
                except SystemExit as e:
                    row = f"{name:<60} {'error':<9} {'':>5} {'':>4}  {str(e).splitlines()[0][:80]}"
                    counts["error"] = counts.get("error", 0) + 1
                    print(row); rows.append(row); continue
                total = res.total_speech or 1.0
                langs = ", ".join(f"{l} {sec / total * 100:.0f}%" for l, sec in sorted(res.seconds.items(), key=lambda kv: kv[1], reverse=True))
                dom = res.dominant or "unknown"
                row = f"{name:<60} {dom:<9} {res.confident_windows:>5} {res.uncertain_windows:>4}  {total / 60:5.1f}m  {langs}"
                counts[dom] = counts.get(dom, 0) + 1
                if not res.dominant or res.confident_windows < 2 or len(res.seconds) > 1:
                    low.append(row)
                print(row); rows.append(row)
            finally:
                if not a.keep_wav:
                    job.wav.unlink(missing_ok=True)
    finally:
        engines.close()
    print(f"\n{len(videos)} file(s): " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)))
    if low:
        print(f"\nworth a look ({len(low)}: unknown, fewer than 2 confident windows, or more than one language):")
        for r in low:
            print("  " + r)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    """Word-level agreement between two ASR results in one file's work file (e.g. the reference --assume-ja run
    of the old code vs the detector-driven run)."""
    import difflib
    from .pipeline import Job
    video = Path(a.video).expanduser()
    if not video.is_file():
        _log(f"not found: {video}"); return 1
    data = work.load(Job(video, work_dir=Path(a.work_dir).expanduser()).work_file)
    entries = data.get("asr") or {}
    if len(entries) < 2:
        _log(f"need two ASR entries in the work file, found {len(entries)}: {list(entries)} (run with --keep-work)"); return 1
    keys = list(entries)
    ka, kb = (a.keys.split(",") if a.keys else keys[-2:])
    ta = "".join(w["text"] for w in entries[ka]["words"])
    tb = "".join(w["text"] for w in entries[kb]["words"])
    ratio = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
    print(f"{video.name}\n  A: {ka}  ({len(entries[ka]['words'])} units, {len(ta)} chars)\n  B: {kb}  ({len(entries[kb]['words'])} units, {len(tb)} chars)")
    print(f"  character-level agreement: {ratio * 100:.1f}%")
    la = {c.get('lang') for c in entries[ka].get('chunks', [])}; lb = {c.get('lang') for c in entries[kb].get('chunks', [])}
    print(f"  chunk languages A {sorted(x for x in la if x)} B {sorted(x for x in lb if x)}; chunk notes B: {entries[kb].get('mismatches', 0)}")
    if ratio < 0.98:
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes()[:200]:
            if tag != "equal" and (i2 - i1 > 6 or j2 - j1 > 6):
                print(f"  {tag:<8} A[{i1}:{i2}] {ta[i1:i2][:60]!r}  →  B {tb[j1:j2][:60]!r}")
    return 0


def enqueue(a: argparse.Namespace) -> int:
    """Queue this run for the mlsubgen-worker service instead of running it here. Paths are stored absolute (the
    worker has no idea what folder you were in); everything else is passed to the child untouched."""
    raw = [t for t in a.argv[1:] if t != "--queue"]
    given = list(a.paths or [])
    paths = [str(Path(p).expanduser().resolve()) for p in (given or ["."])]
    opts: list[str] = []
    for t in raw:
        if t in given:
            given.remove(t)                     # each positional once; everything else is an option or its value
        else:
            opts.append(t)
    conn = jobs.connect()
    try:
        jid = jobs.add(conn, paths, opts)
    finally:
        conn.close()
    _log(f"queued as job {jid} for the mlsubgen-worker service: {jobs.make_label(paths, opts)}\n"
         f"  mlsubgen jobs · mlsubgen log {jid} · mlsubgen cancel {jid}      (to run it here instead: mlsubgen --now …)")
    return 0


# ── the queue: serve | jobs | log | cancel | retry ───────────────────────────────────────────────────────
def cmd_serve(a: argparse.Namespace) -> int:
    return worker.Worker(poll=a.poll).serve()


def cmd_web(a: argparse.Namespace) -> int:
    from .web import serve
    return serve(a.host, a.port)


def cmd_jobs(a: argparse.Namespace) -> int:
    conn = jobs.connect()
    try:
        print(jobs.fmt_table(jobs.list_jobs(conn, all_jobs=a.all)))
    finally:
        conn.close()
    print(f"\nworker: {'running' if worker.is_running() else 'not running (jobs wait; `mlsubgen --now` runs here)'}")
    return 0


def cmd_log(a: argparse.Namespace) -> int:
    conn = jobs.connect()
    try:
        j = jobs.get(conn, a.id)
    finally:
        conn.close()
    if j is None:
        print(f"no job {a.id}"); return 1
    if not j.log:
        print(f"job {a.id} has not started yet ({j.status}{': ' + j.note if j.note else ''})"); return 0
    print(f"job {a.id} ({j.status}): {j.log}\n  tail -f {j.log}\n")
    try:
        lines = Path(j.log).read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-a.lines:]))
    except OSError as e:
        print(f"(cannot read it: {e})")
    return 0


def cmd_cancel(a: argparse.Namespace) -> int:
    conn = jobs.connect()
    try:
        print(f"job {a.id}: {jobs.cancel(conn, a.id)}")
    finally:
        conn.close()
    return 0


def cmd_retry(a: argparse.Namespace) -> int:
    conn = jobs.connect()
    try:
        print(f"job {a.id}: {jobs.retry(conn, a.id)}")
    finally:
        conn.close()
    return 0


def cmd_pause(a: argparse.Namespace) -> int:
    conn = jobs.connect()
    try:
        print(f"job {a.id}: {jobs.pause(conn, a.id)}")
    finally:
        conn.close()
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    conn = jobs.connect()
    try:
        print(f"job {a.id}: {jobs.resume(conn, a.id)}")
    finally:
        conn.close()
    return 0


def cmd_purge(a: argparse.Namespace) -> int:
    """Drop finished jobs from the queue listing (their logs stay in LOG_DIR)."""
    conn = jobs.connect()
    try:
        if a.ids:
            n = jobs.purge(conn, ids=a.ids)
            what = f"job(s) {', '.join(map(str, a.ids))} (only finished ones are removed)"
        else:
            n = jobs.purge(conn, statuses=jobs.FINAL if a.done else (jobs.FAILED, jobs.CANCELLED))
            what = "done, failed and cancelled jobs" if a.done else "failed and cancelled jobs (--done removes finished ones too)"
    finally:
        conn.close()
    print(f"removed {n} {what}")
    return 0


# ── bench ────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_bench(a: argparse.Namespace) -> int:
    from .asr import Engines
    from .audio import load_wav
    from .bench import align_reference, chrf, write_html
    from .pipeline import (Job, NotSupported, asr_key_for, check_source, cue_key_for, emit, speakers_cached, speakers_wanted,
                           stage_asr, stage_audio, stage_cues, stage_lid, stage_merge, stage_speakers, stage_translate,
                           usable_turns)
    from .segment import Cue
    from .srt import read_srt
    from .translate import ClientPool
    from .vad import cover_chunks

    video = Path(a.video).expanduser()
    if not video.is_file():
        _log(f"not found: {video}"); return 1
    if a.profile:
        config.apply_profile(a.profile)
    check_speakers(a)
    context = a.context or (Path(a.context_file).expanduser().read_text(encoding="utf-8").strip() if a.context_file else "")
    glossary = load_glossary(a.glossary)
    clip = clip_arg(a.clip)
    names = [n.strip() for n in a.translators.split(",") if n.strip()]
    target = parse_targets(a.target, allow_withheld=True)[0]        # the bench measures withheld languages too
    source = a.source or ("ja" if a.assume_ja else None)
    job = Job(video, clip, a.audio_track, Path(a.work_dir).expanduser(), config.TMP_DIR, context, glossary, a.genre,
              a.window, [target], source, speakers=a.speakers, speaker_threshold=a.speaker_threshold, terms=a.terms,
              register=a.register, web_context=a.web_context, cross_evidence=a.cross_evidence)
    data = work.load(job.work_file)
    engines = Engines(a.asr_model, a.whisper_model)
    key = asr_key_for(engines, context, a.asr)
    audio = None
    try:
        asr_cached = bool((data.get("asr") or {}).get(key))
        need_spk = speakers_wanted(job) and not speakers_cached(job, data)
        pr, spans = stage_audio(job, data, need_wav=not asr_cached or need_spk)
        turns = []
        if speakers_wanted(job):                              # before the detector (0.4.4)
            if need_spk:
                audio = load_wav(job.wav)
            stage_speakers(job, data, audio)
            turns = usable_turns(data)
        if asr_cached:
            from .asr import words_from_dicts
            words = words_from_dicts(data["asr"][key]["words"]); _log(f"[asr] cached ({key})")
        else:
            if audio is None:
                audio = load_wav(job.wav)
            try:
                res = stage_lid(job, data, audio, spans, engines, turns=turns)
                check_source(res)
            except NotSupported as e:
                _log(f"[lid] {e}"); return 1
            words = stage_asr(job, data, engines, audio, cover_chunks(audio, res.spans, len(audio) / 16000.0, dominant=res.dominant), a.asr)
    finally:
        engines.close()
    if (data.get("asr") or {}).get(key, {}).get("merge") == "pending":
        # dual mode: the engines have left the GPU, the LLM reconciles the two transcripts (the bench's own translator list
        # names the referee: the first available preset)
        presets0 = {n: resolve_translator(n, None, a.backend, a.url, None, a.num_ctx, a.temperature) for n in names}
        pool0 = ClientPool(a.url, a.backend, presets0)
        client = None
        for n in names:
            ok, _ = pool0.client(n).available()
            if ok:
                client = pool0.use(n); break
        words = stage_merge(job, data, key, client)
        pool0.unload_all()
    cues = stage_cues(job, data, key, words, spans, audio)
    if not cues:
        _log("no speech found in the clip"); return 1
    key = cue_key_for(job, key)                       # cues and translations: the speaker-labelled set is its own entry

    bench_dir = Path(a.out_dir).expanduser()
    bench_dir.mkdir(parents=True, exist_ok=True)
    tag = video.stem + (f".{int(clip[0])}-{int(clip[1])}" if clip else "") + (f".spk-{job.speakers}" if speakers_wanted(job) else "") \
        + (".terms" if (job.terms or "auto") != "off" and len(cues) >= config.TERMS_MIN_CUES else "") \
        + (".chars" if (job.register or "auto") != "off" and len(cues) >= config.CHARACTERS_MIN_CUES else "")
    reference = None
    ref_texts = None
    if a.reference:
        reference = read_srt(Path(a.reference).expanduser())
        if clip:
            reference = [c for c in reference if c.end > clip[0] and c.start < clip[1]]
            for c in reference:
                c.start -= clip[0]; c.end -= clip[0]
        ref_texts = align_reference(cues, reference)
        # cue-boundary agreement with the human subtitler (0.4.0): a human breaks lines at speaker changes, so this
        # is the measure that shows whether speaker labels moved our boundaries the right way
        ref_ends = sorted(c.end for c in reference)
        import bisect
        tol = 0.4
        near = 0
        for c in cues:
            i = bisect.bisect_left(ref_ends, c.end)
            cands = [ref_ends[j] for j in (i - 1, i) if 0 <= j < len(ref_ends)]
            if cands and min(abs(c.end - r) for r in cands) <= tol:
                near += 1
        agreement = 100.0 * near / max(1, len(cues))
        _log(f"[bench] cue boundaries: {agreement:.0f}% of our {len(cues)} cue ends fall within {tol} s of one of the "
             f"reference's {len(reference)} cue ends" + (f"  (speaker labels on: --speakers {job.speakers})" if speakers_wanted(job) else ""))

    results: dict[str, list[Cue]] = {}
    stats: dict[str, dict] = {}
    presets = {n: resolve_translator(n, None, a.backend, a.url, None, a.num_ctx, a.temperature) for n in names}
    for name in names:
        pool = ClientPool(a.url, a.backend, presets)
        client = pool.client(name)
        ok, msg = client.available()
        if not ok:
            _log(f"[bench] {name}: {msg} — skipped"); continue
        t0 = time.time()
        done = stage_translate(job, data, key, cues, target, pool, force=True, force_model=name)
        elapsed = time.time() - t0
        pool.unload_all()
        results[name] = done
        n = emit(done, bench_dir / f"{tag}.{name}.{target}.srt", target)
        s = {"summary": f"{client.tr.model} · {elapsed:.0f}s · {client.usage.completion_tokens / max(client.usage.seconds, 0.01):.1f} tok/s"
                        f" · fallbacks {client.usage.fallbacks}", "elapsed": round(elapsed, 1), "srt_cues": n}
        if ref_texts:
            s["chrf"] = chrf([c.en for c in done], ref_texts)
        stats[name] = s
        _log(f"[bench] {name}: {s['summary']}{'  chrF++ ' + str(s['chrf']) if s.get('chrf') is not None else ''}")
    if not results:
        return 2
    html_out = bench_dir / f"{tag}.bench.html"
    shown = [Cue.from_dict(c.to_dict()) for c in cues]
    for c in shown:                                   # the page shows the speaker label the translator was given
        if c.speaker:
            c.ja = f"[{c.speaker}] {c.ja}"
    write_html(html_out, f"mlsubgen bench — {video.name}" + (" — with speaker labels" if speakers_wanted(job) else ""),
               shown, results, stats, ref_texts)
    _log(f"\n[bench] {html_out}")
    for n, s in stats.items():
        _log(f"  {n:<16} {s['summary']}{'  chrF++ ' + str(s['chrf']) if s.get('chrf') is not None else ''}")
    return 0


# ── clean / tracks / models / selftest ───────────────────────────────────────────────────────────────────
def cmd_clean(a: argparse.Namespace) -> int:
    """Remove every work file and temp wav (leftovers of interrupted runs, or runs of older versions)."""
    n = size = 0
    for d, pat in ((config.WORK_DIR, "*.json"), (config.WORK_DIR, "*.json.tmp"), (config.TMP_DIR, "*.wav"), (config.TMP_DIR, "*.part.wav")):
        for p in sorted(d.glob(pat)) if d.is_dir() else []:
            size += p.stat().st_size
            p.unlink()
            n += 1
    print(f"removed {n} file(s), {size / 1e6:.0f} MB — {config.WORK_DIR} and {config.TMP_DIR} are empty"
          f" (bench results in {config.BENCH_DIR} and {config.LOG_DIR} untouched)")
    return 0


def cmd_tracks(a: argparse.Namespace) -> int:
    from .probe import describe_tracks, probe
    for p in find_videos(a.paths, a.recursive):
        try:
            pr = probe(p)
        except SystemExit as e:
            print(f"{p}\n  {e}")
            continue
        print(f"{p}\n  duration {pr.duration / 60:.1f} min · chosen: {pr.reason}\n{describe_tracks(pr)}")
    return 0


def cmd_models(a: argparse.Namespace) -> int:
    from . import models
    st = models.status(a.url)
    o = st["ollama"]
    print(f"profile: {config.PROFILE}" + (f" ({config.VRAM_GB:.0f} GB GPU detected)" if config.VRAM_GB else " (no GPU detected)")
          + f" — MLSUBGEN_PROFILE or --profile to force one of: {', '.join(config.PROFILES)}")
    print(f"translators (Ollama at {o['url']}: {'reachable' if o['reachable'] else 'UNREACHABLE'}):")
    for t in st["translators"]:
        state = f"ready ({t['size'] / 1e9:.1f} GB)" if t["ready"] else "not pulled"
        print(f"  {t['name']:<16} {t['model']:<40} {state:<16} — {t['note']}")
    print("  routes: " + ", ".join(f"{r['pair']}: {r['preset']}" for r in st["routes"]))
    print(f"\nASR models (Hugging Face cache {models.hub_dir()}{', OFFLINE' if st['offline'] else ''}):")
    for m in st["asr"]:
        state = f"ready ({m['size'] / 1e9:.1f} GB)" if m["ready"] else "not downloaded"
        print(f"  {m['name']:<26} {m['model']:<40} {state}")
    from . import speakers as _spk
    ok, why = _spk.available()
    print(f"\nspeaker diarization (optional, --speakers; CPU; {config.SPEAKER_MODEL_DIR}): {'ready' if ok else why}")
    for m in st["speakers"]:
        state = f"ready ({m['size'] / 1e6:.0f} MB)" if m["ready"] else "not downloaded — mlsubgen pull speakers"
        print(f"  {m['name']:<42} {state}")
    missing = [t["name"] for t in st["translators"] if not t["ready"] and t["name"] in set(config.TRANSLATE_ROUTES.values())]
    missing += [m["name"] for m in st["asr"] if not m["ready"]]
    if missing:
        print(f"\nmissing for a default run: {', '.join(missing)}  →  mlsubgen pull")
    return 0


def _merge_intervals(items: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(items):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    j = 0
    for s, e in a:
        while j < len(b) and b[j][1] < s:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            total += max(0.0, min(e, b[k][1]) - max(s, b[k][0]))
            k += 1
    return total


def _intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of the merged intervals `a` that lie inside the merged intervals `b`."""
    out = []
    for s, e in a:
        for bs, be in b:
            if be <= s:
                continue
            if bs >= e:
                break
            out.append((max(s, bs), min(e, be)))
    return out


def _merge_gap(items: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Merge intervals separated by less than `gap` seconds: the blocks of one foreign-language conversation,
    whose cues have pauses between them."""
    out: list[tuple[float, float]] = []
    for s, e in sorted(items):
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _pgs_intervals(video: Path, index: int) -> list[tuple[float, float]]:
    """The on-screen intervals of a bitmap (PGS) subtitle track, without OCR: a display set whose presentation
    composition carries one or more objects puts a subtitle up, one with no objects takes it down. Enough for a
    forced track to serve as "foreign speech here" (2026-10-01: a UHD release with 30 bitmap tracks and no text)."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".sup", delete=False) as f:
        sup = Path(f.name)
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video), "-map", f"0:s:{index}", "-c", "copy",
                        "-f", "sup", str(sup)], check=True, capture_output=True)
        data = sup.read_bytes()
    finally:
        sup.unlink(missing_ok=True)
    out: list[tuple[float, float]] = []
    open_at: float | None = None
    pos = 0
    while pos + 13 <= len(data):
        if data[pos:pos + 2] != b"PG":
            pos += 1
            continue
        pts = int.from_bytes(data[pos + 2:pos + 6], "big") / 90000.0
        seg_type = data[pos + 10]
        size = int.from_bytes(data[pos + 11:pos + 13], "big")
        payload = data[pos + 13:pos + 13 + size]
        if seg_type == 0x16 and len(payload) >= 11:           # presentation composition segment
            n_objects = payload[10]
            if n_objects > 0 and open_at is None:
                open_at = pts
            elif n_objects == 0 and open_at is not None:
                if pts > open_at:
                    out.append((open_at, pts))
                open_at = None
        pos += 13 + size
    return out


def _switches(truth_blocks: list[tuple[float, float]], our_blocks: list[tuple[float, float]],
              tol: float = 3.0, far: float = 5.0) -> dict:
    """Language changes: every block edge is a switch (into the foreign language at its start, back at its end).
    Recall = reference switches we reproduced within `tol` seconds; latency = how far off; false = our switches
    with no reference switch within `far` seconds."""
    rec = {"reference_switches": 2 * len(truth_blocks), "detected": 0, "latencies": [], "false": 0}
    for side in (0, 1):
        t_edges = [b[side] for b in truth_blocks]
        o_edges = [b[side] for b in our_blocks]
        for t in t_edges:
            d = min((abs(o - t) for o in o_edges), default=None)
            if d is not None and d <= tol:
                rec["detected"] += 1
                rec["latencies"].append(round(d, 2))
        rec["false"] += sum(1 for o in o_edges if not any(abs(o - t) <= far for t in t_edges))
    lat = sorted(rec["latencies"])
    rec["median_latency"] = lat[len(lat) // 2] if lat else None
    rec["switch_recall"] = rec["detected"] / rec["reference_switches"] if rec["reference_switches"] else 0.0
    return rec


def cmd_lidbench(a: argparse.Namespace) -> int:
    """How well does the language detector find the foreign-language stretches of a multilingual film? The ground
    truth is the release's FORCED subtitle track — subtitles only where a language other than the main one is
    spoken — whose cue intervals mean "foreign speech here". Reports recall (how much of that time we labelled as
    not the dominant language), precision (how much of our foreign-labelled time lies inside it), the languages we
    found, and the biggest misses and false alarms with their timestamps, so the detector can be compared before
    and after a change on the same film (2026-10-01; the measurement for the speaker-aware detector)."""
    from .asr import Engines
    from .audio import load_wav
    from .pipeline import Job, speakers_wanted, stage_audio, stage_lid, stage_speakers, usable_turns
    from .probe import describe_tracks, probe
    from .srt import read_srt
    from .subs import extract_track

    video = Path(a.video).expanduser()
    if not video.is_file():
        _log(f"not found: {video}"); return 1
    if a.profile:
        config.apply_profile(a.profile)
    clip = clip_arg(a.clip)
    pr = probe(video, a.audio_track)
    from types import SimpleNamespace
    if a.reference:
        ref = read_srt(Path(a.reference).expanduser())
        where = Path(a.reference).name
    else:
        tracks = list(pr.subs)
        if a.forced_track is not None:
            tracks = [t for t in tracks if t.index == a.forced_track]
        else:
            tracks = [t for t in tracks if getattr(t, "forced", False) or "forced" in (t.title or "").lower()]
        # a text forced track first; a bitmap one still gives the timings (no OCR needed for "foreign speech here")
        tracks.sort(key=lambda t: 0 if t.is_text else 1)
        if not tracks:
            _log("no forced subtitle track found — pass --forced-track N (see `mlsubgen tracks`) or --reference forced.srt")
            _log(describe_tracks(pr)); return 1
        t = tracks[0]
        if t.is_text:
            tmp = config.TMP_DIR / f"{video.stem[:60]}.forced.srt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            n = extract_track(video, t.index, tmp, t.codec)
            ref = read_srt(tmp)
            where = f"s:{t.index} {t.codec}{', ' + t.title if t.title else ''} ({n} raw cues)"
        else:
            ref = [SimpleNamespace(start=s, end=e, text="") for s, e in _pgs_intervals(video, t.index)]
            where = f"s:{t.index} {t.codec} (bitmap — timings only, no OCR){', ' + t.title if t.title else ''} ({len(ref)} display sets)"
    if clip:
        ref = [c for c in ref if c.end > clip[0] and c.start < clip[1]]
        for c in ref:
            c.start -= clip[0]; c.end -= clip[0]
    if not ref:
        _log("the reference has no cues (in the clip)"); return 1
    _log(f"[lidbench] ground truth: {where}, {len(ref)} cues, {sum(c.end - c.start for c in ref) / 60:.1f} min of foreign speech")

    job = Job(video, clip, a.audio_track, Path(a.work_dir).expanduser(), config.TMP_DIR, a.context or "", None, a.genre,
              a.window, ["en"], a.source, speakers=getattr(a, "speakers", "off") or "off",
              speaker_threshold=getattr(a, "speaker_threshold", None))
    data = work.load(job.work_file)
    engines = Engines(a.asr_model, a.whisper_model)
    check_speakers(a)
    try:
        _, spans = stage_audio(job, data, need_wav=True)
        audio = load_wav(job.wav)
        turns = []
        if speakers_wanted(job):
            stage_speakers(job, data, audio)
            turns = usable_turns(data)
            if not turns:
                _log("[lidbench] speakers requested but the diarization failed the gate — scoring the plain detector")
        res = stage_lid(job, data, audio, spans, engines, turns=turns)
    finally:
        engines.close()
    dominant = res.dominant or "?"
    speech = _merge_intervals([(s.start, s.end) for s in res.spans])
    truth_raw = _merge_intervals([(c.start, c.end) for c in ref])
    # a forced track is a lower bound on foreign speech, not a complete map: songs, "[speaking German]" cards and
    # lines the subtitler left alone are foreign speech it does not show. Stretches a listener has confirmed as
    # foreign (bench/verified.json: {"<file stem starts with>": [[start, end, "note"], …]}) are taken out of the
    # scoring on both sides — neither a hit nor a false alarm (2026-10-01: two Basterds "false alarms" and the
    # Thai karaoke scene were all the detector being right where the reference was silent)
    verified: list[tuple[float, float]] = []
    vfile = Path(a.out_dir).expanduser() / "verified.json"
    if vfile.is_file():
        import json as _json
        try:
            for key, items in _json.loads(vfile.read_text(encoding="utf-8")).items():
                if video.stem.startswith(key):
                    verified += [(float(s) - (clip[0] if clip else 0.0), float(e) - (clip[0] if clip else 0.0)) for s, e, *_ in items]
        except (ValueError, TypeError) as e:
            _log(f"[lidbench] ⚠ {vfile.name} unreadable ({e}) — ignored")
        verified = _merge_intervals([(s, e) for s, e in verified if e > 0])
        if verified:
            _log(f"[lidbench] {len(verified)} listener-verified stretch(es) ({sum(e - s for s, e in verified):.0f} s) excluded from the scoring")

    def _minus(items: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """The intervals with the verified stretches cut out."""
        out = []
        for s, e in items:
            cur = [(s, e)]
            for vs, ve in verified:
                nxt = []
                for x, y in cur:
                    if ve <= x or vs >= y:
                        nxt.append((x, y))
                    else:
                        if x < vs:
                            nxt.append((x, vs))
                        if ve < y:
                            nxt.append((ve, y))
                cur = nxt
            out += cur
        return [(s, e) for s, e in out if e - s > 0.01]
    # forced subtitles also cover signed dialogue and on-screen text, where nothing is spoken: only the part of the
    # reference that overlaps detected speech can be asked of an acoustic detector
    truth = _minus(_merge_intervals(_intersect(truth_raw, speech)))
    ours_by_lang: dict[str, float] = {}
    foreign = []
    for s in res.spans:
        lang = s.lang or dominant
        ours_by_lang[lang] = ours_by_lang.get(lang, 0.0) + s.dur
        if lang != dominant:
            foreign.append((s.start, s.end))
    foreign = _minus(foreign)
    ours = _merge_intervals(foreign)
    t_raw = sum(e - s for s, e in truth_raw)
    t_truth = sum(e - s for s, e in truth)
    t_ours = sum(e - s for s, e in ours)
    hit = _overlap(truth, ours)
    recall = hit / t_truth if t_truth else 0.0
    precision = hit / t_ours if t_ours else 0.0
    # language changes: blocks of foreign dialogue (cues less than 5 s apart) that contain speech, against ours
    truth_blocks = [b for b in _merge_gap(truth_raw, 5.0) if _overlap([b], speech) > 0.5]
    our_blocks = _merge_gap(foreign, 5.0)
    sw = _switches(truth_blocks, our_blocks)
    langs = ", ".join(f"{config.LANG_NAMES.get(k, k)} {v / 60:.1f}m" for k, v in sorted(ours_by_lang.items(), key=lambda kv: -kv[1]))
    print(f"\nlidbench  {video.name}" + (f"  clip {a.clip}" if clip else ""))
    print(f"  dominant language: {config.LANG_NAMES.get(dominant, dominant)}   detector v{config.LID_VERSION}"
          + (f"   speakers {job.speakers}" if job.speakers != "off" else ""))
    spk_diag: dict = {}
    if speakers_wanted(job):
        # what the speaker evidence did, in one block (2026-10-02): detected vs accepted, fragmentation, how many
        # windows the turns made, priors and purity from the detector's notes, and whether the fallback triggered
        s = data.get("speakers") or {}
        all_turns = s.get("turns") or []
        detected = len({t[2] for t in all_turns})
        accepted = len({t.speaker for t in turns}) if turns else 0
        frag = detected / len(all_turns) if all_turns else 0.0
        note = "; ".join(res.notes)
        import re as _re
        m_pri = _re.search(r"priors for (\d+) voice\(s\) settled (\d+)", note)
        m_pur = _re.search(r"purity ([0-9.]+) over (\d+)", note)
        m_mix = _re.search(r"(\d+) window\(s\) of voices with no single language", note)
        spk_diag = {"detected_speakers": detected, "accepted_speakers": accepted, "turns": len(all_turns),
                    "fragmentation": round(frag, 3), "windows": len(res.windows),
                    "uncertain_windows": res.uncertain_windows, "fallback": not turns,
                    "priors_voices": int(m_pri.group(1)) if m_pri else 0, "priors_settled": int(m_pri.group(2)) if m_pri else 0,
                    "mixed_voice_windows": int(m_mix.group(1)) if m_mix else 0,
                    "purity": float(m_pur.group(1)) if m_pur else None, "embedding": config.SPEAKER_EMBEDDING_FILE[:-5]}
        print(f"  speakers: {detected} detected, {accepted} accepted ({'fallback — plain detector ran' if not turns else 'used'}); "
              f"{len(all_turns)} turns, fragmentation {frag:.2f} clusters/turn; embedding {spk_diag['embedding']}")
        print(f"            {len(res.windows)} windows ({res.uncertain_windows} uncertain); priors for {spk_diag['priors_voices']} voice(s) "
              f"settled {spk_diag['priors_settled']}; {spk_diag['mixed_voice_windows']} window(s) of voices with no single language"
              + (f"; voice/language purity {spk_diag['purity']:.2f}" if spk_diag["purity"] is not None else ""))
    print(f"  foreign speech in the forced track: {t_truth / 60:.1f} min"
          + (f" (of {t_raw / 60:.1f} min of forced subtitles; the rest has no detected speech — signed or on-screen text)" if t_raw - t_truth > 10 else "")
          + f"   labelled foreign by us: {t_ours / 60:.1f} min   both: {hit / 60:.1f} min")
    print(f"  recall {recall:.0%}   precision {precision:.0%}   (by duration)")
    print(f"  switches: {sw['reference_switches']} language changes in the reference ({len(truth_blocks)} foreign blocks); "
          f"detected {sw['switch_recall']:.0%} within 3 s"
          + (f", median latency {sw['median_latency']:.1f} s" if sw["median_latency"] is not None else "")
          + f"; {sw['false']} switches we invented (no reference change within 5 s)")
    print(f"  languages found: {langs}")
    misses = [(s, e, (e - s) - _overlap([(s, e)], ours)) for s, e in truth]
    misses = sorted([m for m in misses if m[2] > 3.0], key=lambda m: -m[2])[:a.show]
    if misses:
        print("  biggest misses (forced subtitles there, we called it the main language):")
        for s, e, gap in misses:
            print(f"    {_hms(s)}–{_hms(e)}  {gap:.0f}s missed of {e - s:.0f}s")
    alarms = [(s, e, (e - s) - _overlap([(s, e)], truth)) for s, e in ours]
    alarms = sorted([x for x in alarms if x[2] > 3.0], key=lambda x: -x[2])[:a.show]
    if alarms:
        print("  biggest false alarms (we called it foreign, no forced subtitles there):")
        for s, e, extra in alarms:
            label = {sp.lang for sp in res.spans if sp.start < e and sp.end > s and sp.lang and sp.lang != dominant}
            print(f"    {_hms(s)}–{_hms(e)}  {extra:.0f}s of {e - s:.0f}s  as {', '.join(sorted(label)) or '?'}")
    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    rec = {"video": str(video), "clip": a.clip, "lid_version": config.LID_VERSION, "speakers": job.speakers,
           "dominant": dominant, "truth_sec": round(t_truth, 1), "truth_raw_sec": round(t_raw, 1),
           "ours_sec": round(t_ours, 1), "hit_sec": round(hit, 1),
           "recall": round(recall, 3), "precision": round(precision, 3),
           "switches": {k: v for k, v in sw.items() if k != "latencies"}, "foreign_blocks": len(truth_blocks),
           "speakers_diag": spk_diag or None,
           "languages_sec": {k: round(v, 1) for k, v in ours_by_lang.items()},
           "misses": [[round(s, 1), round(e, 1), round(g, 1)] for s, e, g in misses],
           "false_alarms": [[round(s, 1), round(e, 1), round(x, 1)] for s, e, x in alarms]}
    tag = video.stem[:80] + (f".{int(clip[0])}-{int(clip[1])}" if clip else "") + (f".spk-{job.speakers}" if job.speakers != "off" else "")
    (out_dir / f"{tag}.lidbench.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_dir / (tag + '.lidbench.json')}")
    return 0


def _hms(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def _pick_bitmap_track(pr, index: int | None, lang: str | None):
    from .subs import PGS_CODECS
    tracks = [t for t in pr.subs if not t.is_text and t.codec in PGS_CODECS]
    if index is not None:
        tracks = [t for t in tracks if t.index == index]
    elif lang:
        from .subs import code_for_tag
        want = {"zh", "yue"} if lang in ("zh", "yue") else {lang}       # a `chi` tag covers both Chinese scripts
        tracks = [t for t in tracks if code_for_tag(t.language) in want and not t.forced] or \
                 [t for t in tracks if code_for_tag(t.language) in want]
    return tracks[0] if tracks else None


def cmd_ocr(a: argparse.Namespace) -> int:
    """`mlsubgen ocr VIDEO --track N` — a bitmap (PGS) subtitle track to .srt through OCR (0.4.8). Writes to the OCR
    cache beside the work files unless --out says otherwise; never beside the video."""
    from . import ocr
    from .probe import describe_tracks, probe
    from .subs import code_for_tag
    video = Path(a.video).expanduser()
    if not video.is_file():
        _log(f"not found: {video}"); return 1
    pr = probe(video)
    t = _pick_bitmap_track(pr, a.track, a.lang)
    if t is None:
        _log("no bitmap subtitle track matched — pick one with --track N:"); _log(describe_tracks(pr)); return 1
    lang = a.lang or code_for_tag(t.language) or "en"
    engine = a.engine if a.engine != "auto" else ocr.engine_for(lang)
    if engine is None:
        _log(f"[ocr] no engine reads {config.LANG_NAMES.get(lang, lang)} bitmap subtitles here (tesseract pack, or a vision model on this profile)"); return 1
    ok, why = ocr.engine_available(engine, lang)
    if not ok:
        _log(f"[ocr] {why}"); return 1
    t0 = time.time()
    _log(f"[ocr] s:{t.index} {t.codec} lang={t.language} → {lang} ({why}) with {engine}")
    out, n, cached = ocr.ocr_track_cached(video, t.index, lang, engine, progress=lambda d, k: _log(f"[ocr] {d}/{k}"))
    if a.out:
        import shutil as _sh
        dest = Path(a.out).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        _sh.copy(out, dest); out = dest
    _log(f"[ocr] {n} cues {'(cached) ' if cached else f'in {time.time() - t0:.0f}s '}→ {out}")
    return 0


def cmd_ocrbench(a: argparse.Namespace) -> int:
    """How good is the OCR? A film with BOTH a bitmap track and a text track in the same language is the free
    ground truth: OCR the bitmap track, pair each OCR'd cue with the text cue it overlaps most, and score the
    characters (chrF) plus how many cues matched. Prints the worst lines so the error kinds are visible."""
    from . import ocr
    from .bench import chrf
    from .probe import describe_tracks, probe
    from .srt import read_srt
    from .subs import code_for_tag, extract_track
    video = Path(a.video).expanduser()
    if not video.is_file():
        _log(f"not found: {video}"); return 1
    pr = probe(video)
    bm = _pick_bitmap_track(pr, a.track, a.lang)
    if bm is None:
        _log("no bitmap track matched"); _log(describe_tracks(pr)); return 1
    lang = a.lang or code_for_tag(bm.language) or "en"
    ref_lang = a.ref_lang or lang                     # chi_tra is read as `yue` here; the text track is still tagged chi
    if a.reference:
        ref = read_srt(Path(a.reference).expanduser()); where = Path(a.reference).name
    else:
        texts = [t for t in pr.subs if t.is_text and code_for_tag(t.language) == ref_lang and not t.forced]
        if a.reference_track is not None:
            texts = [t for t in pr.subs if t.index == a.reference_track]
        if not texts:
            _log(f"no text track in {lang} to compare with — pass --reference FILE.srt or --reference-track N"); _log(describe_tracks(pr)); return 1
        rt = texts[0]
        tmp = config.TMP_DIR / f"{video.stem[:60]}.ocrref.srt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        extract_track(video, rt.index, tmp, rt.codec)
        ref = read_srt(tmp); where = f"s:{rt.index} {rt.codec}"
    if a.engine == "auto":
        a.engine = ocr.engine_for(lang)
        if a.engine is None:
            _log(f"[ocr] no engine reads {config.LANG_NAMES.get(lang, lang)} bitmap subtitles here"); return 1
    ok, why = ocr.engine_available(a.engine, lang)
    if not ok:
        _log(f"[ocr] {why}"); return 1
    t0 = time.time()
    if a.limit:
        bitmaps = ocr.decode_sup(ocr.extract_sup(video, bm.index))
        # with a limit, only the first N images are read (a vision model is slow; a sweep needs many runs)
        if len(bitmaps) > a.limit:
            cut = bitmaps[a.limit].start
            ref = [r for r in ref if r.start < cut]
        cues = ocr.ocr_track_images(bitmaps[:a.limit], lang, a.engine, progress=lambda d, n: _log(f"[ocr] {d}/{n}"), prep=a.prep)
    else:
        out, _n, was_cached = ocr.ocr_track_cached(video, bm.index, lang, a.engine, progress=lambda d, n: _log(f"[ocr] {d}/{n}"))
        cues = [ocr.OcrCue(c.start, c.end, c.text) for c in read_srt(out)]
        if was_cached:
            _log(f"[ocr] cached: {out.name}")
    elapsed = time.time() - t0
    if a.prep:
        where += f" · prep {a.prep}"
    import re as _re
    from .subs import clean_sdh, is_sdh
    # SDH on either side: descriptions and speaker labels are not what the OCR is being measured on (2026-10-02: a
    # Japanese SDH SRT packed two disc cues into one with speaker labels in front, and the per-cue score read 30 %)
    sdh = is_sdh(bm) or (not a.reference and is_sdh(rt))
    norm = lambda s: _re.sub(r"\s+", " ", _re.sub(r"</?i>|\{[^}]*\}", "", clean_sdh(s) if sdh else s)).strip().lower()
    cues = [c for c in cues if norm(c.text)]
    ref = [r for r in ref if norm(r.text)]
    pairs = []
    unmatched = 0
    for c in cues:
        best, best_ov = None, 0.0
        for r in ref:
            ov = min(c.end, r.end) - max(c.start, r.start)
            if ov > best_ov:
                best, best_ov = r, ov
        if best is None or best_ov <= 0:
            unmatched += 1; continue
        pairs.append((c, best))
    score = chrf([norm(c.text) for c, _ in pairs], [norm(r.text) for _, r in pairs]) if pairs else None
    exact = sum(1 for c, r in pairs if norm(c.text) == norm(r.text))
    # the content score: chrF over everything each side says within the same minute of the film, so a cue split
    # on one side and merged on the other is not an error — only the characters are judged
    cjk = _re.compile(r"[\s、。！？!?…「」『』・，,.\-–—－]")
    furigana = _re.compile(r"[\(（][ぁ-ゖー]+[\)）]")        # a transcript's inline readings, 厄介(やっかい): not on the disc
    content = lambda s: cjk.sub("", furigana.sub("", norm(s)) if lang == "ja" else norm(s)) if lang in ("ja", "zh", "yue", "th", "ko") else norm(s)
    span_end = max([c.end for c in cues] + [r.end for r in ref] + [0.0])
    win_h, win_r = [], []
    for w0 in range(0, int(span_end) + 60, 60):
        h = "".join(content(c.text) for c in cues if w0 <= c.start < w0 + 60)
        r = "".join(content(r.text) for r in ref if w0 <= r.start < w0 + 60)
        if h or r:
            win_h.append(h); win_r.append(r)
    content_score = chrf(win_h, win_r) if win_h else None
    print(f"\nocrbench  {video.name}")
    print(f"  bitmap s:{bm.index} ({bm.language}) with {a.engine} vs text {where} · {len(cues)} OCR cues, {len(ref)} reference cues · {elapsed:.0f}s"
          + ("   (SDH: descriptions and speaker labels stripped on both sides)" if sdh else ""))
    print(f"  matched {len(pairs)} ({unmatched} OCR cues overlap no reference cue)   exact {exact} ({100 * exact / max(1, len(pairs)):.0f}%)"
          + (f"   chrF {score:.1f}" if score is not None else "")
          + (f"   content chrF (per minute, cue boundaries ignored) {content_score:.1f}" if content_score is not None else ""))
    worst = sorted(pairs, key=lambda p: -abs(len(norm(p[0].text)) - len(norm(p[1].text))))[:a.show]
    if worst:
        print("  worst length mismatches (OCR | reference):")
        for c, r in worst:
            print(f"    {_hms(c.start)}  {norm(c.text)[:70]!r} | {norm(r.text)[:70]!r}")
    return 0


def cmd_refscore(a: argparse.Namespace) -> int:
    """`mlsubgen refscore VIDEO --out DIR` — the generated files in DIR against the film's own human text tracks
    (0.5.7): chrF++ per minute of film, WER for a same-language pair, coverage. See refscore.py."""
    from . import refscore as rsc
    video, out = Path(a.video).expanduser(), Path(a.out).expanduser()
    langs = [x.strip() for x in a.lang.split(",")] if a.lang else None
    rows = rsc.refscore(video, out, langs, a.bin)
    if not rows:
        _log("nothing to score: no language has both a text track in the file and a generated .srt in --out"); return 1
    rsc.print_table(rows, f"{video.name} — generated files in {out} against the film's own tracks (bins of {a.bin:.0f}s)")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_why(a: argparse.Namespace) -> int:
    """`mlsubgen why VIDEO` — why each subtitle file of this video says what it says (0.5.0.6): where the transcript
    came from, what detected the languages, whether speakers and terms were used, which model translated. Read
    from the work file; nothing is run."""
    from .pipeline import Job, srt_path_for
    video = Path(a.video).expanduser()
    job = Job(video, clip_arg(a.clip) if a.clip else None)
    if not job.work_file.is_file():
        _log(f"no work file for {video.name} ({job.work_file.name}) — it has not been processed here"); return 1
    data = work.load(job.work_file)
    print(f"{video.name}" + (f"  clip {a.clip}" if a.clip else ""))
    print(f"  work file: {job.work_file}")
    src = data.get("source") or {}
    gate = data.get("ocr_gate") or {}
    if src.get("embedded_subtitles") is not None:
        print(f"  transcript: embedded text track s:{src['embedded_subtitles']} ({src.get('codec')}, {config.LANG_NAMES.get(src.get('language'), src.get('language'))}, {src.get('raw_cues')} cues) — no ASR")
    elif src.get("sidecar"):
        print(f"  transcript: {Path(src['sidecar']).name} beside the video ({config.LANG_NAMES.get(src.get('language'), src.get('language'))}) — no ASR")
    elif src.get("ocr_track") is not None:
        g = gate.get(f"s:{src['ocr_track']}") or {}
        print(f"  transcript: bitmap track s:{src['ocr_track']} ({src.get('codec')}, {config.LANG_NAMES.get(src.get('language'), src.get('language'))}) read by OCR"
              f"{' — gate: ' + g.get('why', '?') if g else ''}, {src.get('raw_cues')} cues — no ASR")
    else:
        asr = data.get("asr") or {}
        for key, e in asr.items():
            ms = e.get("merge_stats") or {}
            print(f"  transcript: ASR ({e.get('engines')}, {e.get('mode', 'single')} mode, {e.get('elapsed')}s)"
                  + (f"; merge: agreed {ms.get('agree', 0)}, reconciled by the LLM {ms.get('llm', 0)}"
                     + (f" ({ms['with_evidence']} with the evidence track)" if ms.get("with_evidence") else "")
                     + f", one engine {ms.get('qwen', 0) + ms.get('whisper', 0)}"
                     + (f" ({e.get('merge_model')})" if e.get("merge_model") else "") if ms else (" — merge pending" if e.get("merge") == "pending" else "")))
    lid_d = data.get("lid")
    if lid_d:
        try:
            from .lid import LidResult
            r = LidResult.from_dict(lid_d)
            print(f"  languages: {r.summary()}" + (f"  (forced)" if lid_d.get("forced") else "") + (f"  [speakers {lid_d['speakers']}]" if lid_d.get("speakers") not in (None, "off") else ""))
        except Exception:                                            # noqa: BLE001 — an old work file
            print(f"  languages: {lid_d.get('summary', '?')}")
    spk = data.get("speakers")
    if spk:
        voices = spk.get("voices", len({t[2] for t in spk.get('turns', [])}))
        print(f"  speakers: diarized ({spk.get('mode')}, {voices} voices, {spk.get('turn_count', len(spk.get('turns', [])))} turns, {spk.get('embedding', '?')[:-5] if spk.get('embedding') else '?'})")
    for k, g in gate.items():
        if k != f"s:{src.get('ocr_track')}":
            print(f"  OCR {k} ({config.LANG_NAMES.get(g.get('language'), g.get('language'))}): {'used' if g.get('usable') else 'rejected'} — {g.get('why')}")
    mc = data.get("media_context")
    if mc:
        idn = mc.get("identity") or {}
        wp = mc.get("wikipedia") or {}
        print(f"  lookup: {idn.get('title') or idn.get('series')}" + (f" ({idn['year']})" if idn.get("year") else "")
              + (f" S{idn['season']:02d}E{idn['episode']:02d}" if idn.get("season") and idn.get("episode") else f" ep {idn['episode']}" if idn.get("episode") else "")
              + (f" — used: {wp.get('page')} ({len(wp.get('cast', []))} cast lines, {len(mc.get('results', []))} search results; "
                 f"queries: {', '.join(sorted({q.get('source') for q in mc.get('queries', [])}))})" if mc.get("used") else f" — not used: {mc.get('why')}")
              + (" [cached]" if mc.get("cached") else ""))
    ev = data.get("evidence_track")
    if ev:
        print(f"  evidence: {config.LANG_NAMES.get(ev.get('language'), ev.get('language'))} track s:{ev.get('track')} kept as evidence, not as the source — {ev.get('why')}")
    for tgt, x in (data.get("cross_evidence") or {}).items():
        print(f"  cross-track evidence for {config.LANG_NAMES.get(tgt, tgt)}: "
              + ", ".join(f"{config.LANG_NAMES.get(c, c)} s:{i}" for c, i in (x.get("tracks") or {}).items()) + f" → {x.get('cues')} cues had human lines marking gender/register")
    for ck, cs in (data.get("cues") or {}).items():
        if cs.get("labelled"):
            print(f"  labels: {cs.get('stats', {}).get('labelled_cues', 0)} of {cs.get('cue_count', len(cs.get('cues', [])))} cues carry a voice tag from the audio (text-track source)")
    for key, t in (data.get("terms") or {}).items():
        print(f"  terms: {len(t.get('terms', []))} recurring term(s), rendered for {', '.join(t.get('renderings', {}).keys()) or 'nothing yet'} ({t.get('model')})")
    for key, c in (data.get("characters") or {}).items():
        sheet = c.get("sheet") or []
        print(f"  characters: " + ("; ".join(f"{x['name']} ({x.get('gender', '?')}, {x.get('age', '?')}"
                                              + (", " + ", ".join(f"{r.get('relation')} of {r.get('to')}" for r in x.get("relations", [])[:2]) if x.get("relations") else "")
                                              + (f"; evidence: {x['evidence'][0]!r}" if x.get("evidence") else "") + ")" for x in sheet) or "none")
              + (f" — voices: " + ", ".join(f"{k}={v.get('character') if isinstance(v, dict) else v}"
                                            + (f" ({v['evidence'][:60]!r})" if isinstance(v, dict) and v.get("evidence") else "")
                                            for k, v in sorted((c.get("voices") or {}).items())) if c.get("voices") else "")
              + f" — rules rendered for {', '.join(k for k, v in (c.get('renderings') or {}).items() if v) or 'nothing yet'} ({c.get('model')})")
    for t, o in (data.get("ocr_targets") or {}).items():
        out = srt_path_for(video, t)
        print(f"  {config.LANG_NAMES.get(t, t)}: the real subtitles — bitmap track s:{o.get('track')}{', ' + o['title'] if o.get('title') else ''} read by "
              f"{o.get('engine')} ({o.get('cues')} cues) into {out.name}; " + ("present" if out.is_file() else "not beside the video now"))
    cue_sets = data.get("cues") or {}
    for tkey, t in (data.get("translations") or {}).items():
        target = t.get("target")
        copied = t.get("copied", sum(1 for c in t.get("cues", []) if c.get("flags") and "copied" in c["flags"]))
        out = srt_path_for(video, target)
        if job.clip:
            state = "a clip: the output is a bench page, not a sidecar"
        else:
            state = ("written " + time.strftime("%Y-%m-%d %H:%M", time.localtime(out.stat().st_mtime))) if out.is_file() else "no .srt beside the video"
        print(f"  {config.LANG_NAMES.get(target, target)}: {t.get('cue_count', len(t.get('cues', [])))} cues, translated by {', '.join(t.get('models', []))} in {t.get('elapsed')}s"
              + (f", {copied} copied through" if copied else "") + (" — PARTIAL (interrupted)" if t.get("partial") else "")
              + (" — with speaker labels" if "spk:" in tkey else "") + (" — with the terminology pass" if "|terms" in tkey else "")
              + (" — with the character sheet" if "|chars" in tkey else "") + (" — with cross-track evidence" if tkey.endswith("|xev") else "")
              + (f" — {t['repairs']} hedged line(s) repaired by the checker" if t.get("repairs") else "")
              + (f" — {t['hedged']} still hedged" if t.get("hedged") else "")
              + f"; {state}")
    if not data.get("translations"):
        print("  translations: none recorded" + (f"; cue sets: {len(cue_sets)}" if cue_sets else ""))
    return 0


def cmd_config(a: argparse.Namespace) -> int:
    """`mlsubgen config` shows the settings that matter and where each comes from; `mlsubgen config targets en,th`
    saves the default subtitle languages (the web form's "make these the default" does the same); `--clear` forgets
    the saved value so the environment or the built-in default applies again."""
    if a.key == "targets":
        if a.clear:
            val, src = config.set_default_targets(None)
            print(f"saved default cleared; default subtitle languages are now {val} ({src})")
            return 0
        if a.value:
            codes = parse_targets(a.value)
            val, src = config.set_default_targets(codes)
            print(f"default subtitle languages: {val} (saved in {config.SETTINGS_PATH}; --target on a run still overrides)")
            return 0
    print(f"{'default subtitle languages':<30} {config.DEFAULT_TARGETS:<24} from {config.DEFAULT_TARGETS_SOURCE}")
    print(f"{'profile':<30} {config.PROFILE:<24} " + (f"{config.VRAM_GB:.0f} GB GPU detected" if config.VRAM_GB else "no GPU detected")
          + (f", {config.RAM_GB:.0f} GB RAM" if config.RAM_GB else "")
          + (f", translator layers on the card: {config.OLLAMA_NUM_GPU}" if config.OLLAMA_NUM_GPU else "")
          + (" (MLSUBGEN_PROFILE)" if os.environ.get("MLSUBGEN_PROFILE") else " (auto)"))
    print(f"{'translator server':<30} {config.LLM_URL}")
    print(f"{'media roots':<30} {os.environ.get('MLSUBGEN_MEDIA_ROOTS') or '(not set — every path allowed from the CLI)'}")
    print(f"{'state (MLSUBGEN_HOME)':<30} {config.MLSUBGEN_HOME}")
    print(f"{'settings file':<30} {config.SETTINGS_PATH}{'' if config.SETTINGS_PATH.exists() else ' (none yet)'}")
    print("\nset:  mlsubgen config targets en,th      clear:  mlsubgen config targets --clear")
    return 0


def cmd_pull(a: argparse.Namespace) -> int:
    """Download models ahead of the first run: `mlsubgen pull` = the ASR models + the translators of the default
    routes; names = presets, Ollama tags, ASR labels, 'asr', 'defaults'; --all = every preset too."""
    from . import models
    if getattr(a, "speaker_embedding", None):
        config.set_speaker_embedding(a.speaker_embedding)
    names = list(a.names) or ["defaults"]
    if a.all:
        names = ["asr"] + list(TRANSLATORS)
    rc = 0
    for n in names:
        rc |= models.pull_now(n, a.url)
    return rc


def cmd_selftest(a: argparse.Namespace) -> int:
    import numpy as np
    from .asr import Word
    from .clean import filter_cues
    from .segment import build_cues, normalise_timing
    from .srt import read_srt, typeset, wrap_lines, write_srt
    from .translate import clean_en, parse_numbered
    from .vad import Span, make_chunks, speech_ratio
    import tempfile

    # hardware profiles (2026-10-01): the thresholds, forcing, and what each profile sets; the route assertions
    # below are written for the full profile, so it is made active here whatever card this runs on
    assert config.pick_profile("auto", None) == "full" and config.pick_profile("auto", 24.0) == "full"
    assert config.pick_profile("auto", 12.0) == "12gb" and config.pick_profile("auto", 8.0) == "8gb" and config.pick_profile("auto", 16.0) == "16gb"
    assert config.pick_profile("8gb", 48.0) == "8gb"
    try:
        config.pick_profile("huge", None); raise AssertionError("an unknown profile must be rejected")
    except ValueError:
        pass
    assert all(config.PROFILES[p]["default"] in TRANSLATORS and set(config.PROFILES[p]["routes"].values()) <= set(TRANSLATORS)
               for p in config.PROFILES), "every profile's presets must exist"
    config.apply_profile("8gb-dense")
    assert config.ASR_SEQUENTIAL and config.WHISPER_COMPUTE == "int8_float16" and config.DEFAULT_TRANSLATOR == "gemma4-e4b"
    assert os.environ.get("MLSUBGEN_OCR_VLM") or config.OCR_VLM_MODEL == "gemma4:e4b-it-qat", "the vision model follows the profile"
    # the 26B profiles (2026-10-03): the same mixture-of-experts on 16, 12 and 8 GB cards with a layer split per card
    config.apply_profile("8gb")
    assert config.ASR_SEQUENTIAL and config.DEFAULT_TRANSLATOR == "gemma4-26b" and config.OLLAMA_NUM_GPU == 6
    config.apply_profile("16gb")
    assert not config.ASR_SEQUENTIAL and config.WHISPER_COMPUTE == "float16" and config.DEFAULT_TRANSLATOR == "gemma4-26b" and config.OLLAMA_NUM_GPU == 22
    assert os.environ.get("MLSUBGEN_OCR_VLM") or config.OCR_VLM_MODEL == "gemma4:26b"
    config.apply_profile("12gb")
    assert config.WHISPER_COMPUTE == "int8_float16" and config.DEFAULT_TRANSLATOR == "gemma4-26b" and config.OLLAMA_NUM_GPU == 14
    assert config.pick_profile("auto", 16.0) == "16gb" and config.pick_profile("auto", 12.0) == "12gb" and config.pick_profile("auto", 24.0) == "full" and config.pick_profile("auto", 8.0) == "8gb"
    config.apply_profile("full")
    assert not config.ASR_SEQUENTIAL and config.WHISPER_COMPUTE == "float16" and config.TRANSLATE_ROUTES[("ja", "en")] == "qwen3.8" and config.OLLAMA_NUM_GPU is None
    assert os.environ.get("MLSUBGEN_OCR_VLM") or config.OCR_VLM_MODEL == "gemma4:31b-it-qat"

    # cues from words: sentence end, gap split, overflow
    words = [Word("今日は", 0.0, 0.4), Word("いい", 0.45, 0.6), Word("天気", 0.65, 0.9), Word("ですね。", 0.95, 1.3),
             Word("はい", 2.5, 2.7), Word("、", 2.7, 2.72), Word("そうですね", 2.75, 3.2), Word("。", 3.2, 3.25),
             Word("ご視聴ありがとうございました", 60.0, 61.0)]
    cues = build_cues(words)
    assert len(cues) == 3, cues
    assert cues[0].ja == "今日はいい天気ですね。", cues[0].ja
    assert cues[1].ja == "はい、そうですね。", cues[1].ja
    spans = [Span(0.0, 1.4), Span(2.4, 3.4)]
    kept, stats = filter_cues(cues, spans)
    assert len(kept) == 2 and stats["dropped_no_speech"] == 1, (kept, stats)
    kept = normalise_timing(kept)
    assert kept[0].end - kept[0].start >= config.CUE_MIN_SEC
    # chunking
    chunks = make_chunks([Span(0, 100), Span(101, 250), Span(260, 300), Span(400, 410)], 500.0, max_len=240)
    assert len(chunks) == 4 and chunks[0].end <= 101 and chunks[-1].start >= 399, chunks
    chunks = make_chunks([Span(0, 100), Span(101, 130), Span(131, 160)], 500.0, max_len=240)
    assert len(chunks) == 1, chunks
    assert abs(speech_ratio(spans, 0.0, 2.0) - 0.7) < 0.01
    # parsing
    p = parse_numbered("1\tHello there\n2. Second line\n[3] Third: with colon\n4) Fourth\nnoise\n")
    assert p == {1: "Hello there", 2: "Second line", 3: "Third: with colon", 4: "Fourth"}, p
    assert clean_en('"Quoted." (Note: literal)') == "Quoted."
    # wrapping / typesetting
    assert wrap_lines("short") == ["short"]
    two = wrap_lines("This sentence is deliberately longer than forty-two characters, so it wraps.")
    assert two and len(two) == 2 and all(len(x) <= 42 for x in two), two
    kept[0].en = "Nice weather today, isn't it?"
    kept[1].en = ("Yes, that's right, and this second cue is deliberately far too long to fit on two subtitle "
                  "lines, so the typesetter has to split it into two cues at a natural break point.")
    items = typeset(kept)
    assert len(items) >= 3, [i.text for i in items]
    assert all(len(l) <= 42 for i in items for l in i.text.split("\n")), [i.text for i in items]
    assert all(items[k + 1].start >= items[k].end for k in range(len(items) - 1)), [(i.start, i.end) for i in items]
    assert " ".join(i.text.replace("\n", " ") for i in items[1:]) == kept[1].en, [i.text for i in items]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.srt"
        write_srt(out, items)
        back = read_srt(out)
        assert len(back) == len(items) and back[0].text == items[0].text
        assert not list(Path(d).glob(".mlsubgen-*")), "atomic write left its temp file behind"
    # every module at least imports (a syntax error in a stage only used at run time would otherwise wait for a run)
    import importlib
    for mod in ("audio", "probe", "subs", "vad", "asr", "pipeline", "translate", "jobs", "worker", "web", "models"):
        importlib.import_module(f"mlsubgen.{mod}")
    # translation windows: checkpoint after each, and a resumed list skips the windows it already has
    from .translate import translate_cues
    from .translate import Usage

    class FakeClient:
        def __init__(self):
            self.tr = TRANSLATORS[config.DEFAULT_TRANSLATOR]; self.usage = Usage(); self.calls = 0

        def chat(self, system, user, max_tokens=4096, nudge=None):   # `nudge` arrived with the translator's retry (0.3.0)
            self.calls += 1
            block = user.split("TRANSLATE THESE LINES:\n", 1)[1].split("\n\n")[0]
            return "\n".join(f"{l.split(chr(9))[0]}\tEN {l.split(chr(9))[0]}" for l in block.splitlines())

    from .segment import Cue as _Cue
    cs = [_Cue(i, i * 2.0, i * 2.0 + 1.5, f"行{i}") for i in range(7)]
    fc = FakeClient(); saves = []
    translate_cues(cs, fc, window_size=3, progress=False, checkpoint=lambda c: saves.append(sum(1 for x in c if x.en)))
    assert fc.calls == 3 and saves == [3, 6, 7] and all(c.en == f"EN {c.idx + 1}" for c in cs), (fc.calls, saves)
    for c in cs[3:]:
        c.en = ""                                   # as if interrupted after the first window
    fc = FakeClient()
    translate_cues(cs, fc, window_size=3, progress=False)
    assert fc.calls == 2 and all(c.en for c in cs), fc.calls
    # the queue: add → next → running → interrupted → queued again; cancel; retry
    with tempfile.TemporaryDirectory() as d:
        conn = jobs.connect(Path(d) / "q.db")
        jid = jobs.add(conn, ["/x/videos"], ["--assume-ja", "--keep-work"])
        j = jobs.next_queued(conn)
        assert j and j.id == jid and j.argv()[:6] == ["run", "/x/videos", "--assume-ja", "--keep-work", "--now", "--since"], j
        assert abs(float(j.argv()[6]) - time.time()) < 120, "the --since marker is the job's creation time"
        jobs.update(conn, jid, status=jobs.RUNNING, attempts=1, pid=4242)
        assert jobs.next_queued(conn) is None
        assert jobs.requeue_running(conn, "restart") == [jid] and jobs.get(conn, jid).status == jobs.QUEUED
        # pause: queued → paused at once, invisible to next_queued, untouched by a worker restart; resume → queued;
        # running → pausing; a worker restart mid-pause leaves it paused
        assert jobs.pause(conn, jid) == "paused" and jobs.next_queued(conn) is None
        assert jobs.requeue_running(conn, "restart") == [] and jobs.get(conn, jid).status == jobs.PAUSED
        assert jobs.retry(conn, jid).startswith("job is paused")
        assert jobs.resume(conn, jid).startswith("queued") and jobs.next_queued(conn).id == jid
        jobs.update(conn, jid, status=jobs.RUNNING)
        assert jobs.pause(conn, jid).startswith("pausing") and jobs.get(conn, jid).status == jobs.PAUSING
        assert jobs.requeue_running(conn, "restart") == [] and jobs.get(conn, jid).status == jobs.PAUSED
        assert jobs.cancel(conn, jid) == "cancelled" and jobs.retry(conn, jid) == "queued"
        jobs.update(conn, jid, not_before=time.time() + 3600)
        assert jobs.next_queued(conn) is None, "a delayed retry must not be picked early"
        jobs.update(conn, jid, not_before=0)
        assert jobs.cancel(conn, jid) == "cancelled" and jobs.retry(conn, jid) == "queued"
        jobs.update(conn, jid, status=jobs.RUNNING)
        assert jobs.cancel(conn, jid).startswith("cancelling")
        assert "id" in jobs.fmt_table(jobs.list_jobs(conn))
        assert jobs.purge(conn) == 0, "a running job must never be purged"
        jobs.update(conn, jid, status=jobs.CANCELLED)
        assert jobs.purge(conn, ids=[jid]) == 1 and jobs.get(conn, jid) is None
        conn.close()
    # per-chunk ASR checkpoints: saved under asr_partial (never under asr), reusable by a resumed pass, gone when the stage completes
    from .pipeline import Job as _Job, _partial, _partial_done
    with tempfile.TemporaryDirectory() as d:
        vpath = Path(d) / "v.mkv"; vpath.touch()
        pj = _Job(vpath, work_dir=Path(d))
        pdata: dict = {}
        part, ck = _partial(pj, pdata, "k")
        ck("qwen")(0, Span(0, 10, "ja"), [Word("はい", 0.0, 0.5)], {"i": 0})
        saved = work.load(pj.work_file)
        assert saved["asr_partial"]["k"]["qwen"]["0"]["words"][0]["text"] == "はい" and "asr" not in saved
        assert part["qwen"]["0"]["start"] == 0
        _partial_done(pdata, "k")
        assert "asr_partial" not in pdata
    # --overwrite with --since: an .srt written after the marker is this job's own work and is not redone
    from .pipeline import missing_targets as _mt
    with tempfile.TemporaryDirectory() as d:
        v = Path(d) / "ep1.mkv"; v.touch()
        assert _mt(v, ["en", "th"]) == ["en", "th"]
        (Path(d) / "ep1.en.srt").write_text("1\n", encoding="utf-8")
        assert _mt(v, ["en", "th"]) == ["th"] and _mt(v, ["en", "th"], overwrite=True) == ["en", "th"]
        assert _mt(v, ["en", "th"], overwrite=True, since=time.time() - 60) == ["th"], "rewritten after the marker: done"
        assert _mt(v, ["en", "th"], overwrite=True, since=time.time() + 60) == ["en", "th"], "older than the marker: redo"
    # language layer: decision gate, smoothing, language-aware chunks and cues, routes, copy-through, Thai wrapping
    from . import lid
    from .translate import copy_through, route, untranslated
    from .vad import make_chunks as _mk
    assert lid.script_of("今日はいい天気") == "ja" and lid.script_of("สวัสดี") == "th" and lid.script_of("你好世界") == "han"
    assert not lid.is_confident(*lid.decide("zh", 0.55, "zh", "嗯嗯啊…")[1:]), "agreement without words must not be confident"
    assert lid.decide("zh", 0.6, "ja", "そうですね、本当に")[0] == "ja"
    assert lid.smooth(["ja", "ja", None, "en", "ja", "en", "en", "ja"], [True, True, False, True, True, True, True, True]) \
        == ["ja", "ja", "ja", "ja", "ja", "en", "en", "ja"]
    # LID v3 (2026-10-01): function words tell Latin-script languages apart; a strongly evidenced single window of a
    # third language survives the smoothing; the switch point moves to the exact span
    assert lid.text_language("Che cosa vuoi? Non lo so, perché anche questo è difficile")[0] == "it"
    assert lid.text_language("The cat sat on the mat with you")[0] == "en" and lid.text_language("hello world") == (None, 0)
    assert lid.script_vote("Ich weiß nicht, was das ist und wir haben nichts") == ("de", 0.5)
    assert lid.script_vote("Bonjour madame")[0] == "en" and lid.script_vote("Bonjour madame")[1] < 0.2, "undecided Latin text: a faint lean only"
    assert lid.decide("it", 0.6, "it", "Che cosa vuoi? perché anche questo")[0] == "it"
    assert lid.smooth(["en", "it", "en", "en"], [True] * 4, margins=[0.9, 0.9, 0.9, 0.9]) == ["en", "it", "en", "en"], "strong single window kept"
    assert lid.smooth(["en", "it", "en", "en"], [True] * 4, margins=[0.9, 0.3, 0.9, 0.9]) == ["en", "en", "en", "en"], "weak single window absorbed"
    from .audio import SR as _SR
    _aud = np.zeros(12 * _SR, dtype=np.float32)
    for k in range(6):
        _aud[k * 2 * _SR] = float(k + 1)                      # each 2-second span starts with its own number

    class _FakeWhisper:
        def language_probs(self, piece):
            k = int(round(float(piece[0]))) - 1
            return {"en": 0.9, "fr": 0.1} if k < 4 else {"en": 0.1, "fr": 0.9}
    _sp = [Span(k * 2.0, k * 2.0 + 2.0, "en" if k < 3 else "fr") for k in range(6)]
    _win = [lid.Window(0.0, 6.0, 6.0, [0, 1, 2]), lid.Window(6.0, 12.0, 6.0, [3, 4, 5])]
    assert lid.refine_boundaries(_aud, _sp, _win, ["en", "fr"], _FakeWhisper()) == 1
    assert [s.lang for s in _sp] == ["en", "en", "en", "en", "fr", "fr"], [s.lang for s in _sp]
    # speaker-aware detection (0.4.4): spans cut at turns, a window is one voice, priors from a voice's history,
    # a prior adds evidence but never flips a confident window, an uncertain window follows its own voice
    from .speakers import Turn as _Turn
    pieces = lid.split_at_turns([Span(0.0, 10.0)], [_Turn(0.0, 4.0, "S1"), _Turn(4.0, 10.0, "S2")])
    assert [(p.start, p.end, p.speaker) for p in pieces] == [(0.0, 4.0, "S1"), (4.0, 10.0, "S2")], pieces
    assert lid.split_at_turns([Span(0.0, 10.0)], [_Turn(0.0, 9.9, "S1"), _Turn(9.9, 10.0, "S2")])[0].end == 10.0, "a cut too near the edge is not made"
    sp3 = [Span(0.0, 3.0, speaker="S1"), Span(3.5, 6.5, speaker="S1"), Span(7.0, 10.0, speaker="S2")]
    assert [w.speaker for w in lid.build_windows(sp3, by_speaker=True)] == ["S1", "S2"], "a change of voice closes the window"
    assert len(lid.build_windows(sp3)) == 1, "without speakers the three spans are one window"
    # 0.4.5: a voice change does not close a window of a second or two; the window's speaker is its main voice
    tiny_spans = [Span(0.0, 1.0, speaker="S1"), Span(1.2, 6.2, speaker="S2"), Span(6.5, 9.5, speaker="S2")]
    tiny = lid.build_windows(tiny_spans, by_speaker=True, voice_min_speech=3.0)
    assert len(tiny) == 1 and tiny[0].speaker == "S2", [(w.speaker, w.speech) for w in tiny]
    assert len(lid.build_windows(tiny_spans, by_speaker=True, voice_min_speech=0.0)) == 2, "with no minimum, the change closes at once"
    assert len(lid.build_windows(sp3, by_speaker=True, voice_min_speech=3.0)) == 2
    assert lid.smooth(["en", "fr", "en", "en"], [True] * 4, margins=[0.9] * 4, speakers=["S1", "S2", "S1", "S1"], speech=[10, 2, 10, 10]) \
        == ["en", "en", "en", "en"], "two seconds of a strong other language: absorbed"
    assert lid.smooth(["en", "fr", "en", "en"], [True] * 4, margins=[0.9] * 4, speakers=["S1", "S2", "S1", "S1"], speech=[10, 5, 10, 10]) \
        == ["en", "fr", "en", "en"], "five strong seconds: kept"
    assert lid.smooth(["en", "fr", "fr", "en"], [True] * 4, margins=[0.3] * 4, speakers=["S1", "S2", "S2", "S1"], speech=[10, 5, 5, 10]) \
        == ["en", "fr", "fr", "en"], "ten weak seconds: kept on duration"
    wins = [lid.Window(0, 1, 20.0, [], lang="ja", confident=True, speaker="S1"),
            lid.Window(0, 1, 6.0, [], lang="ja", confident=True, speaker="S2"), lid.Window(0, 1, 4.0, [], lang="en", confident=True, speaker="S2"),
            lid.Window(0, 1, 7.0, [], lang="en", confident=True, speaker="S3"), lid.Window(0, 1, 2.0, [], lang="ja", confident=True, speaker="S3"),
            lid.Window(0, 1, 30.0, [], lang="de", confident=False, speaker="S4")]
    pri = lid.speaker_priors(wins)
    assert pri == {"S1": ("ja", config.LID_PRIOR_STRONG), "S3": ("en", config.LID_PRIOR_WEAK)}, pri   # S2 bilingual: none; S4 never confident
    assert lid.decide("ja", 0.55, "ja", "ん", prior=("ja", 0.5))[0] == "ja" and lid.is_confident(*lid.decide("ja", 0.55, "ja", "ん", prior=("ja", 0.5))[1:])
    assert not lid.is_confident(*lid.decide("ja", 0.55, "ja", "ん")[1:]), "the same window without the prior stays uncertain"
    assert lid.decide("en", 0.9, "en", "the cat and the dog with you", prior=("ja", 0.5))[0] == "en", "a prior never flips strong evidence"
    assert lid.smooth(["ja", "en", None, "en"], [True, True, False, True], margins=[0.9] * 4, speakers=["S1", "S2", "S1", "S2"]) \
        == ["ja", "en", "ja", "en"], "an uncertain window follows its own voice"
    assert lid.smooth(["ja", "en", None, "en"], [True, True, False, True], margins=[0.9] * 4) == ["ja", "en", "en", "en"], "without speakers: the nearest neighbour"
    from .vad import cover_chunks as _cover
    _a2 = np.concatenate([np.full(2 * _SR, 0.01, dtype=np.float32), np.full(18 * _SR, 0.5, dtype=np.float32)])
    ch2 = _cover(_a2, [Span(2.0, 10.0, "en"), Span(10.0, 20.0, "de")], 20.0, max_len=30.0)
    assert [c.lang for c in ch2] == ["en", "de"] and abs(ch2[0].end - 10.0) < 0.3, "a language change cuts a chunk"
    ch = _mk([Span(0, 10, "ja"), Span(12, 20, "ja"), Span(21, 22, "ja"), Span(30, 60, "ja"), Span(61, 70, "en"), Span(200, 201, "en"),
              Span(240, 260, "en")], 300.0, max_len=240, break_silence=6, pad=0)
    assert [(c.start, c.end, c.lang) for c in ch] == [(0, 22, "ja"), (30, 60, "ja"), (61, 70, "en"), (200, 201, "en"), (240, 260, "en")], ch
    mixed = build_cues([Word("今日は", 0, 0.4, "ja"), Word("いい", 0.45, 0.6, "ja"), Word("OK", 0.7, 0.9, "en"), Word("go.", 0.95, 1.1, "en")])
    assert [(c.ja, c.lang) for c in mixed] == [("今日はいい", "ja"), ("OK go.", "en")], mixed
    assert route("ja", "en") == "qwen3.8" and route("ja", "th") == "gemma4" and route("th", "en", "qwen3.8") == "qwen3.8"
    assert copy_through("um, so we, uh, went", "en") == "so we, went" and untranslated("今日は", "ja", "en") and not untranslated("Hi", "ja", "en")
    from .srt import wrap_lines as _wl
    thai = "ฉันเดินทางไปเที่ยวที่จังหวัดเชียงใหม่ในช่วงฤดูหนาวเพื่อสัมผัสอากาศเย็นสบายทุกวันจริงๆ"
    assert _wl(thai, 50) and all(len(x) <= 50 for x in _wl(thai, 50))
    cs2 = [_Cue(0, 0, 1, "はい", lang="ja"), _Cue(1, 2, 3, "OK let's go", lang="en"), _Cue(2, 4, 5, "行きます", lang="ja")]
    fc2 = FakeClient()
    translate_cues(cs2, fc2, window_size=20, progress=False, target="en")
    assert [c.en for c in cs2] == ["EN 1", "OK let's go", "EN 3"] and fc2.calls == 2, ([c.en for c in cs2], fc2.calls)
    assert parse_targets("en, th") == ["en", "th"] and parse_targets(None) == config.DEFAULT_TARGETS.split(",")
    assert parse_targets("en,xx", warn=False) == ["en", "xx"], "an unknown code is passed through, not rejected"
    # saved default languages (2026-10-01): settings.json beats the environment, clearing hands back to it; English
    # is not required anywhere — a th,de default runs the same pipeline
    _orig = (config.SETTINGS_PATH, config.DEFAULT_TARGETS, config.DEFAULT_TARGETS_SOURCE)
    with tempfile.TemporaryDirectory() as d:
        config.SETTINGS_PATH = Path(d) / "settings.json"
        val, src = config.set_default_targets(["th", "DE ", "th"])
        assert (val, src) == ("th,de", "settings") and parse_targets(None) == ["th", "de"], (val, src)
        assert config.load_settings()["targets"] == ["th", "de"] and config.SETTINGS_PATH.exists()
        val, src = config.set_default_targets(None)
        assert src in ("environment", "built-in") and not config.load_settings().get("targets"), (val, src)
    config.SETTINGS_PATH, config.DEFAULT_TARGETS, config.DEFAULT_TARGETS_SOURCE = _orig
    # more subtitle languages (2026-10-01): every code has a native name; CJK targets keep their punctuation and get
    # their own reading speed; cluster-safe cuts work for any script, not just Thai
    assert all(c in config.NATIVE_NAMES for c in config.LANG_NAMES), [c for c in config.LANG_NAMES if c not in config.NATIVE_NAMES]
    assert clean_en("今日は。", "ja") == "今日は。" and clean_en("今日は。", "en") == "今日は."
    from .translate import system_prompt
    assert "简体" in system_prompt("ja", "zh", "a film") and "no trailing" in system_prompt("ja", "en", "a film")
    from .srt import _safe_cut
    lao = "ຂ້ອຍເວົ້າພາສາລາວ"                    # ວົ້ : a vowel and a tone mark stacked on ວ — never cut between them
    k = lao.index("\u0ebb")                     # the vowel mark
    assert _safe_cut(lao, k) < k and _safe_cut(lao, k + 1) < k, "a cut inside a Lao cluster must move back to the base"
    assert _safe_cut("abc", 2) == 2
    from .srt import typeset as _ts
    ja_cue = [_Cue(0, 0.0, 0.5, "x", lang="ja")]; ja_cue[0].en = "今日はとてもいい天気ですね"      # 13 chars in 0.5 s
    assert _ts(ja_cue, lang="ja")[0].end >= 13 / config.SRT_MAX_CPS_BY_LANG["ja"] - 0.01, "CJK reading speed stretches the cue"
    # embedded ASS tracks: dialogue by style, karaoke/romaji and comments out
    from .subs import ass_to_cues
    ass = ("[V4+ Styles]\nFormat: Name, Fontname, Alignment\nStyle: Default,Arial,2\nStyle: OP-Romaji,Arial,8\n[Events]\n"
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           "Dialogue: 0,0:00:02.00,0:00:08.00,OP-Romaji,,0,0,0,,{\\an8}machi no doubutsuen\n"
           "Dialogue: 0,0:00:02.00,0:00:08.00,Default,,0,0,0,,The town zoo{TLC}\\Nis busy\n"
           "Comment: 0,0:00:02.00,0:00:08.00,Default,,0,0,0,,private\n")
    ac = ass_to_cues(ass)
    assert [(c.start, c.end, c.text) for c in ac] == [(2.0, 8.0, "The town zoo\nis busy")], [(c.start, c.end, c.text) for c in ac]
    # embedded tracks in every language (2026-10-01): tags and titles of the 0.3.3 languages are recognised, forced
    # and signs tracks never count, and the spoken language's track is the transcript before any other
    from .probe import SubTrack as _ST
    from .subs import LANG_TAGS as _LT, code_for_tag, pick as _pick, plan_embedded
    assert all(c in _LT for c in config.LANG_NAMES) and code_for_tag("tgl") == "tl" and code_for_tag("khm") == "km"
    # targets withheld (2026-10-03): still known languages, not offered as targets; the CLI refuses them plainly
    assert len(config.TARGET_LANGS) == 44 and len(config.LANG_NAMES) == 45 and "el" not in config.TARGET_LANGS and "el" in config.LANG_NAMES
    assert "hu" in config.TARGET_LANGS and "my" in config.TARGET_LANGS, "returned 2026-10-04"
    # per-profile offer (2026-10-04): a profile withholds what is measured below shippable on ITS translators;
    # unmeasured profiles inherit the full profile's set, and apply_profile() rebuilds the offer
    assert set(config.PROFILE_WITHHELD) == set(config.PROFILES), "every profile has a withheld set"
    assert config.targets_for_profile("8gb").keys() == config.targets_for_profile("full").keys(), "inherited until measured"
    _prev = config.PROFILE
    from .translate import route as _route
    config.apply_profile("8gb")
    assert "el" not in config.TARGET_LANGS and len(config.TARGET_LANGS) == 44
    assert "translategemma-12b" in config.TRANSLATORS and config.TRANSLATORS["translategemma-4b"].prompt_style == "translategemma"
    # the small profiles' routes (2026-10-05): the 12B where the 26B invents words, the 4B on 8 GB, never for Thai
    assert _route("en", "hu") == "translategemma-4b" and _route("en", "sl") == "gemma4-26b" and _route("en", "th") == "gemma4-26b"
    config.apply_profile("12gb")
    assert _route("en", "hu") == "translategemma-12b" and _route("en", "sl") == "translategemma-12b" and _route("en", "th") == "gemma4-26b"
    assert _route("ja", "en") == "gemma4-26b"
    config.apply_profile("full")
    from .translate import foreign_script, route as _route
    assert _route("en", "hu") == "translategemma" and _route("en", "lv") == "translategemma" and _route("en", "fr") == "gemma4" and _route("ja", "en") == "qwen3.8"
    # the foreign-script guard: letters outside the target's script and Latin are a leak; Latin names are not
    assert foreign_script("Ну thôi! Нет.", "ru") == 0, "Vietnamese diacritics are Latin letters: the guard cannot tell — see untranslated()"
    assert foreign_script("நீ ஒரு சிறந்த எழுத்தாளன். என்意思是...", "ta") == 3
    assert foreign_script("ខ្ញុំមានអារម្មណ៍ដូចនៅទីនោះផ្ទាល់เลย", "km") == 3
    assert foreign_script("Tā ir tava дневna", "lv") == 4 and foreign_script("Es izlasīju vienu lapu.", "lv") == 0
    assert foreign_script("ไหน ကြည့်ရအောင်", "my") == 3 and foreign_script("Peace! Peace!", "my") == 0
    assert foreign_script("蠟筆小新 says OK", "zh") == 0 and foreign_script("Shin-chan はカブトムシが好き", "zh") > 0
    assert foreign_script("ทำให้ นะ", "th") == 0 and foreign_script("日本語の字幕", "ja") == 0 and foreign_script("한국어 자막 OK", "ko") == 0
    assert config.LANG_NAMES["zh"].endswith("(Simplified)") and config.LANG_NAMES["yue"].endswith("(Traditional)")
    try:
        parse_targets("en,el", warn=False)
        raise AssertionError("a withheld target must be refused")
    except SystemExit as e:
        assert "Greek" in str(e)
    assert parse_targets("en,th", warn=False) == ["en", "th"]
    assert parse_targets("el", warn=False, allow_withheld=True) == ["el"], "the bench may measure a withheld language"
    assert code_for_tag("JPN") == "ja" and code_for_tag("und") is None and code_for_tag("xx") is None
    tracks = [_ST(0, 2, "khm", "", "subrip", True, False, False), _ST(1, 3, "und", "Bahasa Melayu", "ass", True, False, False),
              _ST(2, 4, "eng", "Signs & Songs", "ass", True, False, False), _ST(3, 5, "eng", "", "subrip", True, True, False),
              _ST(4, 6, "spa", "", "subrip", True, False, False), _ST(5, 7, "jpn", "", "hdmv_pgs_subtitle", False, False, False)]
    assert _pick(tracks, "km").index == 0 and _pick(tracks, "ms").index == 1, "tag and title of the new languages"
    assert _pick(tracks, "en") is None, "a signs track and a forced track are not English subtitles"
    assert _pick(tracks, "ja") is None, "a bitmap track is unusable"
    sat, src = plan_embedded(tracks, ["en", "th"], "auto", spoken="es")
    assert sat == [] and src and src[1] == "es", "the spoken language's track is the transcript"
    sat, src = plan_embedded(tracks, ["en", "th"], "auto", spoken="km")
    assert src and src[1] == "km", "a Khmer audio tag picks the Khmer track"
    sat, src = plan_embedded(tracks, ["en", "th"], "auto", spoken=None)
    assert src and src[1] == "es", "untagged audio: the first usable track in the fixed order (Spanish precedes Khmer)"
    sat, src = plan_embedded(tracks, ["km", "th"], "auto", spoken="es")
    assert sat == ["km"] and src and src[1] == "es", "an embedded target counts as done; the rest come from the spoken track"
    # bitmap tracks through OCR (0.4.8): picked like text tracks; a bitmap track becomes a source only when the OCR
    # engine can read its language here, and never with --ocr off
    from .subs import bitmap_targets as _btg, ocr_ready as _ocr_ready, pick_bitmap as _pb, plan_sources as _ps
    assert _pb(tracks, "ja").index == 5 and _pb(tracks, "en") is None
    # SDH cleaning (0.4.9): descriptions go, speaker labels go, dialogue stays
    from .subs import clean_sdh as _csdh, is_sdh as _issdh
    assert _csdh("（ドアが閉まる）") == "" and _csdh("[door closes]\nWhere are you?") == "Where are you?"
    assert _csdh("♪ ♪") == "" and _csdh("♪ [upbeat music] ♪") == "" and _csdh("MAN: Over here.") == "Over here."
    assert _csdh("（男）こんにちは") == "こんにちは" and _csdh("ジョン：行こう") == "行こう" and _csdh("18:30に会おう") == "18:30に会おう"
    assert _csdh("Wait... (sighs) I'm fine.") == "Wait... (sighs) I'm fine.", "a description inside a spoken line stays"
    assert _issdh(_ST(0, 2, "jpn", "Japanese SDH [JPNBD]", "hdmv_pgs_subtitle", False, False, False)) and not _issdh(tracks[0])
    with tempfile.TemporaryDirectory() as d:
        vid = Path(d) / "film.mkv"; vid.touch()
        bm_tracks = [_ST(0, 2, "eng", "", "hdmv_pgs_subtitle", False, False, False), _ST(1, 3, "eng", "", "hdmv_pgs_subtitle", False, True, False)]
        assert _pb([_ST(0, 2, "eng", "", "dvd_subtitle", False, False, False)], "en") is None, "VobSub is not PGS: not picked"
        assert _ps(vid, bm_tracks, ["th"], "auto", spoken="en", ocr=False) == ([], None), "--ocr off: a bitmap track is not a source"
        assert _btg(bm_tracks, ["en"], [], ocr=False) == {}
        if _ocr_ready("en"):
            sat, src = _ps(vid, bm_tracks, ["th"], "auto", spoken="en", ocr=True)
            assert src and src[0].index == 0 and src[1] == "en", "the non-forced English bitmap track is the source"
            assert list(_btg(bm_tracks, ["en", "th"], [], ocr=True)) == ["en"], "a bitmap track in a target language is OCR'd into that target"
    # models: Ollama's pull stream parses, names resolve to the right kind of download, status needs no server
    from . import models as _models
    assert _models.parse_pull_line('{"status":"pulling abc","total":100,"completed":40}') == (40, 100, "pulling abc", False)
    assert _models.parse_pull_line('{"status":"success"}')[3] is True
    assert _models.resolve(config.DEFAULT_TRANSLATOR) == [(config.DEFAULT_TRANSLATOR, "ollama", TRANSLATORS[config.DEFAULT_TRANSLATOR].model)]
    assert _models.resolve("some/other:tag") == [("some/other:tag", "ollama", "some/other:tag")]
    assert {k for k, _, _ in _models.resolve("asr")} == set(_models.ASR_MODELS)
    assert len(_models.resolve("defaults")) == len(_models.ASR_MODELS) + len(set(config.TRANSLATE_ROUTES.values()))
    st = _models.status("http://127.0.0.1:1")                    # nothing listens there: reported, not raised
    assert st["ollama"]["reachable"] is False and len(st["asr"]) == 3 and all("ready" in t for t in st["translators"])
    # speakers (0.4.0): words take the overlapping turn, a speaker change closes a cue, the cue carries its speaker,
    # the translator prompt tags the lines and gets the rule, and an echoed tag is stripped from the output
    from . import speakers as _spk
    from .translate import build_prompt as _bp, spoken as _spoken
    turns = [_spk.Turn(0.0, 1.0, "S1"), _spk.Turn(1.2, 3.0, "S2"), _spk.Turn(3.1, 5.0, "S1")]
    sw = [Word("おはよう", 0.1, 0.5, "ja"), Word("ございます", 0.5, 0.9, "ja"), Word("はい", 1.3, 1.6, "ja"),
          Word("そうですね", 1.7, 2.5, "ja"), Word("ええ", 3.2, 3.5, "ja"), Word("行きましょう", 3.6, 4.5, "ja"), Word("ん", 7.0, 7.2, "ja")]
    st_ = {}
    assert _spk.label_words(sw, turns, stats=st_) == 6 and [w.speaker for w in sw] == ["S1", "S1", "S2", "S2", "S1", "S1", ""]
    assert st_["ambiguous_words"] == 0
    assert _spk.dominant(sw[:4]) == "S2" and _spk.dominant([sw[-1]]) == ""     # by duration: S2 1.1 s vs S1 0.8 s
    # a word two voices cover about equally stays unlabelled — unknown beats confidently wrong
    amb = [Word("ね", 0.8, 1.4, "ja")]                                        # 0.2 s under S1 (to 1.0), 0.2 s under S2 (from 1.2)
    st_ = {}
    assert _spk.label_words(amb, turns, stats=st_) == 0 and amb[0].speaker == "" and st_["ambiguous_words"] == 1
    cst = {}
    scues = build_cues(sw[:6], stats=cst)
    assert [c.speaker for c in scues] == ["S1", "S2", "S1"] and scues[1].ja == "はいそうですね", [(c.ja, c.speaker) for c in scues]
    assert cst["speaker_splits"] == 1, cst                                   # S1→S2 closed a cue; S2→S1 was already a pause
    assert "speaker(s)" in _spk.summary(turns) and _spk.summary(turns).startswith("2 speaker(s), 3 turns")
    assert _spoken(scues[0]) == "[S1] おはようございます" and _spoken(_Cue(0, 0, 1, "x")) == "x"
    sys_p, user_p = _bp(TRANSLATORS[config.DEFAULT_TRANSLATOR], scues[:2], [], scues[2:], {}, "a film", "en")
    assert "[S1] おはようございます" in user_p and "Never output the tags" in sys_p
    sys_q, _ = _bp(TRANSLATORS[config.DEFAULT_TRANSLATOR], [_Cue(0, 0, 1, "x")], [], [], {}, "a film", "en")
    assert "speaker tag" not in sys_q, "no tags, no rule"
    assert clean_en("[S2] Good morning.") == "Good morning." and clean_en("[S2]Good morning.") == "Good morning."
    assert _spk.available()[0] or "sherpa-onnx" in _spk.available()[1] or "missing" in _spk.available()[1]
    if _spk.available()[0]:                     # the binding's constructors, whenever the package and models are here
        assert _spk.build_config(0, 0.5) is not None and _spk.build_config(4, None) is not None
    assert _models.resolve("speakers") and all(k == "url" for _, k, _ in _models.resolve("speakers"))
    assert "speakers" in _models.status("http://127.0.0.1:1") and len(_models.status("http://127.0.0.1:1")["speakers"]) == 2
    # lidbench interval arithmetic (2026-10-01): merging, overlap, intersection with speech, conversation blocks,
    # and the switch metrics — a reference change we reproduce within 3 s counts, one we invent is false
    assert _merge_intervals([(5, 8), (0, 3), (2, 4)]) == [(0, 4), (5, 8)]
    assert abs(_overlap([(0, 4), (5, 8)], [(2, 6)]) - 3.0) < 1e-9 and _overlap([(0, 1)], [(2, 3)]) == 0.0
    assert _intersect([(0, 10)], [(2, 4), (8, 12)]) == [(2, 4), (8, 10)] and _intersect([(0, 1)], [(5, 6)]) == []
    assert _merge_gap([(0, 2), (3, 5), (20, 22)], 5.0) == [(0, 5), (20, 22)]
    sw_ = _switches([(10, 20), (50, 60)], [(11, 21), (80, 90)])
    assert sw_["reference_switches"] == 4 and sw_["detected"] == 2 and sw_["false"] == 2 and sw_["median_latency"] == 1.0, sw_
    assert _hms(3725) == "1:02:05"
    # untagged subtitle tracks (0.4.7): the language read from the words; nothing decided on too little
    from .probe import language_of_text
    en_srt = "1\n00:00:01,000 --> 00:00:03,000\nWhat are you doing with that?\n\n2\n00:00:04,000 --> 00:00:06,000\n" \
             "I just think there would be a problem because of what they said.\n\n3\n00:00:07,000 --> 00:00:09,000\n" \
             "This is your house and their house, which were from the start the same when you have it.\n\n" \
             "4\n00:00:10,000 --> 00:00:12,000\nAbout that: the people you know and the ones that just left.\n"
    assert language_of_text(en_srt) == "en", language_of_text(en_srt)
    ja_srt = "1\n00:00:01,000 --> 00:00:03,000\n" + "そうですね、本当にそう思います。今日はいい天気ですね。" * 10
    assert language_of_text(ja_srt) == "ja"
    assert language_of_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n") is None, "too little text decides nothing"
    assert language_of_text("1\n00:00:01,000 --> 00:00:02,000\n" + "lorem ipsum dolor sit amet consectetur " * 10) is None, "Latin text with no function words decides nothing"
    # PGS decoding (0.4.8): a hand-built stream — palette, a 4x2 object (run-length coded), a composition at 1.0 s,
    # a clear at 2.5 s — decodes to one bitmap with the right pixels and interval; a replacing composition ends it too
    from . import ocr as _ocr

    def _seg(pts: float, kind: int, payload: bytes) -> bytes:
        return b"PG" + int(pts * 90000).to_bytes(4, "big") + b"\0\0\0\0" + bytes([kind]) + len(payload).to_bytes(2, "big") + payload
    pds = bytes([1, 0]) + bytes([1, 235, 128, 128, 255])                       # palette 1: index 1 = white, opaque
    rle = bytes([0, 0x81, 1, 0, 0x02, 0, 0, 0, 0x84, 1, 0, 0])                 # row 1: 1 white, 2 zeros, (pad) ; row 2: 4 white
    ods = (7).to_bytes(2, "big") + bytes([0, 0xC0]) + (len(rle) + 4).to_bytes(3, "big") + (4).to_bytes(2, "big") + (2).to_bytes(2, "big") + rle
    pcs_on = (1920).to_bytes(2, "big") + (1080).to_bytes(2, "big") + bytes([0x10, 0, 1, 0x80, 0, 1, 1]) + (7).to_bytes(2, "big") + bytes([0, 0]) + (100).to_bytes(2, "big") + (900).to_bytes(2, "big")
    pcs_off = (1920).to_bytes(2, "big") + (1080).to_bytes(2, "big") + bytes([0x10, 0, 2, 0x00, 0, 1, 0])
    stream = _seg(1.0, 0x16, pcs_on) + _seg(1.0, 0x14, pds) + _seg(1.0, 0x15, ods) + _seg(1.0, 0x80, b"") + _seg(2.5, 0x16, pcs_off) + _seg(2.5, 0x80, b"")
    bms = _ocr.decode_sup(stream)
    assert len(bms) == 1 and (bms[0].start, bms[0].end, bms[0].width, bms[0].height) == (1.0, 2.5, 4, 2), bms
    px = [bms[0].rgba[i * 4 + 3] for i in range(8)]                            # alpha per pixel
    assert px == [255, 0, 0, 0, 255, 255, 255, 255], px
    stream2 = _seg(1.0, 0x16, pcs_on) + _seg(1.0, 0x14, pds) + _seg(1.0, 0x15, ods) + _seg(1.0, 0x80, b"") + \
        _seg(3.0, 0x16, pcs_on) + _seg(3.0, 0x14, pds) + _seg(3.0, 0x15, ods) + _seg(3.0, 0x80, b"") + _seg(4.0, 0x16, pcs_off) + _seg(4.0, 0x80, b"")
    assert [(b.start, b.end) for b in _ocr.decode_sup(stream2)] == [(1.0, 3.0), (3.0, 4.0)], "a replacing composition ends the previous one"
    assert _ocr.TESSERACT_LANGS["ja"] == "jpn" and _ocr.TESSERACT_LANGS["th"] == "tha"
    # the measured OCR habits: tight dialogue dashes, a capital I read as a pipe, l'm for I'm (English only)
    assert _ocr.clean_ocr("-Thanks. -You're welcome.", "en") == "- Thanks. - You're welcome."
    assert _ocr.clean_ocr("Girl: | hate you, | hate you!", "en") == "Girl: I hate you, I hate you!"
    assert _ocr.clean_ocr("l'm sure l'll go.", "en") == "I'm sure I'll go." and _ocr.clean_ocr("l'homme", "fr") == "l'homme"
    assert _ocr.clean_ocr("a well-known man\n-Yes.", "en") == "a well-known man\n- Yes.", "hyphenated words keep their hyphen"
    assert _ocr.clean_ocr("Tom | Jerry", "en") == "Tom | Jerry", "a pipe between words stays"
    assert _ocr.clean_ocr("_Emily! _Emily!", "en") == "- Emily! - Emily!", "an underscore read for a dialogue dash"
    assert _ocr.clean_ocr("...l knew...", "en") == "...I knew..." and _ocr.clean_ocr("l knew", "fr") == "l knew"
    assert _ocr.clean_ocr("the_name", "en") == "the_name", "an underscore inside a word stays"
    assert _ocr.clean_ocr("iIn Nazi-occupied France", "en") == "In Nazi-occupied France", "an italic I read twice"
    # the OCR gate (0.5.0.4): subtitles pass; symbol salad, the wrong script, a repeated line and prose do not
    _OC = _ocr.OcrCue
    good_en = [_OC(i, i + 1, t) for i, t in enumerate(["What are you doing?", "I just think there's a problem.", "- Thanks. - You're welcome.",
                                                        "Where were you last night?", "It's fine, really.", "Come on, let's go.",
                                                        "He said no.", "Why not?", "Because I said so.", "Okay then.", "See you tomorrow.", "Bye."])]
    assert _ocr.assess(good_en, "en")[0], _ocr.assess(good_en, "en")
    good_th = [_OC(i, i + 1, t) for i, t in enumerate(["หยุดนะ หยุด", "แกรรี่ นี่มันอะไรกัน", "หมายศาลสำหรับยึดทรัพย์สิน", "ไม่ได้", "ไปกันเถอะ",
                                                        "ฉันคิดถึงไอ้บ้านั่น", "เขาพูดถูก", "ทำไมล่ะ", "ก็เพราะฉันบอกไง", "โอเค", "เจอกันพรุ่งนี้", "บาย"])]
    assert _ocr.assess(good_th, "th")[0], _ocr.assess(good_th, "th")
    salad = [_OC(i, i + 1, t) for i, t in enumerate(["๓% = = %7% ฆ ฉัน", "๓ม = = ขช= 1 «| ร ๐", "%% == ๐ ๐ =“ ฉัน", "= = %7% ฆ", "๓% = =", "«| ร ๐ ๐ =",
                                                      "%7% ฆ ==", "= = ๓ม", "๐ =“ ==", "«| %7%", "== ๓% =", "%% =="])]
    ok, m, why = _ocr.assess(salad, "th")
    assert not ok and "symbols" in why, (m, why)
    wrong = [_OC(i, i + 1, t) for i, t in enumerate(["这是什么", "我不知道", "走吧", "等一下", "为什么", "因为我说了", "好的", "明天见", "再见", "不行", "快点", "来吧"])]
    ok, m, why = _ocr.assess(wrong, "th")
    assert not ok and "script" in why, (m, why)
    rep = [_OC(i, i + 1, "Loading…" if i % 2 else f"line {i} here now") for i in range(12)]
    ok, m, why = _ocr.assess(rep, "en")
    assert not ok and "repeated" in why, (m, why)
    prose = good_en[:11] + [_OC(20, 21, "The text says: hello")]
    ok, m, why = _ocr.assess(prose, "en")
    assert not ok and "describe" in why, (m, why)
    # in-script salad (0.5.0.5): the real Thai case — noise as an extra line of digits and symbols beside real text
    thai_salad = [_OC(i, i + 1, t) for i, t in enumerate(["หยุคนะ หยุด", "4ส4๐ '\nแกรี นีมันอะไรกัน", "๐ ขม% เจ, ๕\nหมายศาลสำหรับยึดทรัพย์สิน",
                                                           "๓๕ ๒\nไปกันเถอะ", "ฉันคิดถึง", "7๐ '๐\nเขาพูดถูก", "ทำไมล่ะ", "๐๐ %\nก็เพราะฉัน",
                                                           "โอเค", "๕'๐\nเจอกัน", "บาย", "๐ ๐\nไม่ได้"])]
    ok, m, why = _ocr.assess(thai_salad, "th")
    assert not ok and "digits" in why, (m, why)
    assert _ocr.assess(good_th + [_OC(50, 51, "1944")], "th")[0], "one year card is not salad"
    assert _ocr.clean_ocr("ท\u0e4d\u0e32ให้", "th") == "ทำให้", "Thai sara am as one character"
    assert _ocr.clean_ocr("什么cdots那是我表哥", "zh") == "什么…那是我表哥" and _ocr.clean_ocr(r"wait\ldots", "en") == "wait…"
    assert "th" in _ocr.VLM_SCRIPTS and "en" not in _ocr.VLM_SCRIPTS and "el" not in _ocr.VLM_SCRIPTS
    _prof = config.PROFILE
    config.apply_profile("8gb-dense")
    assert _ocr.engine_for("th") is None, "the E4B does not read Thai well enough: no engine on the dense 8 GB profile"
    config.apply_profile(_prof)
    # a truncated run-length fragment decodes what it can instead of raising
    assert len(_ocr._rle_decode(bytes([0, 0x84]), 4, 1)) == 4
    assert _ocr.VLM_PROMPT.format(language="Thai", extra="").startswith("This image is one subtitle")
    assert "furigana" in _ocr.VLM_PROMPT.format(language="Japanese", extra=_ocr.VLM_EXTRA["ja"])
    try:
        import PIL  # noqa: F401
        from PIL import Image as _Img
        for m in _ocr.PREP_MODES:
            assert _ocr.to_png(bms[0], mode=m)[:8] == b"\x89PNG\r\n\x1a\n", m
        assert _ocr.vlm_png(bms[0])[:8] == b"\x89PNG\r\n\x1a\n"
        # text-line splitting for stacked scripts: two lines of ink with a wide gap → two images; a thin row of
        # "marks" just above a line stays with that line
        im = _Img.new("L", (200, 120), 255)
        px = im.load()
        for x in range(20, 180):
            for y in list(range(20, 24)) + list(range(30, 50)) + list(range(80, 100)):   # marks 20–23, line 30–49, line 80–99
                px[x, y] = 0
        parts = _ocr.split_text_lines(im)
        assert len(parts) == 2 and parts[0].height > 30 and parts[1].height > 20, [(p.width, p.height) for p in parts]
        assert "th" in _ocr.STACKED_SCRIPTS and "en" not in _ocr.STACKED_SCRIPTS
    except ImportError:
        pass
    try:
        import PIL  # noqa: F401
        png = _ocr.to_png(bms[0])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    except ImportError:
        pass                                                                   # pillow is a dependency of the OCR path only
    # terminology (0.5.1): candidates by script, recurrence, fragments folded into longer terms, user glossary wins
    from . import terms as _tm
    ja_lines = ["しんちゃん、おはよう", "しんちゃんは学校へ", "ドアを開けて", "カスカベに行く", "ミサエとヒロシ", "ミサエ！", "ヒロシは会社", "テレビを見る"]
    hc = _tm.heuristic_candidates(ja_lines, "ja")
    assert hc.get("ミサエ") == 2 and hc.get("ヒロシ") == 2 and hc.get("カスカベ") == 1 and "ドア" not in hc and "テレビ" not in hc, hc
    sel = _tm.select_terms(ja_lines, "ja", hc, ["しんちゃん", "ミサエとヒロシ", "ドア"])
    names = [t for t, _ in sel]
    assert "しんちゃん" in names and "ミサエ" in names and "ヒロシ" in names and "カスカベ" not in names and "ドア" not in names, sel
    # a fragment of a longer word in the text is not a term, even when the whole word is a common one not kept itself
    frag = _tm.select_terms(["ハチミツを食べる", "ハチミツが好き"], "ja", {}, ["ハチミ"])
    assert [t for t, _ in frag] == [], frag
    en_lines = ["Shin-chan went to Kasukabe.", "Misae and Hiroshi laughed.", "Then Misae left. Kasukabe is quiet.", "Hiroshi works."]
    he = _tm.heuristic_candidates(en_lines, "en")
    assert he.get("Misae") == 1 and he.get("Kasukabe") == 1 and he.get("Hiroshi") == 1 and "Then" not in he and "Shin" not in he, he
    # media identification from file names (0.5.8): no network in the selftest, only the parse and the opt-in default
    from . import lookup as _lk
    _i = _lk.identify(Path("/x/Definitely.Maybe.2008.1080p.BluRay.x264-EbP.mkv"))
    assert _i["title"] == "Definitely Maybe" and _i["year"] == 2008 and _i["kind"] == "film" and _i["confidence"] >= 0.6, _i
    _i = _lk.identify(Path("/x/Shin Chan/Season 2/[BuriBuri] Crayon Shin-chan - 0064 [720p][51E88FB6].mkv"))
    assert _i["series"] == "Crayon Shin-chan" and _i["episode"] == 64 and _i.get("season_folder") == 2, _i
    _i = _lk.identify(Path("/x/Blue.Planet.II.S01E02.The.Deep.1080p.AMZN.WEB-DL.DDP5.1.H.264-NTb.mkv"))
    assert _i["series"] == "Blue Planet II" and _i["season"] == 1 and _i["episode"] == 2, _i
    _i = _lk.identify(Path("/x/Kindergarten Cop 1990 1080p UHD BluRay DDP7.1 Atmos DV HDR x265-HiDt.mkv"))
    assert _i["title"] == "Kindergarten Cop" and _i["year"] == 1990, _i
    _i = _lk.identify(Path("/x/Red.Notice.2021.2160p.NF.WEB-DL.DDP.5.1.Atmos.DoVi.HDR.HEVC-SiC.mkv"))
    assert _i["title"] == "Red Notice" and _i["year"] == 2021, _i
    assert _lk.identify(Path("/x/home video.mkv"))["confidence"] < 0.6
    assert not _lk.enabled(None) and not _lk.enabled("off") and _lk.enabled("auto"), "the lookup is opt-in"
    assert _lk.facts_text({"used": False}) == ""
    # cross-track evidence (0.5.15): the picker takes the first track per need, never the target or the source, at
    # most two; the prompt carries the lines beside the window and the rule
    from .pipeline import pick_evidence_tracks
    class _T:
        def __init__(self, i, lang, title="", text=True): self.index, self.language, self.title, self.is_text, self.codec = i, lang, title, text, "subrip"
    _subs = [_T(0, "eng"), _T(1, "ara"), _T(2, "bul"), _T(3, "tha"), _T(4, "fre"), _T(5, "heb", "SDH"), _T(6, "heb"), _T(7, "kor"), _T(8, "jpn", text=False), _T(9, "ger")]
    _pick = [t.index for t in pick_evidence_tracks(_subs, "th", "en")]
    assert _pick == [6, 4], _pick                       # ranked (0.5.16): Hebrew for gender over Arabic and Bulgarian earlier in the file, French for formality
    assert [t.index for t in pick_evidence_tracks(_subs, "he", "en")] == [3], "a Hebrew target: gender from the best-ranked marker left (Thai over Bulgarian and Arabic), never its own track"
    assert not any(t.language == "tha" for t in pick_evidence_tracks(_subs, "th", "en"))
    assert pick_evidence_tracks(_subs, "en", "ja") == [] and pick_evidence_tracks([], "th", "en") == []
    from .translate import build_prompt as _bp, Translator as _Tr
    from .segment import Cue as _SegCue
    _w = [_SegCue(0, 0.0, 1.0, "Are you coming?", "", lang="en"), _SegCue(1, 1.0, 2.0, "Yes.", "", lang="en")]
    _s, _u = _bp(_Tr("gemma4", "gemma4:31b-it-qat"), _w, [], [], {}, "a film", "th", "", {0: [("Hebrew", "את באה?"), ("French", "Tu viens ?")]})
    assert "EVIDENCE" in _u and "[Hebrew] את באה?" in _u and "[French] Tu viens ?" in _u and "EVIDENCE lines are human translations" in _s
    _s2, _u2 = _bp(_Tr("gemma4", "gemma4:31b-it-qat"), _w, [], [], {}, "a film", "th", "", {})
    assert "EVIDENCE" not in _u2 and "EVIDENCE" not in _s2
    # an ad-hoc --model tag is a chat model as far as the terms and the sheet know (the burn-in's KeyError, 0.5.13)
    from .pipeline import translation_only
    assert translation_only("translategemma") and translation_only("translategemma-4b")
    assert not translation_only("gemma4") and not translation_only("nonexistent:tag") and not translation_only(None)
    # a finished file's work file condenses to its provenance and reopens as caches only (0.5.13)
    import tempfile
    with tempfile.TemporaryDirectory() as _td:
        _p = Path(_td) / "x.json"
        work.save(_p, {"video": "/x/v.mkv", "duration": 600.0, "source": {"embedded_subtitles": 0, "language": "en"},
                       "speakers": {"mode": "auto", "turns": [[0, 1, "S1"], [1, 2, "S2"], [2, 3, "S1"]]},
                       "asr": {"k": {"engines": "dual", "words": [{"text": "a"}] * 50, "dual": [1, 2, 3], "merge_stats": {"agree": 3}}},
                       "cues": {"ck": {"cues": [{"idx": i} for i in range(40)], "stats": {"labelled_cues": 38}, "labelled": 1}},
                       "terms": {"ck": {"terms": [["Will", 11]], "renderings": {"th": {"Will": "วิล"}}}},
                       "characters": {"ck": {"sheet": [{"name": "Will"}], "voices": {}, "renderings": {"th": "rules"}}},
                       "translations": {"ck|th": {"target": "th", "cues": [{"idx": 0, "en": "x", "flags": ["copied"]}, {"idx": 1, "en": "y"}],
                                                  "models": ["gemma4:31b-it-qat"], "elapsed": 9.0, "repairs": 1}},
                       "evidence_track": {"track": 3, "language": "it", "lines": [[0, 1, "ciao"]] * 20}})
        work.condense(_p)
        _c = work.load(_p)
        assert _c.get("condensed") and "words" not in _c["asr"]["k"] and _c["asr"]["k"]["merge_stats"] == {"agree": 3}
        assert _c["cues"]["ck"]["cue_count"] == 40 and _c["cues"]["ck"]["stats"]["labelled_cues"] == 38 and "cues" not in _c["cues"]["ck"]
        assert _c["translations"]["ck|th"]["cue_count"] == 2 and _c["translations"]["ck|th"]["copied"] == 1 and "cues" not in _c["translations"]["ck|th"]
        assert _c["speakers"]["voices"] == 2 and _c["speakers"]["turn_count"] == 3 and "turns" not in _c["speakers"]
        assert _c["evidence_track"]["line_count"] == 20 and "lines" not in _c["evidence_track"] and _p.stat().st_size < 3000
        _r = work.reopen(_c)
        assert "asr" not in _r and "cues" not in _r and "translations" not in _r and _r["terms"] and _r["characters"] and "condensed" not in _r
        assert work.reopen({"asr": {"k": {"words": []}}}) == {"asr": {"k": {"words": []}}}, "an uncondensed file is untouched"
    # --speakers has three meanings (0.5.10): labels (default) touches only a text track's cues; auto/N the audio path
    from .pipeline import Job as _Job, labels_wanted as _lw, speakers_wanted as _sw
    _j = _Job(Path("/x/v.mkv"))
    assert _j.speakers == "labels" and not _sw(_j), "the default leaves the audio path alone"
    _d = {"cues": {"embedded|s:0|subrip|en": {"cues": [{"lang": "en", "ja": "x", "start": 0, "end": 1, "idx": i} for i in range(50)]}}}
    _j.targets = ["th"]
    assert _lw(_j, "embedded|s:0|subrip|en", _d), "labels wanted for a text track with a foreign target"
    _j.targets = ["en"]
    assert not _lw(_j, "embedded|s:0|subrip|en", _d), "no labels when every target is the transcript's language"
    _j.targets = ["th"]; _j.speakers = "off"
    assert not _lw(_j, "embedded|s:0|subrip|en", _d) and not _sw(_j), "off is off everywhere"
    _j.speakers = "auto"
    assert _lw(_j, "embedded|s:0|subrip|en", _d) and _sw(_j)
    _j.speakers = "labels"; _j.register = "off"
    assert not _lw(_j, "embedded|s:0|subrip|en", _d), "no sheet, no labels"
    # the command line accepts every meaning (the 2026-10-05 burn-in found the default itself rejected)
    for v in ("off", "labels", "auto", "3"):
        _ns = argparse.Namespace(speakers=v); check_speakers(_ns); assert _ns.speakers == v
    try:
        check_speakers(argparse.Namespace(speakers="many")); raise AssertionError("an unknown value must be refused")
    except SystemExit:
        pass
    # the reference scorer (0.5.7): per-minute bins, chrF++, WER, coverage — on two tiny cue sets
    from .refscore import score_pair, wer as _wer
    from .srt import SrtCue as _SrtCue
    _ref = [_SrtCue(0.0, 2.0, "Come here."), _SrtCue(3.0, 5.0, "I'll help you."), _SrtCue(70.0, 72.0, "Too late.")]
    _hyp = [_SrtCue(0.5, 2.2, "Come here, I'll help you."), _SrtCue(69.0, 73.0, "Too late!")]
    _r = score_pair(_ref, _hyp, "en", 60.0)
    assert _r["bins_ref"] == 2 and _r["bins_hyp"] == 2 and _r["bins_both"] == 2 and _r["coverage"] == 1.0 and _r["chrf"] and _r["chrf"] > 60 and _r["wer"] == 0.0, _r
    assert _wer(["a", "b", "c"], ["a", "b", "c"]) == 0.0 and _wer(["a", "b", "c"], ["a", "c"]) > 0 and _wer([], []) == 0.0
    # the evidence track for the reconciler (0.5.6): the foreign lines for a chunk's seconds, in the prompt only
    # where the LLM has to decide, never in the output
    from .merge import evidence_for, merge_prompt
    _ev = [(10.0, 12.5, "Vieni qui."), (12.6, 15.0, "Ti aiuto io."), (40.0, 42.0, "Troppo tardi.")]
    assert evidence_for(_ev, 11.0, 14.0) == "Vieni qui.\nTi aiuto io." and evidence_for(_ev, 20.0, 30.0) == "" and evidence_for([], 0, 99) == ""
    _s, _u = merge_prompt("come here", "come hear", "English", "a film", "", evidence_for(_ev, 11.0, 14.0), "Italian")
    assert "EVIDENCE (Italian subtitles" in _u and "Vieni qui." in _u and "never output it" in _s
    _s2, _u2 = merge_prompt("a", "b", "English", "a film", "")
    assert "EVIDENCE" not in _u2 and "EVIDENCE" not in _s2
    # the hedge detector (0.5.5): a form hedged with a slash is caught in any script; dates, fractions, URLs, AC/DC
    # and plain alternatives of unrelated long words are not
    from .translate import hedged
    for s in ("Myslím, že by sis to měl/a nechat.", "ฉันยังไม่ได้อ่านค่ะ/ครับ.", "Si odličen/a pisec/pisateljica.",
              "Είσαι εξαιρετικός/ή συγγραφέας.", "Imel/a sem občutek.", "sám/sama"):
        assert hedged(s), s
    for s in ("See https://example.org/path now", "3/4 of them", "12/06/2024", "AC/DC live", "yes/no", "Ich bin müde.",
              "the input/output buffer", "Mr. and/or Mrs."):
        assert not hedged(s), s
    # the character sheet's parsing (0.5.4): one entry per person with aliases, evidence kept, voices mapped only to
    # known names and well-formed tags; a bare JSON array (the old shape) still parses
    from . import characters as _ch
    class _Fake:
        def __init__(self, out): self.out = out
        def chat(self, system, user, **kw): return self.out
    sheet, voices = _ch.build_sheet(_Fake('{"characters": [{"name": "Will", "aliases": ["Dad", "the father"], "gender": "m", "age": "adult", '
                                           '"role": "father", "evidence": ["Daddy!"], "relations": [{"to": "Maya", "relation": "father", "status": "higher", '
                                           '"evidence": "Oh, please! Daddy."}]}, {"name": "Maya", "gender": "f", "age": "child"}], '
                                           '"voices": {"S1": {"character": "Will", "evidence": "[S2] Oh, please! Daddy. [S1] Oh, come on!"}, '
                                           '"S2": "Maya", "S3": {"character": "Nobody", "evidence": "x"}, "S4": {"character": "Maya", "evidence": ""}, '
                                           '"bad": {"character": "Will", "evidence": "y"}}}'), ["x"], "en", "a film", tagged=True)
    assert [c["name"] for c in sheet] == ["Will", "Maya"] and sheet[0]["aliases"] == ["Dad", "the father"] and sheet[0]["evidence"] == ["Daddy!"]
    assert sheet[0]["relations"][0]["evidence"].startswith("Oh, please") and sheet[1]["age"] == "child"
    # evidence-gated (0.5.11): S1 has a quote and a known name → kept; S2 is a bare name, S3 an unknown name, S4 has no
    # evidence, "bad" is not a tag → all dropped
    assert list(voices) == ["S1"] and voices["S1"]["character"] == "Will" and voices["S1"]["evidence"].startswith("[S2] Oh"), voices
    sheet2, voices2 = _ch.build_sheet(_Fake('[{"name": "Shin-chan", "gender": "m", "age": "child"}]'), ["x"], "ja", "an anime")
    assert sheet2[0]["name"] == "Shin-chan" and voices2 == {} and sheet2[0]["aliases"] == []
    rules = _ch.render_sheet(_Fake("Will: refers to himself as 僕…"), sheet, "en", "ja", "a film", voices=voices)
    assert rules.startswith("[S1] is Will\nWill:"), rules[:60]
    # caption remnants in an ordinary track (2026-10-04): uppercase labels and tags go, lowercase dialogue stays
    from .subs import clean_captions
    assert clean_captions("MAYA: (LAUGHING) You wanted to be President?") == "You wanted to be President?"
    assert clean_captions("[DOOR CLOSES]\nWill, wait.") == "Will, wait." and clean_captions("♪ ♪") == ""
    assert clean_captions("Note: he said (quietly) that it was 18:30.") == "Note: he said (quietly) that it was 18:30."
    assert clean_captions("WILL: I'm fine.\n- (SIGHS) Really?") == "I'm fine.\n- Really?"
    he2 = _tm.heuristic_candidates(["- Yeah? What about Emily?", "Hey, Yeah, Em - You know what", "Oh God, April. - What?"], "en")
    assert "Yeah" not in he2 and "What" not in he2 and "You" not in he2 and "Oh" not in he2 and he2.get("Emily") == 1 and he2.get("April") == 1, he2
    # (a name at a sentence start is not counted by the heuristic — the LLM pass is what finds those)
    assert _tm.build_glossary({"ミサエ": "มิซาเอะ", "ヒロシ": "ฮิโรชิ"}, {"ミサエ": "มิซาเอ"}) == {"ミサエ": "มิซาเอ", "ヒロシ": "ฮิโรชิ"}
    assert _tm._chunks(["a" * 100] * 50, 1000) and sum(len(p) for p in _tm._chunks(["a" * 100] * 50, 1000)) >= 5000
    # metrics for node_exporter's textfile collector (0.5.0.11): a valid exposition, written atomically, off when unset
    from . import metrics as _metrics
    _orig_metrics_path = _metrics.PATH
    with tempfile.TemporaryDirectory() as d:
        _metrics.PATH = str(Path(d) / "mlsubgen.prom")
        _metrics.publish(busy=True, queued=3, job_id=42)
        body = (Path(d) / "mlsubgen.prom").read_text(encoding="utf-8")
        assert "mlsubgen_worker_busy 1\n" in body and "mlsubgen_jobs_queued 3\n" in body and "mlsubgen_worker_job_id 42\n" in body and "mlsubgen_worker_up 1\n" in body
        assert not (Path(d) / "mlsubgen.prom.tmp").exists(), "the write is a rename, not a partial file"
        _metrics.publish(busy=False, queued=0, up=False)
        body = (Path(d) / "mlsubgen.prom").read_text(encoding="utf-8")
        assert "mlsubgen_worker_busy 0\n" in body and "mlsubgen_worker_up 0\n" in body
        for line in body.splitlines():
            assert not line or line.startswith("#") or len(line.split(" ")) == 2, f"not exposition format: {line!r}"
        _metrics.PATH = ""
        _metrics.publish(busy=True, queued=1)          # unset: a no-op, nothing raised
        _metrics.PATH = _orig_metrics_path
    # the PGS regression corpus (0.5.0.6): real streams trimmed to a few display sets, kept OUT of the repository
    # (film content) under MLSUBGEN_HOME/tests/pgs with a manifest; checked when present, skipped when not
    import hashlib
    import json as _json
    corpus = config.MLSUBGEN_HOME / "tests" / "pgs"
    manifest = corpus / "manifest.json"
    if manifest.is_file():
        for name, exp in _json.loads(manifest.read_text(encoding="utf-8")).items():
            bms2 = _ocr.decode_sup((corpus / name).read_bytes())
            assert len(bms2) == exp["bitmaps"], f"{name}: {len(bms2)} bitmaps, expected {exp['bitmaps']}"
            f0 = bms2[0]
            assert [round(f0.start, 2), round(f0.end, 2), f0.width, f0.height] == exp["first"], (name, f0.start, f0.end, f0.width, f0.height)
            assert hashlib.sha1(f0.rgba).hexdigest()[:12] == exp["sha1"], f"{name}: first bitmap's pixels changed"
        print(f"PGS corpus: {len(_json.loads(manifest.read_text(encoding='utf-8')))} stream(s) decode as recorded")
    # worker readiness helpers
    assert worker.paths_ready(["/definitely/not/here"], mounts=[]) is not None
    assert worker.paths_ready([str(Path(tempfile.gettempdir()))], mounts=[], roots=[]) is None
    assert worker.paths_ready(["/mnt/nas/x"], mounts=["/mnt/nas"], roots=[]) == "/mnt/nas is not mounted"
    with tempfile.TemporaryDirectory() as d:
        assert worker.paths_ready([d], mounts=[], roots=[d]) == f"{d} is empty — share not mounted yet?"
        Path(d, "x").touch()
        assert worker.paths_ready([d], mounts=[], roots=[d]) is None
    worker.other_mlsubgen_running(set())              # must not raise; may or may not find one
    print("selftest OK")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlsubgen", description="subtitles for videos, entirely on your own machine — "
                                 "`mlsubgen help` for the guide, `mlsubgen help COMMAND` for a command's options",
                                 usage="mlsubgen [run] [PATH ...] [options]  |  mlsubgen COMMAND [arguments]")
    ap.add_argument("--version", action="version", version=f"mlsubgen {__version__}")
    # prog= stops argparse prefixing every command's usage line with the top-level usage string
    sub = ap.add_subparsers(dest="cmd", required=True, prog="mlsubgen")

    r = sub.add_parser("run", help="subtitle every video in the given files/folders: one <video>.<lang>.srt per target language")
    r.add_argument("paths", nargs="*", help="files or folders (default: the current folder)")
    r.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    r.add_argument("-t", "--translator", default=None, help=f"preset: {', '.join(TRANSLATORS)} (default {config.DEFAULT_TRANSLATOR})")
    r.add_argument("--model", default=None, help="explicit model tag (overrides the preset's)")
    r.add_argument("--prompt-style", default=None, choices=["generic", "translategemma"])
    r.add_argument("--target", default=None, help=f"subtitle languages, comma list, one .srt each (default {config.DEFAULT_TARGETS})")
    r.add_argument("--keep-source", "--keep-ja", action="store_true", dest="keep_source",
                   help="also write <video>.<lang>.srt with the transcript in the spoken language")
    r.add_argument("--keep-wav", action="store_true", help="keep the extracted 16 kHz wav in ~/mlsubgen/tmp")
    r.add_argument("--keep-work", action="store_true",
                   help="keep the work file (ASR + translation cache) after the .srt is written, e.g. to re-translate "
                        "with another model without redoing the ASR")
    r.add_argument("--batch", type=int, default=10,
                   help="files per round of ASR-then-translate (default 10); 0 = ASR every file first, then translate every file")
    r.add_argument("--subs", default="auto", choices=["auto", "ja", "ignore"],
                   help="embedded subtitle tracks: auto = a target language that is embedded as a text track is left alone "
                        "(no .srt written) and any other full text track (not forced/signs; the spoken language's first) "
                        "is the transcript; ja = ignore embedded target tracks, make our own translation; ignore = always ASR")
    r.add_argument("--overwrite", action="store_true", help="rewrite an existing .srt")
    r.add_argument("--since", type=float, default=None, metavar="EPOCH",
                   help="with --overwrite: an .srt written after this time counts as done (the worker passes the job's "
                        "creation time, so a paused or rebooted overwrite job carries on where it stopped)")
    r.add_argument("--force-translate", action="store_true", help="ignore the cached translation")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--now", action="store_true", help="run here even when the mlsubgen-worker service is up")
    r.add_argument("--queue", action="store_true", help="queue for the service instead of running here (same as `mlsubgen add`)")
    add_common(r)
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bench", help="compare translators on one video (or a clip of it)")
    b.add_argument("video")
    b.add_argument("--translators", default=",".join(TRANSLATORS), help="comma list of presets")
    b.add_argument("--clip", default=None, help="START-END, e.g. 0:10:00-0:20:00")
    b.add_argument("--reference", default=None, help="official English .srt for chrF++ scoring (kept local)")
    b.add_argument("--out-dir", default=str(config.BENCH_DIR))
    b.add_argument("--target", default=None, help="the target language to bench (default: the first default target)")
    b.add_argument("--force-translate", action="store_true")
    add_common(b)
    b.set_defaults(fn=cmd_bench)

    lb = sub.add_parser("lidbench", help="score the language detector on a multilingual film against its forced subtitle track")
    lb.add_argument("video")
    lb.add_argument("--clip", default=None, help="START-END, e.g. 0:10:00-0:40:00 (default: the whole film)")
    lb.add_argument("--forced-track", type=int, default=None, metavar="N",
                    help="the forced subtitle track s:N (default: the first text track flagged or titled forced)")
    lb.add_argument("--reference", default=None, help="a forced-subtitles .srt instead of an embedded track")
    lb.add_argument("--show", type=int, default=8, help="how many misses and false alarms to list (default 8)")
    lb.add_argument("--out-dir", default=str(config.BENCH_DIR))
    add_common(lb)
    lb.set_defaults(fn=cmd_lidbench)

    oc = sub.add_parser("ocr", help="a bitmap (PGS) subtitle track to .srt through OCR (0.4.8); writes to the OCR cache")
    oc.add_argument("video")
    oc.add_argument("--track", type=int, default=None, metavar="N", help="the bitmap track s:N (see `mlsubgen tracks`)")
    oc.add_argument("--lang", default=None, help="the OCR language (default: the track's tag)")
    oc.add_argument("--engine", default="auto", choices=["auto", "tesseract", "vlm"],
                    help="auto = the engine measured best for the script (tesseract for alphabets, the profile's vision model "
                         "for CJK and stacked scripts); tesseract = the binary + its language pack; vlm = a vision model through Ollama")
    oc.add_argument("--out", default=None, help="write the .srt here instead of the cache")
    oc.set_defaults(fn=cmd_ocr)
    ob = sub.add_parser("ocrbench", help="score the OCR of a bitmap track against a text track of the same film")
    ob.add_argument("video")
    ob.add_argument("--track", type=int, default=None, metavar="N", help="the bitmap track s:N")
    ob.add_argument("--lang", default=None)
    ob.add_argument("--engine", default="auto", choices=["auto", "tesseract", "vlm"])
    ob.add_argument("--prep", default=None, choices=list(config.OCR_PREP_MODES) if hasattr(config, "OCR_PREP_MODES") else ["binary", "fill", "gray", "fill3x"],
                    help="image preparation for tesseract (default: config OCR_PREP)")
    ob.add_argument("--reference", default=None, help="a .srt to compare with (default: the film's text track in that language)")
    ob.add_argument("--reference-track", type=int, default=None, metavar="N")
    ob.add_argument("--ref-lang", default=None, help="language of the text track to compare with when it differs from --lang (zh text vs yue = chi_tra OCR)")
    ob.add_argument("--limit", type=int, default=0, help="score only the first N OCR cues (0 = all)")
    ob.add_argument("--show", type=int, default=10)
    ob.set_defaults(fn=cmd_ocrbench)

    sc = sub.add_parser("scan", help="detect languages only — no ASR, nothing written beside the videos")
    sc.add_argument("paths", nargs="*", help="files or folders (default: the current folder)")
    sc.add_argument("--no-recursive", action="store_true")
    sc.add_argument("--keep-wav", action="store_true")
    add_common(sc)
    sc.set_defaults(fn=cmd_scan)

    cp = sub.add_parser("compare", help="word-level agreement between two ASR results of one file (needs --keep-work)")
    cp.add_argument("video")
    cp.add_argument("--keys", default=None, help="the two ASR keys to compare, comma-separated (default: the last two)")
    cp.add_argument("--work-dir", default=str(config.WORK_DIR))
    cp.set_defaults(fn=cmd_compare)

    c = sub.add_parser("clean", help="delete every work file and temp wav (leftovers of interrupted runs)")
    c.set_defaults(fn=cmd_clean)

    t = sub.add_parser("tracks", help="show audio tracks and which one would be used")
    t.add_argument("paths", nargs="+")
    t.add_argument("-r", "--recursive", action="store_true")
    t.set_defaults(fn=cmd_tracks)

    m = sub.add_parser("models", help="what is ready: the translator presets in Ollama and the ASR models in the Hugging Face cache")
    m.add_argument("--backend", default=None)
    m.add_argument("--url", default=None)
    m.set_defaults(fn=cmd_models)
    rs = sub.add_parser("refscore", help="score generated .srt files against the film's own human text tracks (chrF++ per minute, WER, coverage)")
    rs.add_argument("video"); rs.add_argument("--out", required=True, help="folder holding <stem>.<lang>.srt files to score")
    rs.add_argument("--lang", default=None, help="comma-separated codes to score (default: every language the film has a text track for)")
    rs.add_argument("--bin", type=float, default=60.0, help="bin size in seconds (default 60)")
    rs.add_argument("--json", default=None, help="also write the rows to this JSON file")
    rs.set_defaults(fn=cmd_refscore)
    ln = sub.add_parser("languages", help="list the subtitle languages (codes for --target / --source)")
    ln.add_argument("--profile", default="auto", help="which hardware profile's offer and routes to show (auto = this card's)")
    ln.set_defaults(fn=cmd_languages)
    wy = sub.add_parser("why", help="why each subtitle file of a video says what it says: transcript source, detection, speakers, terms, translator (from the work file)")
    wy.add_argument("video")
    wy.add_argument("--clip", default=None, help="the clip's work file instead (START-END as given to bench)")
    wy.set_defaults(fn=cmd_why)
    cf = sub.add_parser("config", help="show the settings and where they come from; `config targets en,th` saves the default subtitle languages")
    cf.add_argument("key", nargs="?", choices=["targets"], help="what to set (targets = the default subtitle languages)")
    cf.add_argument("value", nargs="?", help="the new value, e.g. en,th")
    cf.add_argument("--clear", action="store_true", help="forget the saved value (the environment or the built-in default applies again)")
    cf.set_defaults(fn=cmd_config)
    pl = sub.add_parser("pull", help="download models before the first run: the ASR models and the default translators, or the names given")
    pl.add_argument("names", nargs="*", help="translator presets, Ollama tags, ASR model names, 'asr' or 'defaults' (default: defaults)")
    pl.add_argument("--all", action="store_true", help="every translator preset as well as the ASR models")
    pl.add_argument("--url", default=None, help="Ollama URL (default: MLSUBGEN_LLM_URL or http://127.0.0.1:11434)")
    pl.add_argument("--speaker-embedding", default=None, metavar="FILE",
                    help="which speaker embedding model `pull speakers` fetches (a file of sherpa-onnx's speaker-recongition-models release)")
    pl.set_defaults(fn=cmd_pull)

    s = sub.add_parser("selftest", help="exercise segmentation, filters, parsing and typesetting without a GPU")
    s.set_defaults(fn=cmd_selftest)

    w = sub.add_parser("serve", help="the worker: run queued jobs one at a time (started by mlsubgen-worker.service)")
    w.add_argument("--poll", type=float, default=config.WORKER_POLL_SEC, help="seconds between queue checks")
    w.set_defaults(fn=cmd_serve)
    wb = sub.add_parser("web", help="the browser front end over the queue (started by mlsubgen-web.service)")
    wb.add_argument("--host", default=config.WEB_HOST)
    wb.add_argument("--port", type=int, default=config.WEB_PORT)
    wb.set_defaults(fn=cmd_web)
    j = sub.add_parser("jobs", help="show the job queue")
    j.add_argument("--all", action="store_true", help="every job, not just the last 20")
    j.set_defaults(fn=cmd_jobs)
    lg = sub.add_parser("log", help="where a job's log is, and its last lines")
    lg.add_argument("id", type=int)
    lg.add_argument("-n", "--lines", type=int, default=20)
    lg.set_defaults(fn=cmd_log)
    cn = sub.add_parser("cancel", help="stop a queued or running job (finished files keep their .srt)")
    cn.add_argument("id", type=int)
    cn.set_defaults(fn=cmd_cancel)
    rt = sub.add_parser("retry", help="queue a failed, cancelled or done job again")
    rt.add_argument("id", type=int)
    rt.set_defaults(fn=cmd_retry)
    pa = sub.add_parser("pause", help="stop a queued or running job and keep it out of the queue until `resume` (survives reboots)")
    pa.add_argument("id", type=int)
    pa.set_defaults(fn=cmd_pause)
    rs = sub.add_parser("resume", help="put a paused job back in the queue; it carries on from its work files")
    rs.add_argument("id", type=int)
    rs.set_defaults(fn=cmd_resume)
    pg = sub.add_parser("purge", help="drop failed and cancelled jobs from the queue listing (--done: finished ones too; or IDs)")
    pg.add_argument("ids", nargs="*", type=int, help="specific job ids (must be finished)")
    pg.add_argument("--done", action="store_true", help="also remove jobs that finished successfully")
    pg.set_defaults(fn=cmd_purge)

    args = list(sys.argv[1:] if argv is None else argv)
    # `mlsubgen help` / `mlsubgen --help` / `mlsubgen help <command>` / `mlsubgen <command> --help`
    if args and args[0] in ("help", "?", "-h", "--help"):
        if len(args) > 1 and args[1] in sub.choices:
            sub.choices[args[1]].print_help()
        else:
            print(HELP_TEXT, end="")
        return 0
    # `mlsubgen add …` = `mlsubgen run … --queue`
    if args and args[0] == "add":
        args = ["run"] + args[1:] + ["--queue"]
    # `mlsubgen` with no subcommand (or straight paths/options) means `mlsubgen run ...`
    if not args or (args[0] not in sub.choices and args[0] not in ("-h", "--help", "--version")):
        args = ["run"] + args
    a = ap.parse_args(args)
    a.argv = args                                   # `enqueue` stores the run arguments as given
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        # Ctrl-C, `mlsubgen cancel/pause`, a worker stop or a reboot all arrive as SIGINT: a clean interruption, not
        # an error — every finished stage (and every decoded ASR chunk) is already in the work files. Exit 130 is
        # what the worker reads as "interrupted" (2026-09-27: it used to end in a torch traceback in the job log).
        _log("\n[interrupted] stopping — finished stages and decoded chunks are in the work files; the next run resumes from there")
        return 130


if __name__ == "__main__":
    sys.exit(main())
