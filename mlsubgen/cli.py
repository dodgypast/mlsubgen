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
    p.add_argument("--speakers", default="off", metavar="off|auto|N",
                   help="speaker diarization (0.4.0, CPU, `mlsubgen pull speakers` first): auto = find the speakers, "
                        "N = there are N; a speaker change closes a cue and the translator is told who is talking "
                        "(labels are hints, never written to the subtitles). Default off")
    p.add_argument("--speaker-threshold", type=float, default=None,
                   help=f"clustering threshold for --speakers auto (default {config.SPEAKER_THRESHOLD}; smaller = more speakers)")
    p.add_argument("--speaker-embedding", default=None, metavar="FILE",
                   help="the speaker embedding model (a file from sherpa-onnx's speaker-recongition-models release; "
                        "`mlsubgen pull speakers` fetches it). See config.py for the known ones")
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


def parse_targets(text: str | None, warn: bool = True) -> list[str]:
    out = []
    for t in (text or config.DEFAULT_TARGETS).replace(";", ",").split(","):
        t = t.strip().lower()
        if t and t not in out:
            out.append(t)
    if not out:
        raise SystemExit("no target language (--target en,th)")
    unknown = [t for t in out if t not in config.LANG_NAMES]
    if unknown and warn:
        # not an error: the translator is simply asked for the code as written, and typesetting uses the defaults
        _log(f"[targets] unknown language code(s) {', '.join(unknown)} — the translator will be asked for them as written; "
             f"known codes: mlsubgen languages")
    return out


def check_speakers(a: argparse.Namespace) -> None:
    v = str(getattr(a, "speakers", "off") or "off").lower()
    if v not in ("off", "auto") and not v.isdigit():
        raise SystemExit("--speakers takes off, auto or a number of speakers (e.g. --speakers 3)")
    a.speakers = v
    if getattr(a, "speaker_embedding", None):
        config.set_speaker_embedding(a.speaker_embedding)


def cmd_languages(a: argparse.Namespace) -> int:
    """The subtitle languages: every code is a target (the translator writes it) and a source (decoded by Qwen3-ASR
    where its aligner covers the language, by whisper elsewhere)."""
    defaults = set(config.DEFAULT_TARGETS.split(","))
    print(f"{'code':<5} {'language':<12} {'native':<18} {'ASR':<8} default")
    for code, name in sorted(config.LANG_NAMES.items(), key=lambda kv: kv[1]):
        engine = "qwen" if code in config.ALIGNER_LANGS else "whisper"
        print(f"{code:<5} {name:<12} {config.NATIVE_NAMES.get(code, ''):<18} {engine:<8} {'yes' if code in defaults else ''}")
    print(f"\n{len(config.LANG_NAMES)} languages · defaults: {config.DEFAULT_TARGETS} (MLSUBGEN_TARGETS, or --target per run)")
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
  mlsubgen --target en,th,de FOLDER      one .srt per language;   mlsubgen languages   lists the {len(config.LANG_NAMES)} codes
  mlsubgen --source ja FOLDER            skip the language detector: the audio is Japanese
  mlsubgen --overwrite FILE              redo one file from scratch

COMMANDS
  subtitles   run        (default) subtitle the videos in the given files/folders
              scan       detect the languages only — no ASR, nothing written; one line per file
              bench      compare translators on one video or a clip:  mlsubgen bench VIDEO --clip 0:10:00-0:20:00
              lidbench   score the language detector on a multilingual film against its forced subtitle track
  models      pull       download models: mlsubgen pull | pull gemma4 | pull some/ollama:tag | pull asr | pull --all
              models     what is ready — translator presets in Ollama, ASR models in the Hugging Face cache
              languages  the {len(config.LANG_NAMES)} subtitle languages: code, name, native name, which engine decodes it
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
    from .pipeline import (Job, NotSupported, asr_key_for, check_source, cue_key_for, emit, missing_targets, speakers_cached,
                           speakers_wanted, srt_path_for, stage_asr, stage_audio, stage_cues, stage_lid, stage_merge,
                           stage_speakers, stage_subs, stage_translate, usable_turns)
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
                 a.window, want, source, speakers=a.speakers, speaker_threshold=a.speaker_threshold) for v, want in todo]
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
            job.work_file.unlink(missing_ok=True)

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
        satisfied, source = plan_sources(job.video, pr.subs, job.targets, a.subs, code_for_tag(pr.chosen.language))
        if source is not None or len(satisfied) == len(job.targets):
            return None                                     # the text route needs no audio
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
                data = work.load(job.work_file)
                _log(f"\n=== {job.video.name}")
                rem = remembered_skip(job)
                if rem:
                    _log(f"[skip] {job.video.name}: {rem['reason']} (remembered from {rem['when']}; "
                         f"--source, --overwrite or --audio-track retries it)")
                    skipped.append(f"{job.video.name} ({rem['reason']}, remembered)")
                    record_skip(job.video, rem["reason"])
                    continue
                data.pop("skipped", None)                 # a retry: the old verdict must not survive a later save
                try:
                    try:
                        done_targets, key = stage_subs(job, data, a.subs, job.targets)
                        if done_targets:
                            from_embedded += len(done_targets)
                            job.targets = [t for t in job.targets if t not in done_targets]
                            if not job.targets:
                                _log("[subs] nothing to write: every target is already embedded")
                                cleanup(job, work_file=True)
                                continue
                        if key:
                            keys[job.video] = key                 # cues are in the work file; stage 2 translates them
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
                n = emit(done, out, target)
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
    target = parse_targets(a.target)[0]
    source = a.source or ("ja" if a.assume_ja else None)
    job = Job(video, clip, a.audio_track, Path(a.work_dir).expanduser(), config.TMP_DIR, context, glossary, a.genre,
              a.window, [target], source, speakers=a.speakers, speaker_threshold=a.speaker_threshold)
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
    tag = video.stem + (f".{int(clip[0])}-{int(clip[1])}" if clip else "") + (f".spk-{job.speakers}" if speakers_wanted(job) else "")
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
    assert config.pick_profile("auto", 12.0) == "12gb" and config.pick_profile("auto", 8.0) == "8gb" and config.pick_profile("auto", 16.0) == "12gb"
    assert config.pick_profile("8gb", 48.0) == "8gb"
    try:
        config.pick_profile("huge", None); raise AssertionError("an unknown profile must be rejected")
    except ValueError:
        pass
    assert all(config.PROFILES[p]["default"] in TRANSLATORS and set(config.PROFILES[p]["routes"].values()) <= set(TRANSLATORS)
               for p in config.PROFILES), "every profile's presets must exist"
    config.apply_profile("8gb")
    assert config.ASR_SEQUENTIAL and config.WHISPER_COMPUTE == "int8_float16" and config.DEFAULT_TRANSLATOR == "gemma4-e4b"
    config.apply_profile("full")
    assert not config.ASR_SEQUENTIAL and config.WHISPER_COMPUTE == "float16" and config.TRANSLATE_ROUTES[("ja", "en")] == "qwen3.8"

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
    ln = sub.add_parser("languages", help="list the subtitle languages (codes for --target / --source)")
    ln.set_defaults(fn=cmd_languages)
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
