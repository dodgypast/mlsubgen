"""Stage orchestration shared by `run`, `scan` and `bench`. Each stage is cached in the work file.

  subs → audio (probe, extract, VAD) → lid (language per span) → asr (per chunk, forced language, routed engine)
       → cues → translate (per target, routed translator; copy-through for cues already in the target) → .srt
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, lid, merge, work
from .asr import Engines, Word, engine_for, words_from_dicts, words_to_dicts
from .audio import extract_wav, load_wav
from .clean import filter_cues
from .probe import ProbeResult, describe_tracks, probe
from .segment import Cue, build_cues, normalise_timing
from .srt import typeset, write_srt
from .subs import MIN_CUES, code_for_tag, cues_from_track, extract_track, pick, plan_sources
from .translate import ClientPool, translate_cues
from .vad import Span, speech_spans


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Job:
    video: Path
    clip: tuple[float, float] | None = None
    audio_track: int | None = None
    work_dir: Path = config.WORK_DIR
    tmp_dir: Path = config.TMP_DIR
    context: str = ""                # free text: what the programme is about, names — biases the ASR and the LLM
    glossary: dict | None = None
    genre: str = "a documentary / interview programme"
    window: int = config.WINDOW_CUES
    targets: list[str] = field(default_factory=lambda: ["en"])
    source: str | None = None        # --source LANG: skip the detector, every span is this language
    speakers: str = "off"            # --speakers off | auto | N (0.4.0): diarize, split cues at speaker changes, hint the translator
    speaker_threshold: float | None = None

    @property
    def work_file(self) -> Path:
        return work.work_path(self.video, self.work_dir, self.clip)

    @property
    def wav(self) -> Path:
        # named after the work file: byte-limited, and unique per video (and clip) even when stems repeat across folders
        return self.tmp_dir / (self.work_file.stem + ".16k.wav")


def srt_path_for(video: Path, lang: str = "en") -> Path:
    return video.with_name(f"{video.stem}.{lang}.srt")


def missing_targets(video: Path, targets: list[str], overwrite: bool = False, since: float | None = None) -> list[str]:
    """Targets still to write. Without --overwrite: the ones with no .srt. With it: every target — except, when
    `since` is given, a target whose .srt was written after that moment: it was rewritten by THIS job before it
    was paused, rebooted or interrupted, and is done (2026-09-27; the worker passes the job's creation time)."""
    out = []
    for t in targets:
        p = srt_path_for(video, t)
        if not p.exists():
            out.append(t)
        elif overwrite:
            try:
                fresh = bool(since) and p.stat().st_mtime >= since
            except OSError:
                fresh = False
            if not fresh:
                out.append(t)
    return out


# ── stage 0: embedded subtitle tracks ─────────────────────────────────────────────────────────────────────
def stage_subs(job: Job, data: dict, mode: str, targets: list[str]) -> tuple[list[str], str | None]:
    """Embedded subtitle tracks, checked before any audio work. Returns (targets that are already embedded as
    text tracks — left alone, nothing written; cue key of an embedded source-language track to translate from,
    or None when the audio has to be transcribed). See subs.plan_embedded for the modes."""
    if mode == "ignore":
        return [], None
    pr = probe(job.video, job.audio_track)
    spoken = code_for_tag(pr.chosen.language) if pr.chosen else None     # the audio tag: prefer a track in that language
    satisfied, source = plan_sources(job.video, pr.subs, targets, mode, spoken)
    for t in satisfied:
        track = pick(pr.subs, t)
        _log(f"[subs] {config.LANG_NAMES.get(t, t)} subtitles are embedded (s:{track.index} {track.codec}"
             f"{', ' + track.title if track.title else ''}) — no .{t}.srt written")
    if source is None:
        bitmap = [t for t in pr.subs if not t.is_text]
        if bitmap and len(satisfied) < len(targets):
            _log(f"[subs] bitmap subtitle track(s) present ({', '.join(t.codec for t in bitmap)}) — need OCR, ignored; using ASR")
        return satisfied, None
    src, lang = source
    name = config.LANG_NAMES.get(lang, lang)
    if isinstance(src, Path):
        cues = cues_from_track(src, lang)
        key = f"sidecar|{src.name}|{lang}"
        where = f"{src.name} beside the video"
        origin = {"sidecar": str(src), "language": lang}
    else:
        tmp = job.tmp_dir / (job.work_file.stem + f".{lang}.srt")
        n = extract_track(job.video, src.index, tmp, src.codec)
        cues = cues_from_track(tmp, lang)
        tmp.unlink(missing_ok=True)
        key = f"embedded|s:{src.index}|{src.codec}|{lang}"
        where = f"embedded s:{src.index} {src.codec}{', ' + src.title if src.title else ''}"
        origin = {"embedded_subtitles": src.index, "codec": src.codec, "language": lang, "raw_cues": n}
    if len(cues) < MIN_CUES:
        _log(f"[subs] {name} subtitles ({where}) have only {len(cues)} usable cues — ignored")
        return satisfied, None
    data.setdefault("video", str(job.video))
    data["duration"] = pr.duration
    data["source"] = origin
    data.setdefault("cues", {})[key] = {"cues": [c.to_dict() for c in cues], "stats": {"source_cues": len(cues)}}
    data["lid"] = lid.forced([], lang).to_dict()
    work.save(job.work_file, data)
    _log(f"[subs] {name} subtitles ({where}): {len(cues)} cues — used as the transcript, no ASR")
    return satisfied, key


# ── stage 1a: audio and VAD ───────────────────────────────────────────────────────────────────────────────
def _audio_note(bad: int, filled: bool) -> str:
    if bad:
        return (f" — {bad} undecodable packets skipped, gaps filled with silence" if filled else
                f" — {bad} undecodable packets skipped; container timestamps unusable, gaps NOT filled: timings may drift")
    if not filled:
        return " — container timestamps unusable, plain extraction used"
    return ""


def stage_audio(job: Job, data: dict, need_wav: bool = True,
                prefetched: tuple[int, bool] | None = None) -> tuple[ProbeResult, list[Span]]:
    """Probe, extract the 16 kHz wav (unless prefetched in the background, or not needed because everything
    downstream is cached), and run the VAD. Returns the probe and the speech spans (unlabelled)."""
    pr = probe(job.video, job.audio_track)
    if pr.chosen is None:
        raise SystemExit(f"{job.video.name}: {pr.reason}\n{describe_tracks(pr)}")
    data.setdefault("video", str(job.video))
    data["audio_track"] = {"index": pr.chosen.index, "language": pr.chosen.language, "reason": pr.reason,
                           "codec": pr.chosen.codec}
    data["duration"] = pr.duration
    if "vad" in data and not need_wav:
        return pr, [Span(s, e) for s, e in data["vad"]["spans"]]
    if not job.wav.exists():
        t0 = time.time()
        start, end = (job.clip if job.clip else (None, None))
        expected = (job.clip[1] - job.clip[0]) if job.clip else (pr.chosen.duration or pr.duration)
        bad, filled = extract_wav(job.video, pr.chosen.index, job.wav, start, end, expected)
        _log(f"[audio] a:{pr.chosen.index} ({pr.reason}) → {job.wav.name} in {time.time() - t0:.1f}s{_audio_note(bad, filled)}")
        if bad or not filled:
            data["audio_track"]["undecodable_packets"] = bad
            data["audio_track"]["gaps_filled"] = filled
    elif prefetched is not None:
        bad, filled = prefetched
        _log(f"[audio] a:{pr.chosen.index} ({pr.reason}) → {job.wav.name} (extracted in the background){_audio_note(bad, filled)}")
        if bad or not filled:
            data["audio_track"]["undecodable_packets"] = bad
            data["audio_track"]["gaps_filled"] = filled
    if "vad" in data:
        return pr, [Span(s, e) for s, e in data["vad"]["spans"]]
    audio = load_wav(job.wav)
    total = len(audio) / 16000.0
    t0 = time.time()
    spans = speech_spans(audio)
    speech = sum(s.dur for s in spans)
    data["vad"] = {"spans": [[round(s.start, 3), round(s.end, 3)] for s in spans],
                   "speech_sec": round(speech, 1), "total_sec": round(total, 1)}
    _log(f"[vad] {len(spans)} speech spans, {speech / 60:.1f} min of speech in {total / 60:.1f} min, {time.time() - t0:.1f}s")
    if not job.clip:
        _check_duration(total, pr, data)
    work.save(job.work_file, data)
    return pr, spans


def _check_duration(total: float, pr: ProbeResult, data: dict) -> None:
    """The wav must be as long as its track, or every cue after the first dropped packet lands early."""
    track, cont = pr.chosen.duration, pr.duration
    ref = track or cont
    if not ref:
        return
    if abs(total - ref) > 1.0:
        _log(f"[audio] ⚠ wav {total:.1f}s vs audio track {track:.1f}s / container {cont:.1f}s — the wav does not match "
             f"its track, so timings may drift; check a line near the end against the video")
        data["vad"]["duration_mismatch_sec"] = round(total - ref, 1)
    elif cont and cont - total > 1.0:
        _log(f"[audio] note: the audio track ends {cont - total:.0f}s before the container's {cont:.0f}s — nothing to fix")


# ── stage 1b: language identification ─────────────────────────────────────────────────────────────────────
class NotSupported(Exception):
    """The file's dominant language is not one we subtitle from (or could not be determined)."""


def stage_lid(job: Job, data: dict, audio, spans: list[Span], engines: Engines, turns=None) -> lid.LidResult:
    """Label every span with its language (cached per LID_VERSION and speaker mode; --source forces one language).
    With `turns` (the speakers stage, run before this one) the detector is speaker-aware: its result's spans are
    the VAD spans cut at speaker changes, so callers must take `res.spans` for the chunking."""
    if job.source:
        res = lid.forced(spans, job.source)
        data["lid"] = res.to_dict()
        work.save(job.work_file, data)
        _log(f"[lid] {res.summary()}")
        return res
    mode = job.speakers if turns else "off"
    cached = data.get("lid")
    if cached and cached.get("version") == config.LID_VERSION and not cached.get("forced") \
            and cached.get("speakers", "off") == mode:
        res = lid.LidResult.from_dict(cached)
        if len(res.spans) == len(spans):
            for s, c in zip(spans, res.spans):
                s.lang, s.speaker = c.lang, c.speaker
            res.spans = spans
        if res.spans:
            _log(f"[lid] cached: {res.summary()}")
            return res
    t0 = time.time()
    res = lid.identify(audio, spans, engines, turns=turns)
    data["lid"] = res.to_dict()
    data["lid"]["speakers"] = mode
    work.save(job.work_file, data)
    _log(f"[lid] {res.summary()}  ({len(res.windows)} windows, {time.time() - t0:.0f}s)"
         + (f"  — {'; '.join(res.notes)}" if res.notes else ""))
    return res


def check_source(res: lid.LidResult, allowed: set[str] = config.SOURCE_LANGS) -> None:
    """Raise NotSupported unless the dominant language is one we subtitle from."""
    if res.forced:
        return
    if not res.dominant:
        raise NotSupported("language could not be determined (no confident window — see the lid entry in the work file)")
    if res.dominant not in allowed:
        raise NotSupported(f"audio is {config.LANG_NAMES.get(res.dominant, res.dominant)} — not a supported source; "
                           f"--source {res.dominant} to force it through")


# ── stage 1c: ASR ─────────────────────────────────────────────────────────────────────────────────────────
def asr_mode(force: str | None) -> str:
    return force or config.ASR_ENGINE            # dual | auto | qwen | whisper


def asr_key_for(engines: Engines, context: str, force: str | None) -> str:
    return work.asr_key(asr_mode(force), f"{engines.key}|v{config.ASR_VERSION}", context)


def _chunk_words(words: list[Word], c: Span) -> list[Word]:
    return [w for w in words if c.start - 0.01 <= w.start <= c.end + 0.01]


def _chunk_text(words: list[Word], c: Span) -> str:
    joiner = "" if (c.lang or "ja") in merge.NO_SPACE else " "
    return joiner.join(w.text for w in _chunk_words(words, c)).strip()


def _partial(job: Job, data: dict, key: str):
    """Per-chunk ASR checkpoints for one ASR key, kept in the work file under `asr_partial` (NOT under `asr`, whose
    entries mean "complete" to every reader) until the stage finishes. -> (done: {pass: {index: {...}}},
    checkpoint(pass)). A pause, reboot or crash mid-file resumes from the next chunk instead of chunk 1
    (2026-09-27: a 247-chunk file paused at chunk 144 used to start again from the top)."""
    part = data.setdefault("asr_partial", {}).setdefault(key, {})

    def checkpoint(name: str):
        def save(i: int, c: Span, got: list[Word], entry: dict) -> None:
            part.setdefault(name, {})[str(i)] = {"start": c.start, "words": words_to_dicts(got), "log": entry}
            work.save(job.work_file, data)
        return save
    return part, checkpoint


def _partial_done(data: dict, key: str) -> None:
    (data.get("asr_partial") or {}).pop(key, None)
    if not data.get("asr_partial"):
        data.pop("asr_partial", None)


def stage_asr(job: Job, data: dict, engines: Engines, audio, chunks: list[Span], force: str | None = None) -> list[Word]:
    """Every chunk decoded with its language forced, by the engine its language routes to (both engines stay
    loaded). Words carry the language of their chunk."""
    key = asr_key_for(engines, job.context, force)
    cached = (data.get("asr") or {}).get(key)
    if cached:
        _log(f"[asr] cached ({key})" + (" — merge pending" if cached.get("merge") == "pending" else ""))
        return words_from_dicts(cached["words"])
    t0 = time.time()
    mode = asr_mode(force)
    if mode == "dual" and engines.whisper is not None:
        return _stage_asr_dual(job, data, engines, audio, chunks, key, t0)
    by_engine: dict[str, list[Span]] = {}
    for c in chunks:
        by_engine.setdefault(engine_for(c.lang, "auto" if mode == "dual" else mode), []).append(c)
    if "whisper" in by_engine and engines.whisper is None:     # not installed: Qwen takes those too
        by_engine.setdefault("qwen", []).extend(by_engine.pop("whisper"))
        by_engine["qwen"].sort(key=lambda c: c.start)
    words: list[Word] = []
    log: list[dict] = []
    spans = [Span(s, e) for s, e in (data.get("vad") or {}).get("spans", [])]
    part, checkpoint = _partial(job, data, key)
    for name, cs in by_engine.items():
        _log(f"[asr] {name}: {len(cs)} chunk(s), languages {sorted({c.lang or '?' for c in cs})}")
        if config.ASR_SEQUENTIAL:               # the 8gb profile: one engine on the card at a time
            engines.release("whisper" if name == "qwen" else "qwen")
        w, lg = engines.get(name).transcribe(audio, cs, context=job.context, done=part.get(name), checkpoint=checkpoint(name))
        w, lg = retry_thin_chunks(engines, name, audio, cs, w, lg, spans, job.context)
        words += w
        log += lg
    words.sort(key=lambda w: w.start)
    mismatches = [l for l in log if l.get("note")]
    data.setdefault("asr", {})[key] = {"engines": engines.key, "context": job.context, "words": words_to_dicts(words),
                                        "chunks": sorted(log, key=lambda l: l["start"]), "elapsed": round(time.time() - t0, 1),
                                        "mismatches": len(mismatches)}
    _partial_done(data, key)
    work.save(job.work_file, data)
    _log(f"[asr] {len(words)} units in {time.time() - t0:.0f}s" + (f", {len(mismatches)} chunk note(s)" if mismatches else ""))
    return words


def _stage_asr_dual(job: Job, data: dict, engines: Engines, audio, chunks: list[Span], key: str, t0: float) -> list[Word]:
    """Both engines decode every chunk in a language both cover; the merge (an LLM pass, after the engines leave
    the GPU) reconciles them. Chunks in languages only whisper covers go to whisper alone. Until the merge runs,
    Qwen's words stand in as the provisional transcript. Every decoded chunk is checkpointed (see _partial)."""
    both = [c for c in chunks if (c.lang or "ja") in config.ALIGNER_LANGS]
    solo = [c for c in chunks if c not in both]
    _log(f"[asr] dual: {len(both)} chunk(s) through both engines" + (f", {len(solo)} through whisper only" if solo else ""))
    part, checkpoint = _partial(job, data, key)
    if config.ASR_SEQUENTIAL:                   # the 8gb profile: one engine on the card at a time
        engines.release("whisper")
    qw, qlog = engines.qwen.transcribe(audio, both, context=job.context, done=part.get("qwen"), checkpoint=checkpoint("qwen")) if both else ([], [])
    if config.ASR_SEQUENTIAL:
        engines.release("qwen")
    ww, wlog = engines.whisper.transcribe(audio, both, context=job.context, done=part.get("whisper"), checkpoint=checkpoint("whisper")) if both else ([], [])
    sw, slog = engines.whisper.transcribe(audio, solo, context=job.context, done=part.get("solo"), checkpoint=checkpoint("solo")) if solo else ([], [])
    dual = []
    for c, ql, wl in zip(both, qlog, wlog):
        dual.append({"start": c.start, "end": c.end, "lang": c.lang or "ja", "retimed": ql.get("retimed", 0),
                     "qwen": {"text": _chunk_text(qw, c), "words": words_to_dicts(_chunk_words(qw, c))},
                     "whisper": {"text": _chunk_text(ww, c), "words": words_to_dicts(_chunk_words(ww, c))}})
    provisional = sorted(qw + sw, key=lambda w: w.start)
    disagree = sum(1 for d in dual if merge.decide(d["qwen"]["text"], d["whisper"]["text"])[0] == "llm")
    data.setdefault("asr", {})[key] = {"engines": engines.key, "mode": "dual", "context": job.context,
                                        "words": words_to_dicts(provisional), "solo_words": words_to_dicts(sw),
                                        "dual": dual, "merge": "pending",
                                        "chunks": sorted(qlog + slog, key=lambda l: l["start"]),
                                        "whisper_chunks": wlog, "elapsed": round(time.time() - t0, 1)}
    _partial_done(data, key)
    work.save(job.work_file, data)
    _log(f"[asr] dual done in {time.time() - t0:.0f}s: {len(dual)} chunk pairs, {disagree} need the LLM to reconcile")
    return provisional


def stage_merge(job: Job, data: dict, key: str, client) -> list[Word]:
    """Reconcile the two transcripts of every dual chunk (LLM where they disagree), time the result from the
    engines' timelines, and make it the file's transcript. `client` may be None: Qwen's text then stands."""
    entry = (data.get("asr") or {}).get(key)
    if not entry or entry.get("merge") != "pending":
        return words_from_dicts(entry["words"]) if entry else []
    t0 = time.time()
    words: list[Word] = words_from_dicts(entry.get("solo_words", []))
    stats_all: dict[str, int] = {}
    by_lang: dict[str, list[dict]] = {}
    for d in entry["dual"]:
        by_lang.setdefault(d["lang"], []).append(d)
    for lang, items in by_lang.items():
        got, stats = merge.merge_chunks(items, client, lang, config.LANG_NAMES.get(lang, lang), job.genre, _log)
        words += got
        for k, v in stats.items():
            stats_all[k] = stats_all.get(k, 0) + v
    words.sort(key=lambda w: w.start)
    entry["words"] = words_to_dicts(words)
    entry["merge"] = "done"
    entry["merge_stats"] = stats_all
    entry["merge_model"] = getattr(getattr(client, "tr", None), "model", None)
    work.save(job.work_file, data)
    _log(f"[merge] {sum(stats_all.values())} chunks in {time.time() - t0:.0f}s: agreed {stats_all.get('agree', 0)}, "
         f"reconciled by the LLM {stats_all.get('llm', 0)}, one engine only {stats_all.get('qwen', 0) + stats_all.get('whisper', 0)}"
         + (f", LLM failed {stats_all['llm_failed']} (Qwen kept)" if stats_all.get("llm_failed") else ""))
    return words


def _speech_in(spans: list[Span], a: float, b: float) -> float:
    return sum(max(0.0, min(s.end, b) - max(s.start, a)) for s in spans)


def retry_thin_chunks(engines: Engines, name: str, audio, chunks: list[Span], words: list[Word], log: list[dict],
                      spans: list[Span], context: str) -> tuple[list[Word], list[dict]]:
    """A chunk that came back with far fewer characters than its speech warrants is decoded again by the other
    engine, whose words replace the thin ones. The two engines fail differently: the LLM decoder can emit nothing
    over music, a whisper window can drop out or loop."""
    other_name = "whisper" if name == "qwen" else "qwen"
    thin = []
    for c, lg in zip(chunks, log):
        sp = _speech_in(spans, c.start, c.end)
        if sp >= config.ASR_RETRY_MIN_SPEECH and lg.get("chars", 0) / sp < config.ASR_RETRY_DENSITY:
            thin.append((c, lg, sp))
    if not thin:
        return words, log
    if config.ASR_SEQUENTIAL and other_name in ("qwen", "whisper"):
        engines.release("whisper" if other_name == "qwen" else "qwen")     # the 8gb profile: swap, never stack
    other = engines.get(other_name) if (other_name != "whisper" or engines.whisper is not None) else None
    if other is None or getattr(other, "engine", other_name) == name:
        _log(f"[asr] ⚠ {len(thin)} chunk(s) came back thin from {name} and no other engine is available")
        return words, log
    _log(f"[asr] {len(thin)} chunk(s) came back thin from {name} (" +
         ", ".join(f"{c.start:.0f}-{c.end:.0f}s: {lg['chars']} chars / {sp:.0f}s speech" for c, lg, sp in thin[:6]) +
         f"{', …' if len(thin) > 6 else ''}) — decoding them again with {other_name}")
    w2, lg2 = other.transcribe(audio, [c for c, _, _ in thin], context=context)
    for (c, lg, sp), l2 in zip(thin, lg2):
        if l2.get("chars", 0) > lg.get("chars", 0):
            words = [w for w in words if not (c.start - 0.01 <= w.start <= c.end + 0.01)]
            words += [w for w in w2 if c.start - 0.01 <= w.start <= c.end + 0.01]
            lg["note"] = (lg["note"] + "; " if lg.get("note") else "") + f"thin ({lg['chars']} chars): re-decoded by {other_name} → {l2['chars']} chars"
            lg["chars"], lg["units"], lg["engine"] = l2["chars"], l2["units"], other_name
        else:
            lg["note"] = (lg["note"] + "; " if lg.get("note") else "") + f"thin ({lg['chars']} chars): {other_name} found no more"
    words.sort(key=lambda w: w.start)
    return words, log


# ── stage 1d: speakers (0.4.0, opt-in) ───────────────────────────────────────────────────────────────────
def speakers_wanted(job: Job) -> bool:
    return (job.speakers or "off") != "off"


def speakers_cached(job: Job, data: dict) -> bool:
    s = data.get("speakers")
    return bool(s and s.get("version") == config.SPEAKERS_VERSION and s.get("mode") == job.speakers
                and s.get("threshold") == (job.speaker_threshold or config.SPEAKER_THRESHOLD)
                and s.get("embedding", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx") == config.SPEAKER_EMBEDDING_FILE)


def stage_speakers(job: Job, data: dict, audio) -> list:
    """Who spoke when, cached in the work file under `speakers` (per mode and threshold). Returns the turns.
    A missing package or model is reported once and the stage is skipped — the run goes on without labels."""
    from . import speakers as spk
    if not speakers_wanted(job):
        return []
    if speakers_cached(job, data):
        turns = [spk.Turn(s, e, l) for s, e, l in data["speakers"]["turns"]]
        _log(f"[speakers] cached: {spk.summary(turns)}")
        return turns
    ok, why = spk.available()
    if not ok:
        _log(f"[speakers] ⚠ {why} — continuing without speaker labels")
        return []
    t0 = time.time()
    n = int(job.speakers) if str(job.speakers).isdigit() else 0
    threshold = job.speaker_threshold or config.SPEAKER_THRESHOLD
    turns = spk.diarize(audio, n, threshold, progress=True)
    data["speakers"] = {"version": config.SPEAKERS_VERSION, "mode": job.speakers, "threshold": threshold,
                        "embedding": config.SPEAKER_EMBEDDING_FILE,
                        "turns": [[round(t.start, 3), round(t.end, 3), t.speaker] for t in turns],
                        "elapsed": round(time.time() - t0, 1)}
    work.save(job.work_file, data)
    _log(f"[speakers] {spk.summary(turns)} in {time.time() - t0:.0f}s")
    return turns


def usable_turns(data: dict) -> list:
    """The cached speaker turns, or [] when the diarization failed the cluster-count gate (one voice, or so many
    clusters that it is fragmentation): the detector then runs exactly as without --speakers (0.4.4)."""
    from . import speakers as spk
    s = data.get("speakers") or {}
    if not s.get("turns"):
        return []
    turns = [spk.Turn(a, b, l) for a, b, l in s["turns"]]
    clusters = len({t.speaker for t in turns})
    if clusters < config.SPEAKER_MIN_CLUSTERS or clusters > max(8, len(turns) * config.SPEAKER_MAX_CLUSTER_RATIO):
        return []
    return turns


def cue_key_for(job: Job, key: str) -> str:
    """Cues built with speaker labels are a different set (split at speaker changes), and a different mode or
    threshold gives different labels again: each gets its own cache entry (and translation entry)."""
    if not speakers_wanted(job):
        return key
    thr = job.speaker_threshold or config.SPEAKER_THRESHOLD
    return f"{key}|spk:{job.speakers}" + (f":{thr}" if job.speakers == "auto" else "")


def stage_cues(job: Job, data: dict, key: str, words: list[Word], spans: list[Span], audio=None) -> list[Cue]:
    """Words → cues → filters. `audio` (the 16 kHz signal, when the wav is still around) lets the gate check that
    a cue the VAD did not see has sound under it before keeping it. With speakers on, the words are labelled from
    the cached turns first, so a speaker change closes a cue and each cue carries its speaker."""
    from . import speakers as spk
    ckey = cue_key_for(job, key)
    cached = (data.get("cues") or {}).get(ckey)
    if cached:
        return [Cue.from_dict(d) for d in cached["cues"]]
    labelled = 0
    sstats: dict = {}
    if speakers_wanted(job) and (data.get("speakers") or {}).get("turns"):
        turns = [spk.Turn(s, e, l) for s, e, l in data["speakers"]["turns"]]
        labelled = spk.label_words(words, turns, stats=sstats)
        clusters = len({t.speaker for t in turns})
        ambiguous = sstats.get("ambiguous_words", 0)
        # the sanity gate: evidence too thin, too muddled or too fragmented is not used at all (the run then behaves
        # as without --speakers) — one speaker found; more ambiguous words than SPEAKER_MAX_AMBIGUOUS allows; or
        # so many clusters that every few turns is "a new voice" (2026-10-01: threshold 0.5 gave 80 speakers for
        # 185 turns of a ten-minute anime clip and the labels were noise — the gate must catch that, not just 1)
        fragmented = clusters > max(8, len(turns) * config.SPEAKER_MAX_CLUSTER_RATIO)
        if clusters < config.SPEAKER_MIN_CLUSTERS or fragmented or \
                (labelled + ambiguous and ambiguous / (labelled + ambiguous) > config.SPEAKER_MAX_AMBIGUOUS):
            _log(f"[speakers] ⚠ evidence not used: {clusters} cluster(s) for {len(turns)} turns, {ambiguous} ambiguous of "
                 f"{labelled + ambiguous} covered words — cues built without speaker labels"
                 + (" (over-fragmented: raise --speaker-threshold or give --speakers N)" if fragmented else ""))
            for w in words:
                w.speaker = ""
            labelled = 0
    cues = build_cues(words, stats=sstats)
    cues, stats = filter_cues(cues, spans, audio)
    cues = normalise_timing(cues)
    if labelled:
        stats["speaker_labelled_words"] = labelled
        stats["ambiguous_words"] = sstats.get("ambiguous_words", 0)
        stats["speaker_splits"] = sstats.get("speaker_splits", 0)
        stats["speaker_changes"] = sum(1 for a, b in zip(cues, cues[1:]) if a.speaker and b.speaker and a.speaker != b.speaker)
        stats["speakers"] = len({c.speaker for c in cues if c.speaker})
    data.setdefault("cues", {})[ckey] = {"cues": [c.to_dict() for c in cues], "stats": stats}
    work.save(job.work_file, data)
    langs: dict[str, int] = {}
    for c in cues:
        langs[c.lang] = langs.get(c.lang, 0) + 1
    _log(f"[cues] {len(cues)} cues {langs}  {stats}")
    return cues


# ── stage 2: translation per target ───────────────────────────────────────────────────────────────────────
def stage_translate(job: Job, data: dict, key: str, cues: list[Cue], target: str, pool: ClientPool,
                    force: bool = False, force_model: str | None = None) -> list[Cue]:
    tkey = f"{key}|{target}"
    cached = (data.get("translations") or {}).get(tkey)
    partial = bool(cached and cached.get("partial"))
    if cached and not partial and not force:
        _log(f"[tl] cached ({target})")
        return [Cue.from_dict(d) for d in cached["cues"]]
    fresh = [Cue.from_dict(c.to_dict()) for c in cues]
    for c in fresh:
        c.en = ""
        c.flags = [f for f in (c.flags or []) if f not in ("copied", "untranslated")] or None
    if partial and not force:
        # a run interrupted mid-translation (reboot, stop): carry over the windows it finished
        have = {int(d["idx"]): d.get("en", "") for d in cached["cues"]}
        n = 0
        for c in fresh:
            if have.get(c.idx):
                c.en = have[c.idx]; n += 1
        _log(f"[tl] resuming {target}: {n}/{len(fresh)} cues already translated")
    t0 = time.time()

    def checkpoint(cs: list[Cue]) -> None:
        data.setdefault("translations", {})[tkey] = {"target": target, "partial": True, "cues": [c.to_dict() for c in cs]}
        work.save(job.work_file, data)

    translate_cues(fresh, None, job.glossary, job.genre, window_size=job.window, checkpoint=checkpoint,
                   target=target, pool=pool, force_model=force_model)
    usage = {n: cl.usage.__dict__ for n, cl in pool.clients.items()}
    data.setdefault("translations", {})[tkey] = {"target": target, "cues": [c.to_dict() for c in fresh],
                                                  "models": sorted({cl.tr.model for cl in pool.clients.values()}),
                                                  "usage": usage, "elapsed": round(time.time() - t0, 1)}
    work.save(job.work_file, data)
    copied = sum(1 for c in fresh if c.flags and "copied" in c.flags)
    _log(f"[tl] {target} done in {time.time() - t0:.0f}s" + (f" ({copied} copied through)" if copied else ""))
    return fresh


def emit(cues: list[Cue], out: Path, lang: str = "en") -> int:
    items = typeset(cues, lang=lang)
    write_srt(out, items)
    return len(items)
