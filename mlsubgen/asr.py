"""ASR engines. Both return the same thing: a list of Word(text, start, end, lang) in absolute seconds.

qwen    — Qwen3-ASR-1.7B (+ Qwen3-ForcedAligner-0.6B for timestamps). Lowest Japanese CER of the open models in
          2026 benchmarks; punctuates naturally, which the cue segmenter relies on. 30 languages; the aligner
          covers ja/zh/yue/en/ko/fr/de/it/pt/ru/es — not Thai.
whisper — faster-whisper (large-v3-turbo by default) with word timestamps: Thai and everything the aligner lacks,
          and the language detector with a probability.

Each chunk carries the language the LID decided (Span.lang); it is decoded with that language FORCED — the
accurate mode — and the result's script is checked against the label. A mismatch is re-decoded in auto mode
rather than accepted silently.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

from . import config
from .audio import SR, slice_audio
from .lid import code_from_name, script_matches
from .vad import Span


@dataclass
class Word:
    text: str
    start: float
    end: float
    lang: str = "ja"
    speaker: str = ""        # "S1", "S2" … from the speakers stage (0.4.0); "" = unlabelled / speakers off


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _plausible(w: Word, prev: Word | None) -> bool:
    dur = w.end - w.start
    return 0.02 <= dur <= config.ALIGN_MAX_WORD_SEC and (prev is None or w.start >= prev.start - 0.001)


def repair_timestamps(words: list[Word], start: float, end: float) -> tuple[list[Word], int]:
    """Undo the forced aligner's collapse. When it loses track it stamps every following word with one instant
    (zero-length, identical times), and the words after that are often junk too (17-second "words"). Within a
    chunk [start, end]: a run of broken words is re-spread from the last good word to the point where three
    plausible words in a row show the aligner back on track (else the chunk end), at ALIGN_SEC_PER_CHAR per
    character when there is room, compressed when there is not, placed right after the last good word (speech
    tends to continue). Returns the words and how many were re-timed."""
    n = len(words)
    if n == 0:
        return words, 0
    out = list(words)
    fixed = 0
    i = 0
    prev_good: Word | None = None
    while i < n:
        w = out[i]
        same_as_next = i + 1 < n and abs(out[i + 1].start - w.start) < 0.001 and abs(out[i + 1].end - w.end) < 0.001
        if _plausible(w, prev_good) and not same_as_next:
            prev_good = w
            i += 1
            continue
        # broken region: from i to the first index where three consecutive plausible words begin — and where the
        # room before them can hold the run at a fast-but-human rate; words stamped 0.3 s after a 1000-word
        # collapse look plausible and are not
        left = prev_good.end if prev_good else start
        j = i + 1
        while j < n:
            ok = all(k < n and _plausible(out[k], out[k - 1] if k > j else prev_good) for k in range(j, min(n, j + 3)))
            same = j + 1 < n and abs(out[j + 1].start - out[j].start) < 0.001 and abs(out[j + 1].end - out[j].end) < 0.001
            room = out[j].start - left
            need = sum(max(1, len(x.text)) for x in out[i:j]) * 0.04          # 25 chars/s: the fastest speech gets
            if ok and not same and room >= need:
                break
            j += 1
        right = out[j].start if j < n else end
        if right <= left:
            right = end if end > left else left + 0.5
        chars = max(1, sum(max(1, len(x.text)) for x in out[i:j]))
        width = min(right - left, chars * config.ALIGN_SEC_PER_CHAR)
        t = left
        for k in range(i, j):
            d = width * max(1, len(out[k].text)) / chars
            out[k] = Word(out[k].text, round(t, 3), round(min(t + d, right), 3), out[k].lang)
            t += d
        fixed += j - i
        prev_good = out[j - 1]
        i = j
    return out, fixed


def engine_for(lang: str | None, force: str | None = None) -> str:
    """Which engine decodes a chunk of this language: --asr forces one, else ASR_ROUTES."""
    if force and force != "auto":
        return force
    return config.ASR_ROUTES.get(lang or "", config.ASR_ROUTES["*"])


class QwenASR:
    engine = "qwen"

    def __init__(self, model_id: str = config.ASR_MODEL_QWEN, aligner_id: str = config.ALIGNER_MODEL,
                 device: str = "cuda:0", batch_size: int = 1, max_new_tokens: int = 3072):
        import torch
        from qwen_asr import Qwen3ASRModel
        try:  # otherwise transformers prints "Setting `pad_token_id`…" for every chunk
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        self.model_id = model_id
        self.aligner_id = aligner_id
        t0 = time.time()
        self.model = Qwen3ASRModel.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            max_inference_batch_size=batch_size, max_new_tokens=max_new_tokens,
            forced_aligner=aligner_id,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=device),
        )
        _log(f"[asr] loaded {model_id} + {aligner_id} in {time.time() - t0:.1f}s")

    def identify(self, piece: np.ndarray) -> tuple[str | None, str]:
        """Auto-mode decode of a short piece: (language code or None, text) — the LID's second opinion."""
        if len(piece) < SR // 2:
            return None, ""
        res = self.model.transcribe(audio=[(piece, SR)], language=[None], return_time_stamps=False)
        r = res[0]
        return code_from_name(getattr(r, "language", None)), (r.text or "").strip()

    def _decode(self, piece: np.ndarray, lang: str | None, context: str, timestamps: bool):
        kwargs = dict(audio=[(piece, SR)], language=[config.LANG_NAMES.get(lang) if lang else None],
                      return_time_stamps=timestamps)
        if context:
            kwargs["context"] = [context]
        return self.model.transcribe(**kwargs)[0]

    def transcribe(self, audio: np.ndarray, chunks: list[Span], context: str = "",
                   done: dict | None = None, checkpoint=None) -> tuple[list[Word], list[dict]]:
        """`done` = chunks already decoded by an earlier, interrupted pass ({index: {start, words, log}}) — reused,
        not decoded again; `checkpoint(i, chunk, words, log)` is called after every decoded chunk so the caller can
        persist it (2026-09-27: a pause or reboot mid-file used to throw the whole file's ASR away)."""
        words: list[Word] = []
        chunk_log: list[dict] = []
        done = done or {}
        reused = 0
        for i, c in enumerate(chunks):
            prev = done.get(str(i))
            if prev and abs(float(prev.get("start", -1.0)) - c.start) < 0.01:
                words += words_from_dicts(prev["words"])
                chunk_log.append(prev["log"])
                reused += 1
                continue
            if reused and reused == i:
                _log(f"[asr] resuming: {reused} of {len(chunks)} chunk(s) already decoded by qwen")
            piece = slice_audio(audio, c.start, c.end)
            if len(piece) < SR // 4:
                continue
            lang = c.lang or "ja"
            timestamps = lang in config.ALIGNER_LANGS
            t0 = time.time()
            r = self._decode(piece, lang, context, timestamps)
            text = (r.text or "").strip()
            note = ""
            fixed = 0
            if text and not script_matches(text, lang):
                # the label and the decode disagree: decode again with the language free, keep whatever it says
                r2 = self._decode(piece, None, context, timestamps)
                lang2 = code_from_name(getattr(r2, "language", None))
                if lang2 and (r2.text or "").strip() and script_matches((r2.text or "").strip(), lang2):
                    note = f"script mismatch for {lang}: re-decoded as {lang2}"
                    r, text, lang = r2, (r2.text or "").strip(), lang2
                else:
                    note = f"script mismatch for {lang}: kept (auto decode did not help)"
            n = 0
            got: list[Word] = []
            if timestamps and r.time_stamps:
                for t in r.time_stamps:
                    tx = (t.text or "").strip()
                    if not tx:
                        continue
                    got.append(Word(tx, c.start + float(t.start_time), c.start + float(t.end_time), lang))
                got, fixed = repair_timestamps(got, c.start, c.end)
                if fixed:
                    note = (note + "; " if note else "") + f"{fixed} of {len(got)} words re-timed (aligner collapse)"
                n = len(got)
            elif text:
                # no aligner for this language (or it returned nothing): spread the text over the chunk so nothing is lost
                got = [Word(text, c.start, c.end, lang)]
                n = 1
                if not timestamps:
                    note = (note + "; " if note else "") + "no aligner for this language: one unit for the chunk"
            words += got
            dt = time.time() - t0
            entry = {"i": i, "start": c.start, "end": c.end, "sec": round(c.dur, 2), "lang": lang,
                     "engine": "qwen", "elapsed": round(dt, 2), "rtf": round(dt / max(c.dur, 0.01), 3),
                     "chars": len(text), "units": n, "note": note, "retimed": fixed if timestamps else 0}
            chunk_log.append(entry)
            _log(f"[asr] chunk {i + 1}/{len(chunks)} {c.start:8.1f}-{c.end:8.1f}s  {lang:<3} {len(text):4d} chars  "
                 f"{n:4d} units  rtf {dt / max(c.dur, 0.01):.3f}{'  ⚠ ' + note if note else ''}")
            if checkpoint is not None:
                checkpoint(i, c, got, entry)
        return words, chunk_log

    def close(self) -> None:
        import torch
        del self.model
        torch.cuda.empty_cache()


class WhisperASR:
    engine = "whisper"

    def __init__(self, model_id: str = config.ASR_MODEL_WHISPER, device: str = "cuda", compute_type: str | None = None):
        import torch  # noqa: F401 — loads the bundled CUDA libraries (cuDNN, cuBLAS) that CTranslate2 links against
        from faster_whisper import WhisperModel
        self.model_id = model_id
        compute_type = compute_type or config.WHISPER_COMPUTE      # int8_float16 on the small profiles: half the memory
        t0 = time.time()
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        _log(f"[asr] loaded faster-whisper {model_id} ({compute_type}) in {time.time() - t0:.1f}s")

    def identify(self, piece: np.ndarray) -> tuple[str | None, float]:
        """(language code, probability) for a short piece — the LID's main detector."""
        if len(piece) < SR // 2:
            return None, 0.0
        if hasattr(self.model, "detect_language"):
            try:
                lang, prob, _ = self.model.detect_language(piece)
                return lang, float(prob)
            except TypeError:
                pass
        _, info = self.model.transcribe(piece, language=None, beam_size=1, without_timestamps=True)
        return info.language, float(getattr(info, "language_probability", 0.0) or 0.0)

    def transcribe(self, audio: np.ndarray, chunks: list[Span], context: str = "",
                   done: dict | None = None, checkpoint=None) -> tuple[list[Word], list[dict]]:
        """Same `done` / `checkpoint` contract as QwenASR.transcribe."""
        words: list[Word] = []
        chunk_log: list[dict] = []
        done = done or {}
        reused = 0
        for i, c in enumerate(chunks):
            prev = done.get(str(i))
            if prev and abs(float(prev.get("start", -1.0)) - c.start) < 0.01:
                words += words_from_dicts(prev["words"])
                chunk_log.append(prev["log"])
                reused += 1
                continue
            if reused and reused == i:
                _log(f"[asr] resuming: {reused} of {len(chunks)} chunk(s) already decoded by whisper")
            piece = slice_audio(audio, c.start, c.end)
            if len(piece) < SR // 4:
                continue
            lang = c.lang or "ja"
            t0 = time.time()
            segments, info = self.model.transcribe(
                piece, language=lang, beam_size=5, word_timestamps=True,
                condition_on_previous_text=False, vad_filter=False,
                initial_prompt=context or None, no_speech_threshold=0.6,
                compression_ratio_threshold=2.2, hallucination_silence_threshold=2.0,
            )
            n = 0
            chars = 0
            texts: list[str] = []
            got: list[Word] = []
            for seg in segments:
                chars += len(seg.text or "")
                texts.append(seg.text or "")
                if seg.words:
                    for w in seg.words:
                        tx = (w.word or "").strip()
                        if tx:
                            got.append(Word(tx, c.start + float(w.start), c.start + float(w.end), lang))
                            n += 1
                elif seg.text and seg.text.strip():
                    got.append(Word(seg.text.strip(), c.start + float(seg.start), c.start + float(seg.end), lang))
                    n += 1
            words += got
            text = " ".join(t.strip() for t in texts).strip()
            note = "" if not text or script_matches(text, lang) else f"script mismatch for {lang} (kept)"
            dt = time.time() - t0
            entry = {"i": i, "start": c.start, "end": c.end, "sec": round(c.dur, 2), "lang": lang,
                     "engine": "whisper", "elapsed": round(dt, 2), "rtf": round(dt / max(c.dur, 0.01), 3),
                     "chars": chars, "units": n, "note": note}
            chunk_log.append(entry)
            _log(f"[asr] chunk {i + 1}/{len(chunks)} {c.start:8.1f}-{c.end:8.1f}s  {lang:<3} {chars:4d} chars  "
                 f"{n:4d} units  rtf {dt / max(c.dur, 0.01):.3f}{'  ⚠ ' + note if note else ''}")
            if checkpoint is not None:
                checkpoint(i, c, got, entry)
        return words, chunk_log

    def close(self) -> None:
        del self.model


class Engines:
    """Both engines, loaded on first use and kept until close(): they fit beside each other (≈5 GB + ≈2 GB), so
    routing chunks between them costs nothing. Whisper is optional: without faster-whisper installed, the LID
    falls back to Qwen alone (less confident) and every chunk goes to Qwen."""

    def __init__(self, qwen_model: str | None = None, whisper_model: str | None = None):
        self.qwen_model = qwen_model or config.ASR_MODEL_QWEN
        self.whisper_model = whisper_model or config.ASR_MODEL_WHISPER
        self._qwen: QwenASR | None = None
        self._whisper: WhisperASR | None = None
        self.whisper_missing = False

    @property
    def qwen(self) -> QwenASR:
        if self._qwen is None:
            self._qwen = QwenASR(self.qwen_model)
        return self._qwen

    @property
    def whisper(self) -> WhisperASR | None:
        if self._whisper is None and not self.whisper_missing:
            try:
                self._whisper = WhisperASR(self.whisper_model)
            except Exception as e:  # noqa: BLE001 — not installed, model not downloaded (HF offline), CUDA libs missing
                self.whisper_missing = True
                _log(f"[asr] ⚠ faster-whisper unavailable ({type(e).__name__}: {str(e)[:160]}) — LID runs on Qwen alone, "
                     f"every chunk goes to Qwen; fix: see README 'whisper model'")
        return self._whisper

    def get(self, name: str):
        if name == "whisper" and self.whisper is not None:
            return self.whisper
        return self.qwen

    @property
    def key(self) -> str:
        return f"{self.qwen_model}+{self.whisper_model}"

    def release(self, name: str) -> None:
        """Free one engine now (the 8gb profile never holds both); it reloads on its next use."""
        if name == "qwen" and self._qwen is not None:
            self._qwen.close(); self._qwen = None
            _log("[asr] released qwen (sequential profile)")
        elif name == "whisper" and self._whisper is not None:
            self._whisper.close(); self._whisper = None
            import torch
            torch.cuda.empty_cache()
            _log("[asr] released whisper (sequential profile)")

    def close(self) -> None:
        for e in (self._qwen, self._whisper):
            if e is not None:
                e.close()
        self._qwen = self._whisper = None


def make_engine(name: str, model_id: str | None = None):
    if name == "qwen":
        return QwenASR(model_id or config.ASR_MODEL_QWEN)
    if name == "whisper":
        return WhisperASR(model_id or config.ASR_MODEL_WHISPER)
    raise ValueError(f"unknown ASR engine {name!r} (qwen | whisper)")


def words_to_dicts(words: list[Word]) -> list[dict]:
    return [asdict(w) for w in words]


def words_from_dicts(items: list[dict]) -> list[Word]:
    return [Word(d["text"], float(d["start"]), float(d["end"]), d.get("lang", "ja"), d.get("speaker", "")) for d in items]
