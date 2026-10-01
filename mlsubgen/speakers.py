"""Speaker diarization — who spoke when (0.4.0, 2026-10-02).

sherpa-onnx's offline pipeline: pyannote's segmentation-3.0 (MIT) exported to ONNX, a 3D-Speaker embedding model
(Apache-2.0) and sherpa's clustering, all fetched from sherpa-onnx's GitHub releases — no account, no user
agreement, no pyannote package — and run on the CPU, so the GPU profiles are untouched.

The labels (S1, S2 …) are hints, never output: they close a cue at a speaker change, and they tell the translator
who is talking so each character's register stays consistent across windows. They can be wrong (sherpa's
clustering is not pyannote's tuned pipeline, and animation with several similar voices over music is the hard
case), which is why the prompt says so and why `bench` decides whether they earn their place.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from . import config
from .asr import Word


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str          # "S1", "S2", … in order of first appearance

    @property
    def dur(self) -> float:
        return self.end - self.start


def model_paths() -> dict[str, "Path"]:
    from pathlib import Path
    d = Path(config.SPEAKER_MODEL_DIR)
    return {"segmentation": d / config.SPEAKER_SEGMENTATION_FILE, "embedding": d / config.SPEAKER_EMBEDDING_FILE}


def available() -> tuple[bool, str]:
    """(usable, why not): the sherpa-onnx package and both model files."""
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False, "sherpa-onnx is not installed (pip install sherpa-onnx)"
    missing = [k for k, p in model_paths().items() if not p.is_file()]
    if missing:
        return False, f"speaker model(s) missing: {', '.join(missing)} — run: mlsubgen pull speakers"
    return True, "ok"


def diarize(audio: np.ndarray, num_speakers: int = 0, threshold: float | None = None,
            progress: bool = False) -> list[Turn]:
    """Speaker turns for a 16 kHz mono signal. num_speakers 0 = decide by clustering threshold (smaller = more
    speakers). Speakers are renamed S1, S2 … in order of first appearance."""
    import sherpa_onnx
    paths = model_paths()
    threshold = config.SPEAKER_THRESHOLD if threshold is None else threshold
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(paths["segmentation"]),
                                                                               window_shift_ratio=0.1)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(paths["embedding"])),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers if num_speakers > 0 else -1,
                                                    threshold=threshold),
        min_duration_on=config.SPEAKER_MIN_ON, min_duration_off=config.SPEAKER_MIN_OFF,
    )
    if not cfg.validate():
        raise RuntimeError("speaker diarization config invalid — are both model files present? (mlsubgen pull speakers)")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if sd.sample_rate != 16000:
        raise RuntimeError(f"the segmentation model wants {sd.sample_rate} Hz audio")
    samples = np.ascontiguousarray(audio, dtype=np.float32)
    last = [time.time()]

    def cb(done: int, total: int) -> int:
        if progress and total and time.time() - last[0] > 5:
            _log(f"[speakers] {100 * done / total:.0f}%"); last[0] = time.time()
        return 0

    result = (sd.process(samples, callback=cb) if progress else sd.process(samples)).sort_by_start_time()
    names: dict[int, str] = {}
    turns: list[Turn] = []
    for r in result:
        label = names.setdefault(int(r.speaker), f"S{len(names) + 1}")
        turns.append(Turn(float(r.start), float(r.end), label))
    return turns


def label_words(words: list[Word], turns: list[Turn], reach: float = 0.35,
                min_share: float | None = None, stats: dict | None = None) -> int:
    """Give every word a speaker — or none. A word takes the speaker whose turns cover it most, but only when that
    speaker clearly wins: at least `min_share` of the word's duration (SPEAKER_MIN_OVERLAP) and at least twice the
    runner-up's share. A word nobody covers takes the nearest turn within `reach` seconds. Anything else stays
    unlabelled: unknown is better than confidently wrong, because the label steers the cue builder and the
    translator. Returns how many words got a label; `stats` (optional) gets the ambiguous count. Mutates in place."""
    if not turns:
        return 0
    min_share = config.SPEAKER_MIN_OVERLAP if min_share is None else min_share
    starts = np.array([t.start for t in turns])
    ends = np.array([t.end for t in turns])
    labels = [t.speaker for t in turns]
    n = ambiguous = 0
    for w in words:
        dur = max(w.end - w.start, 0.02)
        ov = np.minimum(ends, w.end) - np.maximum(starts, w.start)
        share: dict[str, float] = {}
        for k in np.nonzero(ov > 0)[0]:
            share[labels[k]] = share.get(labels[k], 0.0) + float(ov[k]) / dur
        if share:
            ranked = sorted(share.items(), key=lambda kv: -kv[1])
            best, second = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
            if best[1] >= min_share and best[1] >= 2 * second:
                w.speaker = best[0]; n += 1
            else:
                w.speaker = ""; ambiguous += 1
            continue
        dist = np.minimum(np.abs(starts - w.end), np.abs(ends - w.start))
        k = int(dist.argmin())
        if dist[k] <= reach:
            w.speaker = labels[k]; n += 1
        else:
            w.speaker = ""
    if stats is not None:
        stats["ambiguous_words"] = ambiguous
    return n


def dominant(words: list[Word]) -> str:
    """The speaker that covers most of a run of words (by duration); "" when none are labelled."""
    tally: dict[str, float] = {}
    for w in words:
        if w.speaker:
            tally[w.speaker] = tally.get(w.speaker, 0.0) + max(0.0, w.end - w.start) + 0.05
    return max(tally, key=tally.get) if tally else ""


def summary(turns: list[Turn]) -> str:
    by: dict[str, float] = {}
    for t in turns:
        by[t.speaker] = by.get(t.speaker, 0.0) + t.dur
    parts = ", ".join(f"{s} {d / 60:.1f}m" for s, d in sorted(by.items(), key=lambda kv: -kv[1]))
    return f"{len(by)} speaker(s), {len(turns)} turns: {parts}"
