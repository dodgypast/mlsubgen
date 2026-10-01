"""Is the VAD the blind spot? Runs whisper over the WHOLE file in fixed 30 s windows (no VAD involved), then lists
the speech it hears that lies outside every VAD span of the kept work file — i.e. what the pipeline never gave
either engine. Needs the video (re-extracts the wav) and the GPU.

    ~/mlsubgen/.venv/bin/python ~/mlsubgen/tools/vad_check.py "/path/to/video.mp4" [--threshold 0.35]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlsubgen import config, work                          # noqa: E402
from mlsubgen.audio import SR, extract_wav, load_wav          # noqa: E402
from mlsubgen.probe import probe                              # noqa: E402
from mlsubgen.vad import Span, speech_ratio                   # noqa: E402


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


video = Path(sys.argv[1]).expanduser()
threshold = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else None
wf = work.work_path(video)
data = work.load(wf)
spans = [Span(s, e) for s, e in (data.get("vad") or {}).get("spans", [])]
pr = probe(video)
wav = config.TMP_DIR / (wf.stem + ".check.wav")
if not wav.exists():
    extract_wav(video, pr.chosen.index, wav, None, None, pr.chosen.duration or pr.duration)
audio = load_wav(wav)
total = len(audio) / SR
print(f"{video.name}: {total / 60:.1f} min · VAD spans in the work file: {len(spans)} ({sum(s.dur for s in spans) / 60:.1f} min)")

import torch  # noqa: E402,F401
from faster_whisper import WhisperModel  # noqa: E402
model = WhisperModel(config.ASR_MODEL_WHISPER, device="cuda", compute_type="float16")
words = []
t = 0.0
while t < total:
    piece = audio[int(t * SR):int(min(total, t + 30.0) * SR)]
    segs, _ = model.transcribe(piece, language="ja", beam_size=5, word_timestamps=True, condition_on_previous_text=False,
                               vad_filter=False, no_speech_threshold=0.6, compression_ratio_threshold=2.2,
                               hallucination_silence_threshold=2.0)
    for seg in segs:
        for w in (seg.words or []):
            if w.word.strip():
                words.append((t + w.start, t + w.end, w.word.strip()))
    t += 30.0
print(f"whisper (fixed windows, no VAD) heard {len(words)} words")

outside = [(s, e, w) for s, e, w in words if speech_ratio(spans, s, e) < 0.2]
print(f"words with no VAD span under them: {len(outside)} ({len(outside) / max(len(words), 1) * 100:.0f}%)")
# group into stretches
stretches = []
for s, e, w in outside:
    if stretches and s - stretches[-1][1] < 1.0:
        stretches[-1][1] = e; stretches[-1][2] += w
    else:
        stretches.append([s, e, w])
stretches.sort(key=lambda x: -(len(x[2])))
print("longest stretches the VAD missed (listen at these times):")
for s, e, txt in stretches[:25]:
    print(f"  {mmss(s)}–{mmss(e)}  {txt[:70]}")

if threshold is not None:
    from silero_vad import load_silero_vad, get_speech_timestamps  # noqa: E402
    m = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(audio), m, sampling_rate=SR, threshold=threshold,
                               min_silence_duration_ms=config.VAD_MIN_SILENCE_MS, min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
                               speech_pad_ms=config.VAD_SPEECH_PAD_MS, return_seconds=True)
    sp2 = [Span(float(x["start"]), float(x["end"])) for x in ts]
    out2 = [(s, e, w) for s, e, w in words if speech_ratio(sp2, s, e) < 0.2]
    print(f"\nwith VAD threshold {threshold}: {len(sp2)} spans ({sum(s.dur for s in sp2) / 60:.1f} min); "
          f"whisper words still outside: {len(out2)} ({len(out2) / max(len(words), 1) * 100:.0f}%)")
wav.unlink(missing_ok=True)
