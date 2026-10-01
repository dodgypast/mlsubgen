"""Before/after on kept work files: what the aligner collapse and the old gate cost, and what the timestamp repair
plus the content-aware gate recover. Offline the energy rule cannot run (no audio), so "kept" here is the upper
bound of what a real run keeps.

    ~/mlsubgen/.venv/bin/python ~/mlsubgen/tools/gate_report.py ~/mlsubgen/work/*.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlsubgen import config                                   # noqa: E402
from mlsubgen.asr import repair_timestamps, words_from_dicts    # noqa: E402
from mlsubgen.clean import filter_cues                          # noqa: E402
from mlsubgen.segment import build_cues                         # noqa: E402
from mlsubgen.vad import Span, speech_ratio                     # noqa: E402

tot = {"words": 0, "zero_before": 0, "zero_after": 0, "cues_before": 0, "kept_before": 0, "cues_after": 0, "kept_after": 0, "quiet": 0}
kept_quiet: list[str] = []
dropped_after: list[str] = []
for path in sys.argv[1:]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    asr = data.get("asr") or {}
    vad = data.get("vad")
    if not asr or not vad:
        continue
    key = list(asr)[-1]
    words = words_from_dicts(asr[key]["words"])
    spans = [Span(s, e) for s, e in vad["spans"]]
    name = Path(data.get("video", path)).name
    # before: cues from the raw words, the old rule (VAD only)
    cues0 = build_cues(words)
    kept0 = [c for c in cues0 if speech_ratio(spans, c.start, c.end) >= config.MIN_SPEECH_RATIO]
    # after: repair per chunk, then the new gate
    chunks = sorted(asr[key].get("chunks", []), key=lambda c: c["start"])
    fixed_words = []
    fixed_n = 0
    if chunks:
        for ch in chunks:
            part = [w for w in words if ch["start"] - 0.01 <= w.start <= ch["end"] + 0.01]
            part, n = repair_timestamps(part, ch["start"], ch["end"])
            fixed_words += part; fixed_n += n
        seen = len(fixed_words)
        if seen < len(words):                      # words outside every chunk (rounding): keep them as they are
            fixed_words += [w for w in words if not any(c["start"] - 0.01 <= w.start <= c["end"] + 0.01 for c in chunks)]
        fixed_words.sort(key=lambda w: w.start)
    else:
        fixed_words, fixed_n = repair_timestamps(words, words[0].start if words else 0, words[-1].end if words else 0)
    cues1 = build_cues(fixed_words)
    kept1, stats = filter_cues([c for c in cues1], spans, None)
    zero0 = sum(1 for w in words if w.end - w.start <= 0.001)
    zero1 = sum(1 for w in fixed_words if w.end - w.start <= 0.001)
    tot["words"] += len(words); tot["zero_before"] += zero0; tot["zero_after"] += zero1
    tot["cues_before"] += len(cues0); tot["kept_before"] += len(kept0); tot["cues_after"] += len(cues1); tot["kept_after"] += len(kept1)
    tot["quiet"] += stats["kept_quiet"]
    print(f"{name[:52]:<52} words {len(words):5d} zero {zero0:5d}→{zero1:3d} re-timed {fixed_n:5d} │ cues kept {len(kept0):4d}/{len(cues0):4d} → "
          f"{len(kept1):4d}/{len(cues1):4d}  (quiet kept {stats['kept_quiet']:3d}, dropped {stats['dropped_no_speech']:3d})")
    kept_quiet += [c.ja for c in kept1 if c.flags and "quiet" in c.flags]
    keep_ids = {id(c) for c in kept1}
    dropped_after += [c.ja for c in cues1 if id(c) not in keep_ids]

w = tot["words"] or 1
print(f"\nTOTAL words {tot['words']}: zero-length {tot['zero_before']} ({tot['zero_before'] / w * 100:.0f}%) → {tot['zero_after']}")
print(f"cues kept: before {tot['kept_before']}/{tot['cues_before']} ({tot['kept_before'] / max(tot['cues_before'], 1) * 100:.0f}%) → "
      f"after {tot['kept_after']}/{tot['cues_after']} ({tot['kept_after'] / max(tot['cues_after'], 1) * 100:.0f}%), of which kept as quiet lines: {tot['quiet']}")
random.seed(1)
print("\nquiet lines the new gate KEEPS (sample) — these should read as speech:")
for t in random.sample(kept_quiet, min(25, len(kept_quiet))):
    print("  ", t[:70])
print("\nwhat it still DROPS (sample) — these should read as moans, fillers, junk:")
for t in random.sample(dropped_after, min(25, len(dropped_after))):
    print("  ", t[:70])
