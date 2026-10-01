"""Where did the speech go? For a kept work file: how much VAD speech lies outside every chunk (never decoded),
how much lies inside a chunk but has no word on it (decoded away), how dense each chunk's output is, and the
biggest wordless stretches with timestamps so they can be checked by ear.

    ~/mlsubgen/.venv/bin/python ~/mlsubgen/tools/coverage_report.py ~/mlsubgen/work/_MLSUB-RAW_Crayonshinchan_1020*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlsubgen.asr import words_from_dicts   # noqa: E402


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


for path in sys.argv[1:]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    asr = data.get("asr") or {}
    vad = data.get("vad")
    if not asr or not vad:
        continue
    spans = [(float(s), float(e)) for s, e in vad["spans"]]
    speech = sum(e - s for s, e in spans)
    for key, entry in asr.items():
        words = words_from_dicts(entry["words"])
        chunks = sorted(((c["start"], c["end"]) for c in entry.get("chunks", [])), key=lambda c: c[0])
        print(f"\n{Path(data.get('video', path)).name}\n  ASR entry: {key}\n  {len(words)} words · {len(spans)} VAD spans, {speech / 60:.1f} min speech · {len(chunks)} chunks")
        # 1. speech outside every chunk
        outside = 0.0
        outside_spans = []
        for s, e in spans:
            cov = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in chunks)
            if (e - s) - cov > 0.2:
                outside += (e - s) - cov
                outside_spans.append((s, e))
        print(f"  speech outside every chunk (never decoded): {outside:.0f} s in {len(outside_spans)} spans")
        for s, e in sorted(outside_spans, key=lambda x: x[0] - x[1])[:8]:
            print(f"     {mmss(s)}–{mmss(e)}  ({e - s:.1f} s)")
        # 2. speech inside a chunk with no word overlapping it
        starts = [w.start for w in words]
        wordless = 0.0
        wordless_spans = []
        for s, e in spans:
            if not any(w.start < e and w.end > s for w in words):
                inside = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in chunks)
                if inside > 0.2:
                    wordless += inside
                    wordless_spans.append((s, e))
        print(f"  speech inside a chunk but with NO word on it (decoded away or mistimed): {wordless:.0f} s in {len(wordless_spans)} spans")
        for s, e in sorted(wordless_spans, key=lambda x: x[0] - x[1])[:12]:
            print(f"     {mmss(s)}–{mmss(e)}  ({e - s:.1f} s)")
        # 3. output density per chunk: characters per second of VAD speech inside the chunk
        rows = []
        for c in entry.get("chunks", []):
            cs, ce = c["start"], c["end"]
            sp = sum(max(0.0, min(e, ce) - max(s, cs)) for s, e in spans)
            rows.append((c["chars"] / max(sp, 0.1), cs, ce, c["chars"], sp, c.get("retimed", 0), c.get("units", 0)))
        rows.sort()
        print("  thinnest chunks (chars per second of speech in them; Japanese dialogue runs ~5–8):")
        for d, cs, ce, ch, sp, rt, un in rows[:8]:
            print(f"     {mmss(cs)}–{mmss(ce)}  {d:4.1f} chars/s  ({ch} chars over {sp:.0f} s of speech, {un} units, {rt} re-timed)")
        retimed = sum(c.get("retimed", 0) for c in entry.get("chunks", []))
        print(f"  words re-timed by the repair: {retimed} of {len(words)} ({retimed / max(len(words), 1) * 100:.0f}%)")
