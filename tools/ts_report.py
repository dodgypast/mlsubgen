"""Shape of the aligner's timestamp failures: runs of words stamped with the same time, zero-length words,
non-monotonic words — with examples showing the good words on either side of a broken run.

    ~/mlsubgen/.venv/bin/python ~/mlsubgen/tools/ts_report.py ~/mlsubgen/work/MKMP-429*.json ~/mlsubgen/work/_MLSUB-RAW_Crayonshinchan_1051*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlsubgen.asr import words_from_dicts   # noqa: E402

for path in sys.argv[1:]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    asr = data.get("asr") or {}
    if not asr:
        continue
    key = list(asr)[-1]
    words = words_from_dicts(asr[key]["words"])
    zero = [w for w in words if w.end - w.start <= 0.001]
    runs: list[tuple[int, int]] = []          # (first index, length) of runs sharing start and end
    i = 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and abs(words[j + 1].start - words[i].start) < 0.001 and abs(words[j + 1].end - words[i].end) < 0.001:
            j += 1
        if j > i:
            runs.append((i, j - i + 1))
        i = j + 1
    back = sum(1 for a, b in zip(words, words[1:]) if b.start < a.end - 0.5)
    chars_in_runs = sum(len(w.text) for s, n in runs for w in words[s:s + n])
    name = Path(data.get("video", path)).name
    print(f"\n{name}: {len(words)} words · zero-length {len(zero)} · same-stamp runs {len(runs)} "
          f"(longest {max((n for _, n in runs), default=0)}, {chars_in_runs} chars in them) · jumps back >0.5 s {back}")
    chunks = asr[key].get("chunks", [])
    if chunks:
        print(f"  chunks: {len(chunks)}, longest {max(c['sec'] for c in chunks):.0f} s; "
              f"chunks with a note: {sum(1 for c in chunks if c.get('note'))}")
    shown = 0
    for s, n in sorted(runs, key=lambda r: -r[1])[:3]:
        lo, hi = max(0, s - 2), min(len(words), s + n + 2)
        print(f"  run of {n} at words {s}–{s + n - 1}:")
        for k in range(lo, hi):
            w = words[k]
            tag = "RUN " if s <= k < s + n else "    "
            if s + 4 < k < s + n - 1:
                if k == s + 5:
                    print("      …")
                continue
            print(f"    {tag}{k:5d} {w.start:9.2f}–{w.end:9.2f}  {w.text}")
        shown += 1
    if zero and not runs:
        for w in zero[:5]:
            print(f"    zero {w.start:9.2f}  {w.text}")
