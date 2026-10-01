"""Bench: the same cues through several translators, side by side, with optional chrF against a reference .srt.
Everything stays local — the only judge is you (and chrF if you supply an official English subtitle file)."""
from __future__ import annotations

import html
from pathlib import Path

from .segment import Cue
from .srt import SrtCue, read_srt, ts


def align_reference(cues: list[Cue], ref: list[SrtCue]) -> list[str]:
    """For each cue, the reference text with the largest time overlap (empty when none)."""
    out = []
    j = 0
    for c in cues:
        best, best_ov = "", 0.0
        while j > 0 and ref[j - 1].end > c.start:
            j -= 1
        k = j
        while k < len(ref) and ref[k].start < c.end:
            ov = min(ref[k].end, c.end) - max(ref[k].start, c.start)
            if ov > best_ov:
                best, best_ov = ref[k].text.replace("\n", " "), ov
            k += 1
        out.append(best)
    return out


def chrf(hyps: list[str], refs: list[str]) -> float | None:
    pairs = [(h, r) for h, r in zip(hyps, refs) if r and h]
    if not pairs:
        return None
    try:
        from sacrebleu.metrics import CHRF
        metric = CHRF(word_order=2)
        return round(metric.corpus_score([h for h, _ in pairs], [[r for _, r in pairs]]).score, 1)
    except Exception:  # noqa: BLE001
        return None


def write_html(out: Path, title: str, cues: list[Cue], results: dict[str, list[Cue]], stats: dict[str, dict],
               reference: list[str] | None = None) -> None:
    names = list(results)
    head = "".join(f"<th>{html.escape(n)}<br><small>{html.escape(stats[n].get('summary', ''))}"
                   f"{'<br>chrF++ ' + str(stats[n]['chrf']) if stats[n].get('chrf') is not None else ''}</small></th>"
                   for n in names)
    rows = []
    for i, c in enumerate(cues):
        cells = "".join(f"<td>{html.escape(results[n][i].en if i < len(results[n]) else '')}</td>" for n in names)
        refcell = f"<td class=ref>{html.escape(reference[i])}</td>" if reference else ""
        rows.append(f"<tr><td class=t>{ts(c.start)[:8]}</td><td class=ja>{html.escape(c.ja)}</td>{cells}{refcell}</tr>")
    doc = f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font:14px/1.4 system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top}}th{{background:#eee;position:sticky;top:0}}
td.t{{white-space:nowrap;color:#666}}td.ja{{width:22%}}td.ref{{background:#f6fff6}}tr:nth-child(even){{background:#fafafa}}</style>
<h2>{html.escape(title)}</h2><p>{len(cues)} cues · {', '.join(html.escape(n) for n in names)}</p>
<table><tr><th>t</th><th>Japanese (ASR)</th>{head}{'<th>reference</th>' if reference else ''}</tr>{''.join(rows)}</table>"""
    out.write_text(doc, encoding="utf-8")
