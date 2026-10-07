"""Score generated subtitles against a film's own human tracks (0.5.7, 2026-10-05).

A release that carries thirty text tracks is thirty references. For each language L that the file has as a text
track and that also has a generated `<stem>.L.srt` in the output folder, the human track is extracted (cleaned as
the pipeline cleans any track: tags, caption remnants) and compared with the generated file:

  chrF++ per minute of film — the two subtitle sets are joined per 60-second bin and scored bin by bin, so that
      a distributor cutting the same dialogue into different cues (which every release does) is not punished;
      the figure is the mean over bins with text on both sides;
  WER — for a same-language case (English subtitles generated from English audio against the English track):
      word error rate over the same bins, normalised (lower-case, punctuation stripped);
  coverage — the share of reference bins the generated set has text for, and the reverse, so a set that goes
      silent for a scene is visible beside its score.

    mlsubgen refscore VIDEO --out DIR [--lang th,ko] [--bin 60] [--hide-source en]

`--hide-source` names a track that the generation was NOT allowed to read (the English track hidden for an
English-from-Italian run), so it can be used as the reference without question.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from . import config
from .probe import probe
from .srt import read_srt
from .subs import code_for_tag, cues_from_track, extract_track

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _bins(cues, size: float) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for c in cues:
        text = getattr(c, "text", None)
        if text is None:
            text = getattr(c, "ja", "")
        if not text or not text.strip():
            continue
        mid = (c.start + c.end) / 2
        out.setdefault(int(mid // size), []).append(text.replace("\n", " ").strip())
    return out


def _norm_words(s: str) -> list[str]:
    return _PUNCT.sub(" ", s.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate by Levenshtein over tokens; 0 for two empty lists."""
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (0 if ref[i - 1] == hyp[j - 1] else 1))
            prev = cur
    return d[len(hyp)] / len(ref)


def score_pairs(ref_cues, hyp_cues, lang: str) -> dict:
    """Cue-level scoring (0.6.2): each reference cue is paired with the generated cue that overlaps it most, and
    chrF++ and WER are computed over the paired texts only. For a forced track, which carries only a film's
    foreign-language lines, the per-minute bins are diluted by the generated cues for the English lines in the
    same minute; pairing scores only what the reference has."""
    from sacrebleu.metrics import CHRF
    from .register import match_cues
    chrf = CHRF(word_order=2)
    pairs = match_cues(ref_cues, hyp_cues)
    def txt(c):
        t = getattr(c, "text", None)
        return t if t is not None else getattr(c, "ja", "")
    refs = [txt(r).replace("\n", " ") for r, _ in pairs]; hyps = [txt(h).replace("\n", " ") for _, h in pairs]
    score = chrf.corpus_score(hyps, [refs]).score if pairs else None
    w = [wer(_norm_words(r), _norm_words(h)) for r, h in zip(refs, hyps)]
    return {"lang": lang, "pairs": len(pairs), "ref_cues": len(ref_cues), "hyp_cues": len(hyp_cues),
            "chrf": round(score, 1) if score is not None else None, "wer": round(100 * sum(w) / len(w), 1) if w else None,
            "coverage": round(len(pairs) / len(ref_cues), 2) if ref_cues else None, "bins_ref": len(ref_cues), "bins_hyp": len(hyp_cues), "bins_both": len(pairs)}


def score_pair(ref_cues, hyp_cues, lang: str, size: float = 60.0) -> dict:
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)
    rb, hb = _bins(ref_cues, size), _bins(hyp_cues, size)
    both = sorted(set(rb) & set(hb))
    scores, wers = [], []
    for b in both:
        r, h = " ".join(rb[b]), " ".join(hb[b])
        scores.append(chrf.sentence_score(h, [r]).score)
        wers.append(wer(_norm_words(r), _norm_words(h)))
    return {"lang": lang, "bins_ref": len(rb), "bins_hyp": len(hb), "bins_both": len(both),
            "chrf": round(sum(scores) / len(scores), 1) if scores else None,
            "wer": round(100 * sum(wers) / len(wers), 1) if wers else None,
            "coverage": round(len(both) / len(rb), 2) if rb else None,
            "ref_cues": len(ref_cues), "hyp_cues": len(hyp_cues)}


def reference_tracks(subs, langs: list[str] | None, include_forced: bool = False, assume_und: str | None = None) -> list[tuple]:
    """(track, code) for the text tracks usable as references. SDH and commentary tracks never; forced tracks only
    with include_forced (a forced track is the human translation of a film's foreign-language stretches, the
    reference for foreign audio → English); an untagged track counts as assume_und when given (a fansub's `und`
    ASS dialogue track)."""
    out = []
    for t in subs:
        if not t.is_text:
            continue
        title = t.title or ""
        if re.search(r"sdh|commentary|signs", title, re.I):
            continue
        is_forced = bool(getattr(t, "forced", False)) or bool(re.search(r"forced", title, re.I))
        if is_forced and not include_forced:
            continue
        code = code_for_tag(t.language)
        if not code and assume_und and (t.language or "und") in ("und", "", "mis", "zxx"):
            code = assume_und
        if not code:
            continue
        if langs and code not in langs:
            continue
        out.append((t, code))
    return out


def refscore(video: Path, out_dir: Path, langs: list[str] | None = None, size: float = 60.0, tmp: Path | None = None,
             include_forced: bool = False, assume_und: str | None = None, pairs: bool = False) -> list[dict]:
    pr = probe(video)
    tmp = tmp or out_dir
    rows = []
    for t, code in reference_tracks(pr.subs, langs, include_forced, assume_und):
        hyp = out_dir / f"{video.stem}.{code}.srt"
        if not hyp.is_file():
            continue
        tmp_srt = tmp / f"{video.stem}.ref.{code}.s{t.index}.srt"
        try:
            extract_track(video, t.index, tmp_srt, t.codec)
            ref = cues_from_track(tmp_srt, code)
        finally:
            tmp_srt.unlink(missing_ok=True)
        if len(ref) < 20:
            continue
        row = score_pairs(ref, read_srt(hyp), code) if pairs else score_pair(ref, read_srt(hyp), code, size)
        row["track"] = t.index
        rows.append(row)
    return rows


def print_table(rows: list[dict], title: str = "") -> None:
    if title:
        print(title)
    print(f"   {'lang':<5} {'language':<22} {'chrF++':>7} {'WER%':>6} {'cover':>6} {'ref':>5} {'hyp':>5} {'bins':>5}")
    for r in sorted(rows, key=lambda x: -(x['chrf'] or 0)):
        print(f"   {r['lang']:<5} {config.LANG_NAMES.get(r['lang'], r['lang']):<22} {str(r['chrf'] or '-'):>7} {str(r['wer'] or '-'):>6} "
              f"{str(r['coverage'] or '-'):>6} {r['ref_cues']:>5} {r['hyp_cues']:>5} {r['bins_both']:>5}")
