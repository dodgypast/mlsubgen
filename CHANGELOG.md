# Changelog

Versions are tags on `main`. Every version was committed behind a passing selftest; the numbers quoted are in the
README's *How good is it*, with their material and reference named.

## 0.6.0 — 2026-10-07

The character work, measured and shipped. Six 25-minute cuts of three films, twelve languages, scored against the
films' own human tracks: Thai politeness particles 35 → 57 % agreement, Thai forms of address 67 → 75, French
*tu/vous* 74 → 77, Russian *ty/vy* 60 → 65, Japanese pronouns 29 → 35, Korean politeness 60 → 62, Vietnamese
77 → 79, Czech 57 → 59; chrF++ unchanged in every language; Polish and Greek down by eight and four, with a
calibration added and unmeasured. Greek returns as a target (chrF++ 49, between Russian and Vietnamese); nothing
is withheld. `regscore` (register agreement per feature) and `refscore` (chrF++ per minute, WER, coverage) score
generated files against a release's human tracks. The renderer of the character sheet is told the informal/formal
mapping and how kin-term languages choose a pair's terms. The burn-ins' fixes: `--speakers labels` accepted by the
command line, an ad-hoc `--model` no longer a KeyError, `why` answering after a finished run from a condensed work
file, an unwritable folder or state directory refused in one sentence rather than a traceback.

## 0.5.x — 2026-10-04 to 10-06

- **Routes per language and profile.** TranslateGemma 27B for Hungarian, Lithuanian, Latvian, Estonian, Catalan
  and Finnish on the full profile; the 12B for those plus Slovak and Slovenian on 16 and 12 GB cards; the 4B on
  8 GB; the 26B MoE the default on every small profile. From one film in 44 languages and five translators.
- **The character sheet** (`--register`): who speaks, their gender, age, relationships with evidence, rendered per
  target into rules of address and given to every window. **Speaker labels** for a text-track source: the
  diarizer runs on the audio with nothing transcribed, cues take their voice, the sheet maps voices to characters
  only with a quoted line of evidence. `--speakers off | labels | auto | N`.
- **The foreign-script guard** (a leaking line goes back through the fallback) and **the hedge repair** (a
  slash-hedged form goes to a checker of the other model family to choose one).
- **Listen when the target is the spoken language**; a foreign text track becomes the reconciler's evidence
  (68–69 → 69–71 chrF++ against the human English on three cuts; translating the track back scores 51–66).
- **The title lookup** (`--web-context auto`, opt-in): Wikipedia, then your SearXNG, then Brave under a daily cap;
  cast, relationships and localised titles as priors; every query recorded. The sheet is markedly better with it.
- A hardware profile per job (web form and `--profile`); the languages a profile offers; the worker waits for free
  GPU memory and an optional lease hook; model digests pinned in every translation record; the job form redone.
- Cross-track evidence (`--cross-evidence`), measured with no gain, off by default.

## 0.5.0 — 2026-10-02

Bitmap (PGS) subtitle tracks read through OCR: an own PGS decoder, tesseract for alphabets, the profile's vision
model for Thai, Chinese, Japanese and Korean; a gate that refuses unreadable output; `ocr`, `ocrbench`. The
terminology pass (names rendered once per target). `why`: where each subtitle file came from.

## 0.4.x — 2026-10-01 to 10-02

Speaker diarization (sherpa-onnx, CPU); language detection v3 with function-word evidence and speaker-aware
windows (Babel 76 → 90 % recall); untagged text tracks read for their language; CAM++ embeddings.

## 0.3.3 — 2026-10-01

First public release: 45 languages, the queue and worker, the web UI, per-stretch language identification, two
speech recognisers with an LLM reconciling them, hardware profiles, `pull`, Docker.
