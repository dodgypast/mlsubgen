# mlsubgen — subtitles in 44 languages for your videos, entirely on your own machine

```
cd /some/folder/of/videos
mlsubgen
```

Every video in the folder and below it gets one `.srt` per subtitle language you asked for (`<video>.en.srt`,
`<video>.th.srt` …). Your media and its subtitle text never leave the machine: the speech is transcribed by local
models and translated by a local LLM. The only files written beside your videos are the `.srt` files; mlsubgen's
own state lives under its own directory, and the only network traffic is the one-time download of the models —
unless you turn on the optional title lookup (`--web-context auto`), which sends the film's title and a search
term to Wikipedia and, where you configure them, your own SearXNG and the Brave API, and nothing else.

**mlsubgen** = *multi-language* + *machine-learning* subtitle generator. It began as a Japanese→English tool for
one collection and grew into a general one. Linux and NVIDIA only. Apache-2.0. Written with the help of Claude
(Anthropic's model), reviewed and run in daily use by one person; a hobby project, not a product.

## The idea

**Use the best evidence already in the file before inferring anything.** For each subtitle language you ask for:

```
a text subtitle track in that language           → used as is; nothing written
a bitmap (PGS) track in that language            → OCR'd into the .srt — the real subtitles
a text track in the spoken language              → the transcript; translated, no listening
a bitmap track in the spoken language            → OCR'd, then the transcript; translated
nothing to read                                  → listen: language detection, two speech recognisers, translation
```

Every step has a gate that drops evidence which is too thin or too muddled and falls through to the next, and
`mlsubgen why FILE` tells you afterwards which path a subtitle file came by.

When it has to listen, it does three things the usual whisper wrapper does not: the language is decided **per
stretch of speech**, not per file, so a film that switches languages mid-scene works; **two speech recognisers**
(Qwen3-ASR with its forced aligner, faster-whisper large-v3) decode every chunk and a local LLM reconciles them
where they disagree; and the translation keeps a film's **names consistent** and its **characters in character**
(who speaks, their gender, how they address each other — worked out once per film and given to every window).
Every one of those claims has a measurement in this README, with its material and its reference named.

## Install

**Docker** (needs the NVIDIA Container Toolkit and [Ollama](https://ollama.com) on the host):

```
git clone https://github.com/dodgypast/mlsubgen.git && cd mlsubgen
cp .env.example .env            # MEDIA_DIR = the folder with your videos; default languages; your uid/gid
mkdir -p data models            # state and the model cache, created by you so the containers can write them
docker compose up -d --build    # the image is ~14 GB (CUDA torch and the ASR stacks), 10–20 minutes the first time
docker compose run --rm worker pull   # the ASR models (~8 GB) and the profile's translators (into Ollama)
```

Then `http://<host>:8790`. The **Models** panel shows what is ready and pulls anything missing with a click. The
containers use host networking (Ollama at `127.0.0.1:11434`); the web UI has no login unless `MLSUBGEN_WEB_AUTH`
is set — see *Running it as a service*. Verified on 2026-10-03 from a fresh clone, as written.

**On the host** (Python 3.12, [uv](https://docs.astral.sh/uv/), `ffmpeg`):

```
git clone https://github.com/dodgypast/mlsubgen.git ~/mlsubgen && cd ~/mlsubgen
bash setup.sh                   # venv, CUDA torch, pinned deps, selftest, the `mlsubgen` command, the two user units
```

`setup.sh` ends with the remaining steps: the two unit files (`MLSUBGEN_MEDIA_ROOTS`, `MLSUBGEN_TARGETS`),
`mlsubgen pull`, then `systemctl --user enable --now mlsubgen-worker mlsubgen-web`.

## Hardware

| | needed | notes |
|---|---|---|
| **GPU** | NVIDIA, **8 GB** at least; **24 GB** for the full-quality setup | CUDA only. The card never holds the speech engines and the translator at once (a run transcribes a round of files, frees the card, then translates), so it needs to hold only the larger stage. Developed and measured on an RTX A5000 (24 GB) |
| **RAM** | **16 GB** minimum, **32 GB** recommended | models stream through RAM; four tesseract processes during OCR; on a small card part of the translator runs from RAM |
| **CPU** | 4 cores or more | diarization, bitmap decoding and tesseract run on the CPU |
| **Disk** | **30–85 GB** by profile | image ~14 GB (or the venv ~6 GB) · ASR models ~8 GB · translators in Ollama: `full` ≈ 54 GB (three models), `16gb`/`12gb`/`8gb` 18 GB, `12gb-dense` 7 GB, `8gb-dense` 6 GB · speaker models 33 MB · tesseract packs ~0.5 GB · work files and OCR cache a few MB per video; the temporary wav (~120 MB per hour of video) is deleted after use |
| **OS / tools** | Linux; `ffmpeg`/`ffprobe`; Ollama on the same host | tested on Arch-family (CachyOS) and Debian-family systems |

The card's memory picks a **profile** at start (`mlsubgen models` shows it; `MLSUBGEN_PROFILE=12gb` or `--profile`
forces one; `mlsubgen pull` downloads what the active profile needs).

| | `full` — 20 GB and up | `16gb` — 15 to 20 GB | `12gb` — 11 to 15 GB | `8gb` — under 11 GB |
|---|---|---|---|---|
| **cards, for example** | RTX 3090 / 4090 / 5090, A5000 / A6000, L4 | RTX 4080, 4070 Ti Super, 4060 Ti 16 GB, 5070 Ti, A4000 | RTX 3060 12 GB, 4070, 3080 12 GB, 5070 | RTX 3050 / 3070 / 4060 (8 GB), 2070 / 2080 |
| **system RAM** | 16 GB | 16 GB | 24 GB (else `12gb-dense`) | 32 GB (else `8gb-dense`) |
| **speech recognition** | both engines resident, whisper float16 (≈ 10 GB) | the same | both resident, whisper int8 (≈ 8 GB) | one engine on the card at a time; the same two engines, slower per file |
| **translation** | `gemma4:31b` by default, `qwen3.8:27b` for Japanese → English, `translategemma:27b` for six languages (see *Languages*); all fully on the card (the test clip: 103 s) | `gemma4:26b`, 22 of 30 layers on the card (≈ 13 GB): the clip in 68 s, measured; `translategemma:12b` whole for eight languages | `gemma4:26b`, 14 layers (≈ 9 GB): ~50 tok/s, the clip in roughly 100 s; `translategemma:12b` for eight languages | `gemma4:26b`, 6 layers (≈ 5 GB): ~40 tok/s, the clip in roughly 2 min; `translategemma:4b` for seven languages |
| **`-dense` fallback** (RAM short) | — | — | `gemma4:12b` (7.2 GB): chrF++ 29.0 | `gemma4:e4b` (6.1 GB): weaker again |
| **bitmap subtitles, Thai / CJK / stacked scripts** (vision model) | the 31B | the 26B — matched the 31B on the Thai benchmark | the 26B (`12gb-dense`: the 12B, weaker) | the 26B, slower (`8gb-dense`: **no** — the E4B cannot read them; those tracks are left alone and the audio transcribed) |
| everything else — per-stretch detection, embedded text tracks, tesseract OCR, diarization, terms, characters, benches | yes | yes | yes | yes |

The profiles below 24 GB share one trick, measured on 2026-10-03: the translator is Gemma 4's **26B
mixture-of-experts** with only as many of its layers on the card as fit, the rest run from system RAM by Ollama;
with ~4B parameters active per token it stays fast mostly off the card — 84 tok/s with 24 layers resident, 51
with 16, 40 with 8 — and translates almost as well as the 31B (chrF++ 30.2 against 31.0 on the same clip and
reference) and read Thai bitmaps as well (76 % / 93.8 against 77 % / 93.7). The speeds are from a 24 GB card
limited to those splits, so a real 16 GB card with the same split should land close, a slower CPU and RAM lower.
Only `full` has run on its own hardware for weeks; the TranslateGemma routes apply to `full` only until it is
measured on the smaller profiles. **Reports from real 8, 12 and 16 GB cards are the most useful issue this
repository can receive.**

## Using it

```
mlsubgen                         # every video below here → the default languages; queued for the worker
mlsubgen --target en,th,de .     # three subtitle files per video
mlsubgen --source ja FOLDER      # skip the language detector: the audio is Japanese
mlsubgen --overwrite FILE        # redo one file from scratch
mlsubgen --speakers auto FOLDER  # with speaker diarization (mlsubgen pull speakers once);  --speakers 3  when you know the count
mlsubgen languages               # the 44 offered codes, native names, which engine decodes each, which translator each goes to
mlsubgen config targets th,de    # save the default languages;  config  shows all settings
mlsubgen models                  # what is ready: translators in Ollama, ASR models in the cache
mlsubgen pull                    # download what the profile needs;  pull gemma4 · pull some/tag:latest · pull --all
mlsubgen jobs                    # the queue;  mlsubgen log ID · pause ID · resume ID · cancel ID · retry ID
mlsubgen why FILE                # where each subtitle file came from: track or engines, detection, speakers, terms, characters, translator
mlsubgen tracks FILE             # the audio and subtitle tracks, with which bitmap tracks OCR can read
mlsubgen help                    # the one-screen guide;  mlsubgen help run  for every option
```

While the worker service is up, `mlsubgen` in a folder queues the job and returns; `mlsubgen --now …` runs in the
foreground. `--context "NHK documentary about the Tōhoku coast; presenter Tanaka Yūki"` biases the ASR and the
translation; `--glossary names.tsv` (`source<TAB>target` per line) pins names; `--genre "a slapstick family anime"`
sets the register. `--terms off`, `--register off`, `--ocr off` switch the terminology pass, the character sheet
and bitmap OCR off.

**The web UI** has a new-job form (folder picker confined to your media roots, tick boxes for the languages, the
spoken language, translator, ASR mode, speakers, embedded-subtitle policy, context, glossary), a **Preview** that
lists exactly which files a job would touch, the queue with pause / resume / cancel / retry, a job page with
per-file outcomes and the live log, and a page of skipped files that can be re-queued with the language forced.
Jobs survive reboots: every finished stage and every decoded ASR chunk is checkpointed.

**What a run leaves behind:** the `.srt` files and `logs/skipped.log`. The temporary wav goes after each file's
ASR; the per-file work file (detection, ASR words, cues, translations) once its last `.srt` is written. An
interrupted run resumes from the work files; a skipped file remembers its verdict (`--source`, `--overwrite` or
`--audio-track` retries it); `--keep-work` keeps the ASR cache to re-translate with another model.

## How it works

| stage | what |
|---|---|
| embedded subs | a text track in a target language → that target is done; any other full text track (the spoken language's first; ASS cleaned of tags, karaoke and comments; an untagged track has its language read from its own words) → the transcript, no ASR. A **bitmap** (PGS) track is read through OCR — below |
| probe · audio | `ffprobe` picks the audio track (tag, title, default; `--audio-track N` overrides); `ffmpeg` → 16 kHz mono wav |
| speakers | *optional, `--speakers`*: speaker turns from sherpa-onnx on the CPU — below. **Labels** (0.5.4): when the transcript came from a text track and a target needs to know who speaks, the diarizer runs on the audio anyway (nothing is transcribed) and each cue takes the voice that covers it — a fact for the character sheet, which then says which voice is which character |
| lookup | *opt-in, `--web-context auto`* (0.5.8): the title is identified from the file name and folders and looked up — Wikipedia first (summary, cast as "actor as character", the title in every target language), then your SearXNG, then the Brave API under a daily cap when SearXNG found too little; confirmed as a film or series, cached per title, every query recorded. The sheet gets the cast and relationships as priors, the glossary the localised title. Only the title and a search term leave the machine |
| language ID | per ~10 s of speech, or per speaker turn: whisper's probability + Qwen's decode + the words' script and function words all have to agree; switch points refined to the exact span; a short run of another language needs strong evidence |
| chunks · ASR | ≤ 30 s, one language each, covering the whole timeline (only the noise floor is skipped); both engines decode every chunk with its language forced; checkpointed per chunk |
| merge | the translator LLM reconciles the two transcripts where they differ; chunks that agree need no LLM. When a target is the spoken language the audio is the source even if a foreign text track exists (an English film with only Italian subtitles is transcribed, not back-translated), and that track becomes **evidence**: its lines for the disputed seconds are shown to the reconciler, never output (0.5.6) |
| cues | sentence ends, pauses, speaker changes, length limits, hallucination filters |
| terms | once per film, per target: the recurring names and terms (a script heuristic plus one LLM pass) rendered once — standard transliteration for names, the established form for titles — into the glossary every window reads; your `--glossary` wins |
| characters | once per film: a chat model reads the transcript and lists who speaks — gender, age, role, who is whose parent, spouse, boss, friend, with *unknown* where the dialogue does not say — then, per target, turns that into the rules of address for that language: how each character refers to themselves, how they address each of the others (pronoun, kin term, title, politeness level, particles), and the grammatical gender of their own speech. Every window gets the sheet, so a father does not answer in the feminine, a ten-year-old does not call her father *vous*, and a Thai child stays หนู to her mother from the first scene to the last |
| translate | per target: cues already in the target copied through; the rest in windows of 20 with context, glossary, characters, per-language register rules, speaker continuity, retry and per-line fallback; the **foreign-script guard** sends back any line with letters of a script that is neither the target's nor Latin (names and brands stay) |
| repair | the **register repair** (0.5.5): a line that hedges a form with a slash (*měl/a*, *ค่ะ/ครับ*, *he/she*) is unusable on screen and detectable in any script, so only those lines go to a checker of the other model family — Qwen for a Gemma translation — with the source line and the sheet, to choose one form and change nothing else; counts in `why` |
| typeset | ≤ 2 lines, per-language line width and reading speed, minimum duration and gaps, cluster-safe breaks for Thai, Lao, Khmer, Burmese and Devanagari → `<video>.<lang>.srt` |

**Bitmap subtitles.** Blu-ray remuxes carry their subtitles as PGS bitmaps, often a dozen languages and no text
track. mlsubgen decodes the stream itself and hands each image to the engine measured best for the script:
tesseract for Latin, Greek and Cyrillic (CPU, four images at a time; the Docker image ships 19 language packs),
the profile's vision model through Ollama for Thai, Chinese, Japanese, Korean and the stacked scripts (one image a
second on the GPU, unloaded before the ASR engines load). A gate refuses a track whose text is in the wrong
script, mostly symbols, repeated, full of replacement characters or digit-and-symbol lines; a refused track is
left alone and the audio transcribed. Results are cached, so a track is OCR'd once; `mlsubgen ocr VIDEO --track N`
runs it by hand. Measured results are in *How good is it*.

**Speakers.** `--speakers auto` (or `N` when you know the cast size) runs pyannote's `segmentation-3.0` and a
3D-Speaker embedding through sherpa-onnx on the CPU (33 MB, pulled from sherpa's GitHub releases — no Hugging Face
token) before anything listens to the words. The turns cut the language-detection windows at speaker changes, so a
character who switches language mid-scene is followed and a bilingual voice is learnt as bilingual; after ASR they
close cues and give the translator an anonymous tag per line (`[S2]`: "same voice / different voice", nothing
about who). A file whose diarization gives no usable structure — one voice, or a cast fragmented into dozens — is
processed as without the option. Off by default: on feature films the clustering tends to fragment a cast.

## Languages and translators

45 languages are known: every one is a **source** (a subtitle track in it is read; Qwen3-ASR decodes the languages
its aligner covers — Japanese, Chinese, Cantonese, Korean, English, French, German, Italian, Portuguese, Russian,
Spanish — and whisper the rest), and **44 are offered as targets**. Which translator a target goes to is a
measured choice per language, and `mlsubgen languages` shows it for the active profile:

| profile | default translator | Japanese → English | the invented-word languages go to | withheld |
|---|---|---|---|---|
| `full` | `gemma4:31b-it-qat` (19 GB), with the character sheet and the per-language register rules | `qwen3.8:27b` (18 GB) | `translategemma:27b` (17 GB): Hungarian, Lithuanian, Latvian, Estonian, Catalan, Finnish | Greek |
| `16gb`, `12gb` | `gemma4:26b` (18 GB, MoE, part on the card) | the 26B | `translategemma:12b` (8.1 GB, whole on the card): the six above plus Slovak and Slovenian | Greek |
| `8gb` | `gemma4:26b` (6 layers on the card) | the 26B | `translategemma:4b` (3.3 GB): Hungarian, Lithuanian, Latvian, Estonian, Catalan, Finnish, Slovak | Greek |
| `12gb-dense`, `8gb-dense` | `gemma4:12b` / `gemma4:e4b` | the same | the same 12B / 4B routes (unmeasured on these) | Greek |

The routes come from one film in 44 languages and five translators, scored on 2026-10-04 and 05 by
dictionary-unknown words, script purity and reading. On the routed languages the TranslateGemmas' unknown-word
rates are a quarter to a tenth of the Gemma 4s' (Latvian: 31B 5.6 %, 26B 4.8, TranslateGemma 27B 0.7, 12B 1.7,
4B 3.4; Hungarian 3.5 / 2.1 / 0.4 / 0.8 / 1.1), the 12B keeps nearly all of the 27B's advantage, and the 4B keeps
it for Hungarian, Catalan and Slovak. TranslateGemma is not the default for the rest because it has faults of its
own: it defaults to formal address (a child saying *vous* to her father), hedges gender with slashes where the
speaker is unknown (the 12B hedges every Thai line with *ค่ะ/ครับ*, so no TranslateGemma ever sees Thai), and its
fixed prompt cannot take the character sheet — **so the routed languages get its words without the sheet's rules
of address.** A narrow repair pass that touches only the lines a validator flags is the next step. The 27B on a
16 GB split took 37 minutes a language, which is why the small profiles get the 12B whole rather than the 27B
in part. Greek is withheld everywhere because the choice there is between the Gemmas' grammar errors and
TranslateGemma's gender slashes; it returns when the repair pass is measured on it.

How the 44 read, from one episode translated into all of them on 2026-10-03 and the same scene read in each by a
competent reader rather than a native speaker (English is a target but not in the tiers: it was the source):

- **Read well (29):** Japanese, Cantonese, Korean, Thai, French, German, Spanish, Italian, Portuguese, Russian,
  Indonesian, Vietnamese, Turkish, Dutch, Polish, Czech, Swedish, Danish, Finnish, Norwegian, Romanian, Ukrainian,
  Malay, Filipino, Persian, Arabic, Hindi, Bengali, Tamil — natural register for the material, idioms where the
  language has them, names consistent.
- **Usable, with known errors (7):** Chinese, Hebrew, Bulgarian, Croatian, Catalan, Slovak, Slovenian — a wrong
  or invented word every ten to twenty lines. Offered, because something is better than nothing when the limit
  is stated; `bench` on your own material before a long run.
- **Returned (7):** Hungarian, Lithuanian, Latvian, Estonian on TranslateGemma; Khmer, Lao and Burmese on the 31B
  once the foreign-script guard existed — their fault was other scripts' letters leaking in, and two independent
  reviewers found the Burmese otherwise usable (meaning fidelity good, specialised nouns weak, register
  inconsistent — the last is what the character sheet is for).

The tiers are this translator on this material, not a verdict on the languages; the same measurement is one run
away on any file. Other presets (`mlsubgen models` shows which are pulled): `qwen3-30b` (a fast MoE fallback),
`gemma4-26b`, `gemma4-12b`, `gemma4-e4b` (the small profiles'). `-t NAME` forces one preset for every pair;
`--model TAG` any Ollama model; `--backend openai --url … --model …` any OpenAI-compatible server.

## How good is it

Every number names its material and its reference, and the command that produced it. They are enough to choose
between alternatives on the same material, which is what they were used for; a different film can rank the
alternatives differently, which is why the benches exist — run the same comparison on yours.

**Against the film's own human tracks** (`mlsubgen refscore VIDEO --out DIR`, 0.5.7): a release that carries
thirty text tracks is thirty references. The generated file for each language is scored against the human track,
chrF++ per minute of film so that distributors cutting the same dialogue into different cues is not punished,
WER for a same-language pair (English generated from English audio against the English track), and coverage.
This is how the register work, the labels and the routes are measured from here on, across every language a
reference exists for, rather than by one reader's sample.

**Language detection** (`mlsubgen lidbench VIDEO`, scored against the film's forced subtitle track — a lower bound,
since songs and untranslated lines are foreign speech it does not show):

| film | stretch recall | precision | switch recall | latency |
|---|---|---|---|---|
| Babel | 90 % | 80 % | 66 % | 0.3 s |
| Inglourious Basterds | 97 % | 75 % | 57 % | 0.4 s |
| Only God Forgives | 83 % | 92 % | 61 % | 1.1 s |

**Transcription and translation** (`mlsubgen bench VIDEO --clip … --reference REF.srt`, chrF++ against a fansub
of a Shin-chan episode, Japanese audio → Thai): the 31B 31.0, the 26B 30.2, the 12B 29.0; the terminology pass
took a recurring noun from one consistent line in three to three in three. `--asr qwen`, `--asr whisper` and the
dual default on the same clip show what the second recogniser is worth on your material.

**Bitmap OCR** (`mlsubgen ocrbench VIDEO`, against a text track of the same film, exact cues / chrF):

| script | engine | result |
|---|---|---|
| English Blu-ray, 1,800 cues | tesseract | **95 % / 99.4** |
| Thai | vision 31B / 26B / 12B / E4B | **77 % / 93.7** · 76 % / 93.8 · 65 % / 89.3 · 18 % / 66.5 (not used) |
| Chinese Simplified | vision 31B · tesseract | **79 % / 90.0** · 41 % / 77.6 |
| Japanese Blu-ray SDH, against another distributor's transcript | vision 31B | 51 % identical cues / chrF 78 — paired lines match to the character; the gap is cue cutting and furigana the model correctly left out |

Reference traps found on the way: "Hybrid" releases pair bitmaps and text from different translations; a streaming
SRT muxed beside a Blu-ray can be out of sync with the disc's cut; distributors cut the same dialogue into cues
differently (so a per-minute content score ignores cue boundaries).

**Questions a sceptical reader should ask.** *Why two recognisers and an LLM?* Because they fail differently — one
goes silent over music, the other hallucinates stock phrases — and where they agree, most chunks, no LLM is
involved. *Does the vision OCR make things up?* It can; the strict prompt, the gate, the measurements above and
`why` stand against it, and a two-engine disagreement check is the obvious next step and not built. *Do the small
profiles work on real cards?* Measured on a 24 GB card limited to their splits: memory and quality fixed, speed
not. *What stays local?* Everything except the one-time model downloads; `HF_HUB_OFFLINE=1` after that.

## Running it as a service

- **Login.** None by default. `MLSUBGEN_WEB_AUTH=user:password` puts HTTP Basic on every page and API route; a
  reverse proxy with its own login is the answer for anything reachable beyond a LAN or VPN.
- **Networking.** `network_mode: host`, because Ollama is on the same host at `127.0.0.1:11434`; the web UI
  listens on 8790. To run bridged, drop the host mode, publish the port, and point `MLSUBGEN_LLM_URL` at the
  host's Ollama (`compose.yaml` has the lines). Nothing listens but the web UI.
- **What it writes.** Beside your videos: the `.srt` files only. Under `./data`: the queue (`mlsubgen.db`), job
  logs, work files, the OCR cache, temporary audio, glossaries and `settings.json` — back up the last two, the rest
  is cache. Under `./models`: the Hugging Face cache and the speaker models. Translators live in Ollama's store.
- **Idle footprint.** Nothing on the card between jobs: the speech engines are freed after each round, the vision
  model after each track, and Ollama drops the translator after its `keep_alive`.
- **Updating.** `git pull && docker compose up -d --build`; a running job is interrupted cleanly and resumes. On a
  host install, `git pull` then `systemctl --user restart mlsubgen-worker mlsubgen-web` — the services keep the
  code they imported until restarted.
- **Scheduling.** No watch folder: a nightly `mlsubgen /media` from cron queues whatever is new; files that already
  have their subtitles are skipped at the scan.
- **Media servers.** Jellyfin, Plex, Emby and Kodi pick the sidecars up on their next library scan. Not a Bazarr
  provider, no webhook; the queue has a JSON API (`POST /api/jobs` with a `path`) for anything that wants to
  start a job.
- **Monitoring.** `MLSUBGEN_METRICS_FILE` pointed at node_exporter's textfile directory makes the worker publish
  `mlsubgen_worker_busy`, job id, queue depth, up and a heartbeat, so a CPU alert can tell a transcription from a
  fault.

## Configuration

| variable | default | |
|---|---|---|
| `MLSUBGEN_HOME` | `~/mlsubgen` (`/data` in Docker) | state: work files, logs, the queue |
| `MLSUBGEN_MEDIA_ROOTS` | *(none — set it)* | colon-separated folders the worker and the picker may use |
| `MLSUBGEN_TARGETS` | `en` | default subtitle languages; `mlsubgen config targets …` or the web form's *make these the default* beats it; `--target` beats both |
| `MLSUBGEN_LLM_URL` | `http://127.0.0.1:11434` | the translator server |
| `MLSUBGEN_WEB_HOST` / `MLSUBGEN_WEB_PORT` | `0.0.0.0` / `8790` | the web UI |
| `MLSUBGEN_WEB_AUTH` | *(none)* | `user:password` — HTTP Basic on every page and API route |
| `MLSUBGEN_METRICS_FILE` | *(none)* | a `.prom` file in node_exporter's textfile directory |
| `MLSUBGEN_WEB_CONTEXT` | `off` | `auto` turns the title lookup on (opt-in; `--web-context` per run); the character sheet gets the cast, relationships and localised titles as priors, with sources recorded |
| `MLSUBGEN_SEARXNG_URL` / `BRAVE_API_KEY` / `MLSUBGEN_BRAVE_DAILY_CAP` | *(none)* / *(none)* / `100` | the lookup's second and third sources after Wikipedia; Brave only when SearXNG found too little, under the daily cap |
| `MLSUBGEN_PROFILE` | `auto` | `full`, `16gb`, `12gb`, `8gb`, `12gb-dense` or `8gb-dense` |
| `HF_HUB_OFFLINE` | `0` | `1` after the models are downloaded |

Everything else (chunk lengths, cue limits, line widths, reading speeds, hallucination patterns, the routes, the
gates' thresholds) is in `mlsubgen/config.py` and documented there.

## Known limitations

- Language detection on material that alternates languages quickly is the weakest link; `--source` forces the
  language, `mlsubgen scan` shows what the detector sees without running the ASR.
- The six TranslateGemma-routed languages get no character sheet; Greek is withheld; the tiers above are one
  reader's judgement on one episode and one film.
- The small profiles are measured on a 24 GB card limited to their splits, and TranslateGemma on them is not
  measured at all.
- Diarization fragments feature-film casts and is off by default.
- Specialised vocabulary in low-resource languages (an insect, a tree, a snake) is where the translators err
  most; a user glossary of a film's twenty key nouns is the cure.
- The web UI has HTTP Basic auth at most. NVIDIA only.

## Licence and credits

Apache-2.0 (see `LICENSE`). Built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and
Qwen3-ForcedAligner (Apache-2.0), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT),
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT), [Ollama](https://ollama.com), and for speakers
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0) with pyannote's
[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (MIT) and a
[3D-Speaker](https://github.com/modelscope/3D-Speaker) embedding model (Apache-2.0). The translator models carry
their own licences (Qwen: Apache-2.0; Gemma and TranslateGemma: Google's Gemma terms). Issues and pull requests
are welcome, and issues get answered; the reference environment is the pinned one in `requirements.txt` and the
Dockerfile.
