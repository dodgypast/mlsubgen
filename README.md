# mlsubgen — subtitles in 45 languages for your videos, entirely on your own machine

```
cd /some/folder/of/videos
mlsubgen
```

Every video in the folder and below it gets one `.srt` per subtitle language you asked for
(`<video>.en.srt`, `<video>.th.srt` …). Your media and its subtitle text never leave the machine: the speech is
transcribed by local models and translated by a local LLM. The only files written beside your videos are the
`.srt` files; mlsubgen's own state (work files, logs, the OCR cache, temporary audio) lives under its own
directory, and the only network traffic is the one-time download of the models.

**mlsubgen** = *multi-language* + *machine-learning* subtitle generator. It started as a Japanese→English tool
for a personal video collection and grew into a general one: the language of every stretch of speech is detected,
each stretch is transcribed with that language forced, and each of 45 target languages gets its own file.

## Hardware

| | needed | notes |
|---|---|---|
| **GPU** | an NVIDIA card with **8 GB** of memory at least; **24 GB** for the full-quality setup | CUDA only — no AMD, Apple or Windows path, because the ASR engines are CUDA builds. The card never holds the speech engines and the translator at the same time (a run transcribes a round of files, frees the card, then translates), so it needs to hold only the larger stage. Developed and measured on an RTX A5000 (24 GB); the smaller profiles run the same code with less resident and are lightly tested |
| **RAM** | **16 GB** minimum, **32 GB** recommended | models are read through RAM on their way to the card; four tesseract processes run at once during OCR; on a small card Ollama spills part of a translator into RAM when it does not fit |
| **CPU** | 4 cores or more | speaker diarization, bitmap decoding and tesseract OCR run on the CPU, not the GPU |
| **Disk** | **30–65 GB** depending on the profile | Docker image ~14 GB (or the venv ~6 GB on the host) · ASR models ~8 GB (Qwen3-ASR + its aligner, whisper large-v3) · translators in Ollama: `full` ≈ 37 GB (two models), `12gb` 7 GB, `8gb` 6 GB · speaker models 33 MB · tesseract packs ~0.5 GB · work files and OCR cache a few MB per video; the temporary wav is ~120 MB per hour of video and deleted after use |
| **OS / tools** | Linux; `ffmpeg`/`ffprobe`; [Ollama](https://ollama.com) on the same host; Docker with the NVIDIA Container Toolkit, or Python 3.12 + [uv](https://docs.astral.sh/uv/) for a host install | tested on Arch-family (CachyOS) and Debian-family systems |

The GPU's memory picks a **hardware profile** at start (`mlsubgen models` or `mlsubgen config` shows it;
`MLSUBGEN_PROFILE=12gb` or `--profile 12gb` forces one; `mlsubgen pull` downloads what the active profile needs).

The profiles below 24 GB share one trick, measured on 2026-10-03: the translator is Gemma 4's **26B
mixture-of-experts** (`gemma4:26b`, 18 GB on disk) with only as many of its 30 layers on the card as fit, the rest
run from system RAM by Ollama. Because only ~4B of its parameters are active per token, it stays fast with most of
it off the card — generation at 84 tok/s with 24 layers resident, 51 with 16, 40 with 8 (short prompts, measured
on a 24 GB card limited to those splits) — and it translates almost as well as the 31B (chrF++ 30.2 against 31.0
on the same clip and reference; the dense 12B scores 29.0) and matched the 31B on the Thai OCR benchmark (76 % /
93.8 against 77 % / 93.7). One clip, one reference, one script: enough to choose a default, not a general ranking. End to end, prompts included, the ten-minute test clip
translated into Thai in **68 s with the `16gb` split** (22 layers), 31 s with the 26B fully on a 24 GB card, and
103 s with the 31B. The price is system RAM for the part that is not on the card: a machine without it gets the
dense small model instead (the `-dense` profiles, chosen automatically when the RAM is short).

| | `full` — 20 GB and up | `16gb` — 15 to 20 GB | `12gb` — 11 to 15 GB | `8gb` — under 11 GB |
|---|---|---|---|---|
| **cards, for example** | RTX 3090 / 4090 / 5090, RTX A5000 / A6000, L4 (24 GB) | RTX 4080, 4070 Ti Super, 4060 Ti 16 GB, 5070 Ti, RTX A4000, V100 | RTX 3060 12 GB, 4070, 3080 12 GB, 5070 | RTX 3050 / 3070 / 4060 (8 GB), 2070 / 2080 |
| **system RAM for this profile** | 16 GB | 16 GB | 24 GB (else `12gb-dense`) | 32 GB (else `8gb-dense`) |
| **speech recognition** | both engines resident, whisper in float16 (≈ 10 GB) | both engines resident, whisper in float16 (≈ 10 GB) | both engines resident, whisper in int8 (≈ 8 GB) | one engine on the card at a time (≈ 5 GB, then ≈ 2.5 GB); the same two engines, slower per file |
| **translation** | `qwen3.8:27b` for Japanese → English, `gemma4:31b-it-qat` for every other pair, fully on the card (the test clip: 103 s) | `gemma4:26b`, 22 of 30 layers on the card (≈ 13 GB): the test clip in 68 s, measured | `gemma4:26b`, 14 layers on the card (≈ 9 GB): generation ~50 tok/s, the clip in roughly 100 s | `gemma4:26b`, 6 layers on the card (≈ 5 GB): generation ~40 tok/s, the clip in roughly 2 min |
| **translation, `-dense` fallback** | — | — | `gemma4:12b-it-qat` (7.2 GB): chrF++ 29.0 | `gemma4:e4b-it-qat` (6.1 GB): weaker again |
| **language detection, per stretch** | yes | yes | yes | yes |
| **embedded text subtitles** (used before the audio) | yes | yes | yes | yes |
| **bitmap subtitles, Latin / Greek / Cyrillic** (tesseract, CPU) | yes | yes | yes | yes |
| **bitmap subtitles, Thai / Chinese / Japanese / Korean / stacked scripts** (vision model) | yes — the 31B | yes — the 26B, which matched the 31B on the Thai benchmark | yes — the 26B (`12gb-dense`: the 12B, usable but weaker) | yes — the 26B, slower (`8gb-dense`: **no**, the E4B cannot read them; those tracks are left alone and the audio transcribed) |
| **speaker diarization** (CPU) | yes | yes | yes | yes |
| **terminology pass** (names rendered once per film) | yes | yes | yes | yes |
| **bench / lidbench / ocrbench** | yes | yes | yes | yes |

The layer splits leave room for the context window and the vision encoder; the speeds are from a 24 GB card
limited to those splits, so a real 16 GB card with the same split should land close, a slower CPU and RAM lower.
Only the `full` profile has been run on its own hardware for weeks; the others run the same code with less on the
card. `mlsubgen bench` shows what any choice costs on your own material before a long run.

## What it does

The one idea underneath everything: **use the best evidence already in the file before inferring anything.** For
each subtitle language you ask for, in this order:

```
a text subtitle track in that language           → used as is; nothing written
a bitmap (PGS) track in that language            → OCR'd into the .srt — the real subtitles
a text track in the spoken language              → the transcript; translated, no listening
a bitmap track in the spoken language            → OCR'd, then the transcript; translated
nothing to read                                  → listen: language detection, two speech recognisers, translation
```

Every step has a gate that drops evidence which is too thin or too muddled, and falls through to the next.

- **Language detection per stretch of speech**, not per file: a Japanese programme with English interview
  segments is handled as both. Whisper's language probabilities, Qwen3-ASR's own decode and the script of the
  words all have to agree before a language counts.
- **Two speech recognisers on every chunk** — Qwen3-ASR-1.7B with its forced aligner (word timestamps) and
  faster-whisper large-v3 — and the translator LLM reconciles them where they disagree. Chunks cover the whole
  timeline (only stretches at the noise floor are skipped), so dialogue over music is not lost to the VAD.
- **Translation by an LLM** (Ollama, or any OpenAI-compatible server) in windows of 20 cues with rolling
  context, a glossary for names, per-language register rules (Thai particles, du/Sie, Simplified vs
  Traditional Chinese, …), and a per-line fallback for anything the model skips.
- **45 subtitle languages**, chosen per job with tick boxes in the web UI or `--target en,th,de`; a cue already
  in the target language is copied through, not translated.
- **Subtitle typesetting** per language: line length by script, reading-speed ceilings (slower for CJK), minimum
  durations and gaps, cluster-safe line breaks for Thai, Lao, Khmer, Burmese and Devanagari.
- **Optional speaker diarization** — sherpa-onnx finds the speaker turns locally on the CPU. Speaker changes
  sharpen language detection and cue boundaries, and give the translator anonymous speaker continuity; uncertain
  speaker evidence is left unknown rather than forced. `--speakers auto`, or `--speakers N` when you know the
  count. See *Speakers* below.
- **Embedded subtitles are used before the audio is**: a video that already carries full subtitles in a target
  language is left alone for that target, and any other full text track in one of the 45 languages — not forced,
  not signs-and-songs — becomes the transcript to translate from, the spoken language's track first. No ASR time
  spent, and no translating a translation when the original is there.
- **A job queue with a web UI**: jobs run one at a time on the GPU, survive reboots (every finished stage and
  every decoded ASR chunk is checkpointed), and can be paused and resumed. Pausing a job keeps it out of the
  queue across reboots until you resume it.

## Install — Docker

```
git clone https://github.com/dodgypast/mlsubgen.git && cd mlsubgen
cp .env.example .env            # MEDIA_DIR = the folder with your videos; default languages; your uid/gid
mkdir -p data models            # state and the model cache, created by you so the containers can write them
docker compose up -d --build    # the image is ~14 GB (CUDA torch and the ASR stacks), 10–20 minutes the first time
docker compose run --rm worker pull   # the ASR models (~8 GB) and the two default translators (~37 GB, into Ollama)
```

Or open `http://<host>:8790` and use the **Models** panel: it shows what is ready and pulls anything missing with
a click, with progress. The worker uses a model as soon as it is there. The containers use host networking (Ollama at `127.0.0.1:11434`); the web UI
has **no login** unless `MLSUBGEN_WEB_AUTH` is set, so keep it on a LAN or VPN address or behind your proxy (see
*Running it as a service*). Your videos are bind-mounted at `/media`, and
`MLSUBGEN_MEDIA_ROOTS=/media` confines the folder picker and the worker to them. Stopping the worker container
interrupts the running job cleanly; it resumes from its checkpoints on the next start.

## Install — on the host

```
git clone https://github.com/dodgypast/mlsubgen.git ~/mlsubgen && cd ~/mlsubgen
bash setup.sh                   # venv, CUDA torch, pinned deps, selftest, the `mlsubgen` command, the two user units
```

`setup.sh` ends with the remaining steps: edit the two unit files (`MLSUBGEN_MEDIA_ROOTS`, `MLSUBGEN_TARGETS`),
`mlsubgen pull` for the models, then `systemctl --user enable --now mlsubgen-worker mlsubgen-web`.

## Using it

```
mlsubgen                         # every video below here → the default languages; queued for the worker
mlsubgen --target en,th,de .     # three subtitle files per video
mlsubgen --source ja FOLDER      # skip the language detector: the audio is Japanese
mlsubgen --overwrite FILE        # redo one file from scratch
mlsubgen --speakers auto FOLDER  # with speaker diarization (mlsubgen pull speakers once);  --speakers 3  when you know the count
mlsubgen languages               # the 45 codes, their native names, which engine decodes each
mlsubgen config targets th,de    # save the default languages (the web form has "make these the default");  config  shows all settings
mlsubgen models                  # what is ready: translators in Ollama, ASR models in the cache
mlsubgen pull                    # download what a default run needs;  pull gemma4 · pull some/tag:latest · pull --all
mlsubgen jobs                    # the queue;  mlsubgen log ID · pause ID · resume ID · cancel ID · retry ID
mlsubgen why FILE                # where each subtitle file came from: track or engines, detection, speakers, terms, translator
mlsubgen tracks FILE             # the audio and subtitle tracks, with which bitmap tracks OCR can read
mlsubgen help                    # the one-screen guide;  mlsubgen help run  for every option
```

While the worker service is up, `mlsubgen` in a folder queues the job and returns; the worker runs it and the
web UI shows the per-file outcome and the live log. `mlsubgen --now …` runs in the foreground instead.

### The web UI

A new-job form (folder picker confined to your media roots, tick boxes for the subtitle languages, the spoken
language if you want to override the detector, translator, ASR mode, speaker diarization, embedded-subtitle
policy, context, glossary), a **Preview (dry run)** that lists exactly which files a job would touch, the queue
with pause / resume / cancel / retry, a job page with per-file outcomes and the live log, and a page of skipped
files that can be re-queued with the language forced.

### Pipeline

| stage | what |
|---|---|
| embedded subs | a text track in a target language → that target is done; any other full text track (45 languages; the spoken language's first; ASS cleaned of tags, karaoke and comments) → the transcript, no ASR. An untagged text track has its language read from its own words. A **bitmap** (PGS) track is read through OCR (0.4.8): in a target language it becomes that target's .srt, in the spoken language it becomes the transcript — see *Bitmap subtitles* |
| probe | `ffprobe` picks the audio track (tag, then title, then the default) — `mlsubgen tracks FILE` shows them, `--audio-track N` overrides |
| audio | `ffmpeg` → 16 kHz mono wav; deleted after the file's diarization and ASR are complete |
| speakers | *optional, `--speakers`*: speaker turns from sherpa-onnx on the CPU, before anything listens to the words; see *Speakers* |
| language ID | per ~10 s of speech — or per speaker turn when the turns are known: whisper's probability + Qwen's decode + the words' script and function words; switch points refined to the exact span; a short run of another language needs strong evidence |
| chunks | ≤ 30 s, one language each (a language change always cuts), covering the whole timeline; only stretches at the noise floor are skipped |
| ASR | both engines decode every chunk with its language forced; checkpointed after every chunk |
| merge | the translator LLM reconciles the two transcripts where they differ (chunks that agree need no LLM) |
| word ↔ speaker | *with `--speakers`*: each aligned word takes the turn that covers it clearly, or stays unlabelled |
| cues | sentence ends, pauses, speaker changes, length limits, hallucination filters (evidence-gated) |
| terms | per target, once per film: the recurring names and terms of the transcript are found (a script heuristic plus one LLM pass) and rendered once — standard transliteration for names, the established form for titles — into the glossary every window reads, so a character is spelt the same way in the last scene as in the first; your own `--glossary` wins. `--terms off` skips it |
| translate | per target: cues already in the target copied through; the rest in windows of 20 with context, glossary, register rules, speaker continuity, retry and per-line fallback |
| typeset | ≤ 2 lines, per-language line width and reading speed, minimum duration and gaps → `<video>.<lang>.srt` |

Without `--speakers` the two speaker rows simply do not run and everything else is unchanged.

### Bitmap subtitles

Blu-ray remuxes carry their subtitles as PGS bitmaps, often a dozen languages of them and no text track at all,
and until 0.4.8 mlsubgen could use none of them: a film with English subtitles in it was transcribed. Now a bitmap
track is read through OCR (`--ocr auto`, the default; `--ocr off` restores the old behaviour):

- a bitmap track in a **target** language is OCR'd straight into that target's `.srt` — the real subtitles, no
  translation (a text track in that language still wins, and a bitmap target track only counts when a text one is
  absent);
- a bitmap track in the **spoken** language (or, failing that, in the usual source order) becomes the transcript
  the other targets are translated from — human subtitles with OCR noise still beat a transcription.

mlsubgen decodes the PGS stream itself (compositions, palettes, run-length objects) and hands each subtitle image
to the engine measured best for the script:

| script | engine | measured (exact cues / chrF) |
|---|---|---|
| Latin, Greek, Cyrillic | `tesseract` with the language's pack (`tesseract-data-<lang>` on Arch, `tesseract-ocr-<lang>` on Debian; the Docker image ships 19) — CPU, four images at a time | English Blu-ray, 1,800 cues: **95 % / 99.4** |
| Thai, Chinese, Japanese, Korean, and scripts with stacked marks (Lao, Khmer, Burmese, Indic, Arabic, Hebrew) | the hardware profile's vision model through Ollama (`gemma4:31b` on `full`, `12b` on `12gb`) — one image a second on the GPU; unloaded before the ASR engines load | Thai: 31B **77 % / 93.7**, 12B 65 % / 89.3; Chinese Simplified: 31B **79 % / 90.0**; Japanese (a Blu-ray SDH track against another distributor's SDH transcript, descriptions and labels stripped): 31B 51 % identical cues / chrF 78 — the gap is the disc keeping a line on screen while adding the next, interjections the other transcript lacks, and furigana the model correctly left out; paired lines match to the character. tesseract on the same: Thai 50 / 71, Chinese 41 / 78 |
| those scripts on the `8gb` profile | none — the E4B read Thai at 18 % / 66.5, not enough to translate from; the track is left alone and the audio transcribed, as before 0.4.8 | |

Results are cached beside the work files, so a track is OCR'd once. `mlsubgen ocr VIDEO --track N` runs it by
hand (`--engine` overrides the choice); `mlsubgen ocrbench VIDEO` scores the OCR against a text track of the same
film, which is how the numbers above were taken (2026-10-02, a web release whose text tracks come from the same
source as its bitmaps; "Hybrid" releases pair bitmaps and text from different translations and cannot be used,
and a streaming service's SRT muxed beside a Blu-ray's bitmaps may be out of sync with the disc's cut — one
scored 1 % against a track that scored 51 % against another transcript). SDH tracks are compared with
descriptions and speaker labels stripped, and a per-minute content score ignores cue boundaries, because
distributors cut the same dialogue into cues differently.
Systematic habits found this way are corrected after the engine: tight dialogue dashes, a capital I read as a pipe
or an underscore for a dash, Thai *sara am* as two code points, an ellipsis written as LaTeX.

### Speakers

`mlsubgen --speakers auto` (or the Speakers option in the web form) runs speaker diarization before the detector
and uses one result twice:

```
segmentation-3.0 (ONNX)  →  3D-Speaker embeddings  →  sherpa clustering  →  speaker turns
                                                                              ├─ before ASR: language-detection windows and chunk boundaries
                                                                              └─ after ASR:  word attribution → cue boundaries → translator continuity
```

- **Models.** `mlsubgen pull speakers` downloads two ONNX files (33 MB) from sherpa-onnx's GitHub Releases:
  pyannote's `segmentation-3.0` (MIT) and a 3D-Speaker CAM++ embedding model trained on Chinese and English
  (Apache-2.0), and writes a `NOTICE.txt` with the attributions beside them. Any embedding model from that release
  can be used instead (`--speaker-embedding FILE`); `config.py` records how five of them compared. mlsubgen uses sherpa-onnx's ONNX distribution from GitHub, so
  this feature does not require a Hugging Face account or token. The Models panel shows both files like the other
  models. sherpa's clustering is not pyannote's full pipeline; mlsubgen does not claim to reproduce it.
- **`auto` or `N`.** `auto` lets the clustering decide the speaker count (threshold in `config.py`); `--speakers N`
  fixes it, which removes the hardest part of diarization when a cast size is known.
- **Speaker-aware language detection.** Speech is cut at detected speaker changes so ordinary turns from
  different speakers are not judged together (overlapping speech can still share a window), every detected
  speaker gets sampled, and a voice's language history is evidence for its *uncertain* windows — never for its
  confident ones. So a character who switches language mid-scene is still followed, and a bilingual voice is
  learnt as bilingual rather than locked to one language. That is a feature, not a limit.
- **Cues and translation.** A speaker change closes a cue; the translator sees an anonymous tag per line (`[S2]`)
  with the rule that tags mean only "same voice / different voice", nothing about who the speaker is, and never
  appear in the output. A word that two voices cover about equally stays unlabelled.
- **Fallback.** A file whose diarization provides no useful speaker structure — a single voice (a narrator, a
  lecture: nothing to tell apart), so many clusters that it is fragmentation rather than speakers, or more
  ambiguous words than labelled ones — is processed exactly as without `--speakers`.
- **Measuring it.** `mlsubgen bench VIDEO --speakers auto` against the same clip without it shows the effect on cue
  boundaries and translation; `mlsubgen lidbench VIDEO --speakers auto` scores the detector against a film's forced
  subtitle track. A forced track is subtitle-derived evidence, not a transcript: its cue intervals say "a language
  other than the main one is spoken here", and only as a lower bound — songs, lines left untranslated and
  "[speaking German]" cards are foreign speech it does not show, which is what `bench/verified.json` is for. It
  cannot say *which* language; that needs hand-labelled intervals, which `--reference` also accepts. On the
  author's test films (Babel, Inglourious Basterds, Only God Forgives) the speaker-aware detector raised recall
  and switch recall where speakers map cleanly to languages and was mixed on feature films where the clustering
  fragments into many voices — which is why it is off by default.

### Translators

Presets in `mlsubgen/config.py`; `mlsubgen models` shows which are pulled.

| preset | model | used for |
|---|---|---|
| `qwen3.8` | `qwen3.8:27b` (q4_K_M, 18 GB) | Japanese → English (default route) |
| `gemma4` | `gemma4:31b-it-qat` (19 GB) | every other language pair (default route) |
| `translategemma` | `translategemma:27b` (17 GB) | translation-only Gemma; fixed prompt |
| `qwen3-30b` | `qwen3:30b-a3b-instruct-2507-q4_K_M` (18 GB) | the fast MoE fallback (~3 B active) |
| `gemma4-12b` | `gemma4:12b-it-qat` (7.2 GB) | the `12gb` profile's translator |
| `gemma4-e4b` | `gemma4:e4b-it-qat` (6.1 GB) | the `8gb` profile's translator |
| `gemma4-26b` | `gemma4:26b` (18 GB; ~4B active) | the `16gb`, `12gb` and `8gb` profiles' translator and vision model — part of it on the card, the rest in RAM (see Hardware); on a 24 GB card `-t gemma4-26b` trades 0.8 chrF++ for 3.7× the speed |

`-t NAME` forces one preset for every pair; `--model TAG` any Ollama model; `--backend openai --url http://host:port
--model NAME` any OpenAI-compatible server (llama-server, vLLM …). `mlsubgen bench VIDEO --clip 0:10:00-0:20:00`
runs the ASR once and every translator on the same cues, and writes a side-by-side HTML page (with a chrF++ score
when you give it a reference `.srt`) — the way to choose a model for a language pair.

### Languages

`mlsubgen languages` lists the 45 codes. Every one is a target (the translator writes it) and a source (Qwen3-ASR
decodes the languages its aligner covers — Japanese, Chinese, Cantonese, Korean, English, French, German, Italian,
Portuguese, Russian, Spanish — and whisper the rest). Target quality is the translator's: the presets above are
strong in the major languages and Thai, and `bench` is the way to judge a pair before a long run. Low-resource
targets (Khmer, Lao, Burmese) depend heavily on the model.

### Context and glossary

`--context "NHK documentary about the Tōhoku coast; presenter Tanaka Yūki"` biases both the ASR (fed to
Qwen3-ASR) and the translation. `--glossary names.tsv` (`source<TAB>target` per line) pins names and terms.
`--genre "a slapstick family anime"` sets the register.

### What a run leaves behind

Only the `.srt` files, plus `logs/skipped.log` (date, path, reason — one line per skipped file). The temp wav is
deleted after each file's diarization and ASR are complete, and the per-file work file (detection, ASR words, cues, translations) once its last
`.srt` is written. An interrupted run keeps the work files of unfinished files and resumes from them; a skipped
file remembers its verdict so a repeated run does not re-read it (`--source`, `--overwrite` or `--audio-track`
retries it). `--keep-work` keeps the ASR cache to re-translate with another model; `mlsubgen clean` wipes leftovers.

## Running it as a service

The things a self-hoster asks before `docker compose up`:

- **Login.** None by default. Set `MLSUBGEN_WEB_AUTH=user:password` in `.env` (or the unit's environment) and
  every page and API route of the web UI asks for it (HTTP Basic). That is the minimum, not a security boundary:
  for anything reachable from outside your LAN or VPN, put it behind your reverse proxy with its own login.
- **Networking.** The compose stack uses `network_mode: host` because the translator is Ollama on the same host
  at `127.0.0.1:11434`, and the web UI listens on `MLSUBGEN_WEB_PORT` (8790). To run it bridged, drop the host
  mode, publish the port, and point `MLSUBGEN_LLM_URL` at the host's Ollama (`compose.yaml` has the lines).
  Nothing listens but the web UI; the worker makes outbound calls to Ollama only.
- **What it writes, and where.** Beside your videos: the `.srt` files, nothing else. Under `./data` (the
  `MLSUBGEN_HOME` volume): the job queue (`mlsubgen.db`), job logs, `logs/skipped.log`, per-file work files (deleted
  once a file's last `.srt` is written), the OCR cache, temporary audio, saved glossaries and `settings.json`.
  Under `./models`: the Hugging Face cache (~8 GB) and the speaker models. Worth backing up: `settings.json` and any
  glossaries; everything else is a cache or a log. The translators live in Ollama's own store.
- **Idle footprint.** Between jobs the worker holds nothing on the card: the speech engines are loaded for a
  round of files and freed after it, the vision model is unloaded after each track, and Ollama drops the
  translator after its `keep_alive` (5 minutes by default). Idle, the two containers are a small Python process
  each. Busy, see Hardware.
- **Updating.** `git pull && docker compose up -d --build`; a running job is interrupted cleanly and resumes
  from its checkpoints. On a host install: `git pull`, then `systemctl --user restart mlsubgen-worker
  mlsubgen-web` — the services keep the code they imported until they are restarted.
- **Scheduling.** There is no watch folder. A nightly `mlsubgen /media` from cron (or any scheduler) queues
  whatever is new — files that already have their subtitle files are skipped at the scan, so running it over the
  whole library every night is cheap — and the worker runs the queue one job at a time.
- **Media servers.** Jellyfin, Plex, Emby and Kodi pick the `.srt` sidecars up on their next library scan, by the
  language code in the file name. mlsubgen is not a Bazarr provider and has no webhook; it is a folder tool with a
  queue. If you want a request from a media server to start a job, the queue has a JSON API (`POST /api/jobs` with
  a `path`), which is what the web form calls.
- **Monitoring.** A transcription keeps every core busy for the length of a film, which a CPU alert cannot tell
  from a fault. Set `MLSUBGEN_METRICS_FILE` to a `.prom` file in node_exporter's textfile directory (the worker
  must be allowed to write it; in Docker, bind-mount that directory) and the worker publishes
  `mlsubgen_worker_busy`, `mlsubgen_worker_job_id`, `mlsubgen_jobs_queued`, `mlsubgen_worker_up` and a
  heartbeat, atomically, a minute apart while a job runs. The alert rule then reads *CPU saturated **and**
  `mlsubgen_worker_busy == 0`*. Unset, nothing is written.
- **Compared with the whisper wrappers** (whisper-asr-webservice, Subgen, Bazarr's whisper provider): the speech
  recognition is not the difference; the order of evidence is. Those transcribe; mlsubgen reads the subtitles the
  file already has first, text or bitmap, and transcribes only what nothing covers, with the language decided per
  stretch of speech and the translation done locally with the film's names rendered once. Every claim above has
  a measurement behind it in this README; the wrappers are faster where a file has nothing to read.

## Configuration

| variable | default | |
|---|---|---|
| `MLSUBGEN_HOME` | `~/mlsubgen` (`/data` in Docker) | state: work files, logs, the queue database |
| `MLSUBGEN_MEDIA_ROOTS` | *(none — set it)* | colon-separated folders the worker and the picker may use |
| `MLSUBGEN_TARGETS` | `en` | default subtitle languages — a saved setting (`mlsubgen config targets …` or the web form's *make these the default*, kept in `settings.json` under `MLSUBGEN_HOME`) beats it; `--target` on a run beats both. English need not be among them: every route, rule and file name is per target. |
| `MLSUBGEN_LLM_URL` | `http://127.0.0.1:11434` | the translator server |
| `MLSUBGEN_WEB_HOST` / `MLSUBGEN_WEB_PORT` | `0.0.0.0` / `8790` | the web UI |
| `MLSUBGEN_WEB_AUTH` | *(none)* | `user:password` — HTTP Basic auth on every page and API route of the web UI |
| `MLSUBGEN_METRICS_FILE` | *(none)* | a `.prom` file in node_exporter's textfile directory: the worker's busy / queue / heartbeat metrics for Prometheus |
| `MLSUBGEN_PROFILE` | `auto` | `full`, `16gb`, `12gb`, `8gb`, `12gb-dense` or `8gb-dense` — see Hardware |
| `HF_HUB_OFFLINE` | `0` | `1` after the models are downloaded: no contact with huggingface.co |

Everything else (chunk lengths, cue limits, line widths, reading speeds, hallucination patterns) is in
`mlsubgen/config.py` and documented there.

## Questions a sceptical reader should ask

**Why two speech recognisers and an LLM, rather than one good one?** Because they fail differently. An LLM
decoder (Qwen3-ASR) can go silent over music; a whisper window can drop out or hallucinate a stock phrase; each
hears names and numbers the other misses. Where the two transcripts agree closely — most chunks — no LLM is
involved and nothing is slower than one engine. Where they differ, the translator model reconciles them, and a
chunk one engine returned thin is decoded again by the other. Whether that is worth the compute on *your*
material is a measurement, not a promise: `mlsubgen bench … --asr qwen`, `--asr whisper` and the default dual
mode on the same clip, with a reference `.srt`, give a chrF++ for each.

**Does the vision-model OCR make things up?** It can: a generative model's failure is plausible text that is not
on the screen, which is worse than tesseract's symbol salad because it looks right. Four things stand against
it. The prompt asks for the visible characters only — no correcting, no paraphrasing, no inferred words, no
translation. The OCR gate refuses a track whose text is not in the expected script, is mostly symbols, repeats
one line, contains replacement characters, carries lines of digits and symbols, or describes the image; a
refused track is left alone and the audio is transcribed. Every engine–script pairing in the table above was
measured against a text track of the same film before it was allowed to be a default. And `mlsubgen why VIDEO`
tells you afterwards which track, engine and gate verdict a subtitle file came from. The risk is reduced, not
gone; a disagreement check between two engines is the obvious next step and is not built.

**Do the 8, 12 and 16 GB profiles work on real cards?** They were measured on a 24 GB RTX A5000 limited to their
layer splits, which fixes the memory and the quality but not the speed: a real card's CPU and RAM carry the rest
of the 26B, and a slower machine will be slower than the figures in the table. Nothing about correctness depends
on the card. A report from a 3060, a 4070, a 4060 Ti 16 GB or a 3070 is the most useful issue this repository
can receive.

**How were the numbers made, and what do they leave out?** Each measurement names its material and its
reference: one clip of one anime with a fansub reference for translation and cue boundaries; three multilingual
films against their forced subtitle tracks for language detection; five films against their own text tracks for
OCR. They are enough to choose between alternatives on the same material, which is what they were used for.
They are not a benchmark of the field, and a different film can rank the alternatives differently — which is why
`bench`, `lidbench` and `ocrbench` exist, so the same comparison can be run on yours.

**What stays local?** Everything except the one-time model downloads (Hugging Face for the speech models, GitHub
for the speaker models, Ollama for the translators). Set `HF_HUB_OFFLINE=1` after that and the speech side makes
no network calls at all; Ollama and the web UI listen on the addresses you give them.

## Known limitations

- **Language detection on mixed material is the weakest link.** It has improved a lot (per-stretch detection, three
  sources of evidence) but a file with two languages in quick alternation can still get a stretch wrong; `--source`
  forces the language, and `mlsubgen scan` shows what the detector sees without running the ASR.
- The web UI has no authentication unless `MLSUBGEN_WEB_AUTH` is set, and then only HTTP Basic: a reverse proxy
  with a real login is the answer for anything exposed beyond a LAN or VPN.
- NVIDIA only.
- The `16gb`, `12gb` and `8gb` profiles have been measured on a 24 GB card limited to their layer splits, not on
  their own hardware: the ASR side is the same code with less resident at once, and the 26B's speed on a real
  card depends on its CPU and RAM as much as on the GPU. Reports from those cards are the most useful thing a
  user can send.
- Diarization can be unreliable with overlapping speech, similar voices and music-heavy material, and on a
  feature film the clustering tends to split a cast into many more "voices" than there are. Speaker evidence is
  therefore advisory and ignored where it is too thin, too muddled or too fragmented; the feature is off by
  default until `bench` or `lidbench` shows it helps on your material.

## Support and provenance

Issues and pull requests are welcome, and issues get answered. The reference environment is the pinned one in
`requirements.txt` / the Dockerfile; "works for me" means works there. This code was written with the help of
Claude (Anthropic's model), reviewed and run in daily use by the author; treat it as a hobby project maintained by
one person, not a product.

## Licence and credits

Apache-2.0 (see `LICENSE`). Built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and
Qwen3-ForcedAligner (Apache-2.0), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT),
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT), [Ollama](https://ollama.com), and for speakers
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0) with pyannote's
[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (MIT) and a
[3D-Speaker](https://github.com/modelscope/3D-Speaker) embedding model (Apache-2.0). The translator models carry
their own licences (Qwen: Apache-2.0; Gemma: Google's Gemma terms).
