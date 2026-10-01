# mlsubgen — subtitles for videos, entirely on your own machine

```
cd /some/folder/of/videos
mlsubgen
```

Every video in the folder and below it gets one `.srt` per subtitle language you asked for
(`<video>.en.srt`, `<video>.th.srt` …). Nothing leaves the machine: the speech is transcribed by local models,
translated by a local LLM, and the only things written are the `.srt` files beside the videos.

**mlsubgen** = *multi-language* + *machine-learning* subtitle generator. It started as a Japanese→English tool
for a personal video collection and grew into a general one: the language of every stretch of speech is detected,
each stretch is transcribed with that language forced, and each of 45 target languages gets its own file.

## What it does

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
- **Embedded subtitles are used before the audio is**: a video that already carries full subtitles in a target
  language is left alone for that target, and any other full text track in one of the 45 languages — not forced,
  not signs-and-songs — becomes the transcript to translate from, the spoken language's track first. No ASR time
  spent, and no translating a translation when the original is there.
- **A job queue with a web UI**: jobs run one at a time on the GPU, survive reboots (every finished stage and
  every decoded ASR chunk is checkpointed), and can be paused and resumed. Pausing a job keeps it out of the
  queue across reboots until you resume it.

## Requirements

| | |
|---|---|
| OS | Linux (tested on Arch-family and Debian-family). No AMD, Apple or Windows path: the ASR engines are CUDA builds. |
| GPU | NVIDIA. **8 GB** is the floor; **24 GB** gets the full-quality setup. The hardware profile (below) adapts the pipeline to what the card has. |
| Translator | [Ollama](https://ollama.com) on the same host (or any OpenAI-compatible server) with a model pulled — see Translators. |
| Tools | `ffmpeg` / `ffprobe`; Python 3.12 and [uv](https://docs.astral.sh/uv/) for the host install, or Docker with the NVIDIA Container Toolkit. |
| Disk | ~8 GB of models downloaded from Hugging Face on first run, plus the translator in Ollama (17–19 GB each). |

The ASR engines and the translator never share the GPU: a run transcribes a round of files first, frees the
card, then translates the round. So the card only has to hold the bigger of the two stages, and a **hardware
profile**, picked from the GPU's memory at start, sets what each stage loads:

| profile | GPU | ASR stage | translators |
|---|---|---|---|
| `full` | 20 GB and up | both engines resident (≈ 10 GB) | `qwen3.8:27b` for Japanese → English, `gemma4:31b-it-qat` for every other pair (17–19 GB each) |
| `12gb` | 11–20 GB | both engines resident, whisper in int8 (≈ 8 GB) | `gemma4:12b-it-qat` (7.2 GB) for every pair |
| `8gb` | under 11 GB | one engine on the card at a time (≈ 5 GB, then ≈ 2.5 GB) | `gemma4:e4b-it-qat` (6.1 GB) for every pair |

`mlsubgen models` shows the active profile; `MLSUBGEN_PROFILE=12gb` (or `--profile 12gb` on a run) forces one, and
`mlsubgen pull` downloads what the active profile needs. The smaller translators are noticeably weaker, most of all
for Japanese → English; `mlsubgen bench` shows by how much on your own material. On a 16 GB card the `12gb` profile
applies; `-t gemma4-26b` tries the 26B-A4B mixture-of-experts (16–19 GB) there, with Ollama offloading part of it
to the CPU.

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
has **no login**, so keep it on a LAN or VPN address. Your videos are bind-mounted at `/media`, and
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
mlsubgen languages               # the 45 codes, their native names, which engine decodes each
mlsubgen models                  # what is ready: translators in Ollama, ASR models in the cache
mlsubgen pull                    # download what a default run needs;  pull gemma4 · pull some/tag:latest · pull --all
mlsubgen jobs                    # the queue;  mlsubgen log ID · pause ID · resume ID · cancel ID · retry ID
mlsubgen help                    # the one-screen guide;  mlsubgen help run  for every option
```

While the worker service is up, `mlsubgen` in a folder queues the job and returns; the worker runs it and the
web UI shows the per-file outcome and the live log. `mlsubgen --now …` runs in the foreground instead.

### The web UI

A new-job form (folder picker confined to your media roots, tick boxes for the subtitle languages, the spoken
language if you want to override the detector, translator, ASR mode, embedded-subtitle policy, context, glossary),
a **Preview (dry run)** that lists exactly which files a job would touch, the queue with pause / resume / cancel /
retry, a job page with per-file outcomes and the live log, and a page of skipped files that can be re-queued with
the language forced.

### Pipeline

| stage | what |
|---|---|
| embedded subs | a text track in a target language → that target is done; any other full text track (45 languages; the spoken language's first; ASS cleaned of tags, karaoke and comments) → the transcript, no ASR |
| probe | `ffprobe` picks the audio track (tag, then title, then the default) — `mlsubgen tracks FILE` shows them, `--audio-track N` overrides |
| audio | `ffmpeg` → 16 kHz mono wav (deleted after the file's ASR) |
| language ID | per ~10 s of speech: whisper's probability + Qwen's decode + the script of the words; a second language needs consecutive confident windows |
| chunks | ≤ 30 s, one language each, covering the whole timeline; only stretches at the noise floor are skipped |
| ASR | both engines decode every chunk with its language forced; checkpointed after every chunk |
| merge | the translator LLM reconciles the two transcripts where they differ (chunks that agree need no LLM) |
| cues | sentence ends, pauses, length limits, hallucination filters (evidence-gated) |
| translate | per target: cues already in the target copied through; the rest in windows of 20 with context, glossary, register rules, retry and per-line fallback |
| typeset | ≤ 2 lines, per-language line width and reading speed, minimum duration and gaps → `<video>.<lang>.srt` |

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
| `gemma4-26b` | `gemma4:26b` (16–19 GB) | 26B-A4B MoE for 16 GB cards (`-t gemma4-26b`) |

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
deleted after each file's ASR and the per-file work file (detection, ASR words, cues, translations) once its last
`.srt` is written. An interrupted run keeps the work files of unfinished files and resumes from them; a skipped
file remembers its verdict so a repeated run does not re-read it (`--source`, `--overwrite` or `--audio-track`
retries it). `--keep-work` keeps the ASR cache to re-translate with another model; `mlsubgen clean` wipes leftovers.

## Configuration

| variable | default | |
|---|---|---|
| `MLSUBGEN_HOME` | `~/mlsubgen` (`/data` in Docker) | state: work files, logs, the queue database |
| `MLSUBGEN_MEDIA_ROOTS` | *(none — set it)* | colon-separated folders the worker and the picker may use |
| `MLSUBGEN_TARGETS` | `en` | default subtitle languages |
| `MLSUBGEN_LLM_URL` | `http://127.0.0.1:11434` | the translator server |
| `MLSUBGEN_WEB_HOST` / `MLSUBGEN_WEB_PORT` | `0.0.0.0` / `8790` | the web UI |
| `MLSUBGEN_PROFILE` | `auto` | `full`, `12gb` or `8gb` — see Requirements |
| `HF_HUB_OFFLINE` | `0` | `1` after the models are downloaded: no contact with huggingface.co |

Everything else (chunk lengths, cue limits, line widths, reading speeds, hallucination patterns) is in
`mlsubgen/config.py` and documented there.

## Known limitations

- **Language detection on mixed material is the weakest link.** It has improved a lot (per-stretch detection, three
  sources of evidence) but a file with two languages in quick alternation can still get a stretch wrong; `--source`
  forces the language, and `mlsubgen scan` shows what the detector sees without running the ASR.
- The web UI has no authentication.
- NVIDIA only.
- The `12gb` and `8gb` profiles are new and lightly tested: the ASR side is the same code with less resident at
  once, but the small translators have had far less use than the 27–31B ones. Reports welcome.

## Support and provenance

Issues and pull requests are welcome, and issues get answered. The reference environment is the pinned one in
`requirements.txt` / the Dockerfile; "works for me" means works there. This code was written with the help of
Claude (Anthropic's model), reviewed and run in daily use by the author; treat it as a hobby project maintained by
one person, not a product.

## Licence and credits

Apache-2.0 (see `LICENSE`). Built on [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and
Qwen3-ForcedAligner (Apache-2.0), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT),
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT) and [Ollama](https://ollama.com). The translator models
carry their own licences (Qwen: Apache-2.0; Gemma: Google's Gemma terms).
