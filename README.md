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
mlsubgen --speakers auto FOLDER  # with speaker diarization (mlsubgen pull speakers once);  --speakers 3  when you know the count
mlsubgen languages               # the 45 codes, their native names, which engine decodes each
mlsubgen config targets th,de    # save the default languages (the web form has "make these the default");  config  shows all settings
mlsubgen models                  # what is ready: translators in Ollama, ASR models in the cache
mlsubgen pull                    # download what a default run needs;  pull gemma4 · pull some/tag:latest · pull --all
mlsubgen jobs                    # the queue;  mlsubgen log ID · pause ID · resume ID · cancel ID · retry ID
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
| embedded subs | a text track in a target language → that target is done; any other full text track (45 languages; the spoken language's first; ASS cleaned of tags, karaoke and comments) → the transcript, no ASR. An untagged text track has its language read from its own words |
| probe | `ffprobe` picks the audio track (tag, then title, then the default) — `mlsubgen tracks FILE` shows them, `--audio-track N` overrides |
| audio | `ffmpeg` → 16 kHz mono wav; deleted after the file's diarization and ASR are complete |
| speakers | *optional, `--speakers`*: speaker turns from sherpa-onnx on the CPU, before anything listens to the words; see *Speakers* |
| language ID | per ~10 s of speech — or per speaker turn when the turns are known: whisper's probability + Qwen's decode + the words' script and function words; switch points refined to the exact span; a short run of another language needs strong evidence |
| chunks | ≤ 30 s, one language each (a language change always cuts), covering the whole timeline; only stretches at the noise floor are skipped |
| ASR | both engines decode every chunk with its language forced; checkpointed after every chunk |
| merge | the translator LLM reconciles the two transcripts where they differ (chunks that agree need no LLM) |
| word ↔ speaker | *with `--speakers`*: each aligned word takes the turn that covers it clearly, or stays unlabelled |
| cues | sentence ends, pauses, speaker changes, length limits, hallucination filters (evidence-gated) |
| translate | per target: cues already in the target copied through; the rest in windows of 20 with context, glossary, register rules, speaker continuity, retry and per-line fallback |
| typeset | ≤ 2 lines, per-language line width and reading speed, minimum duration and gaps → `<video>.<lang>.srt` |

Without `--speakers` the two speaker rows simply do not run and everything else is unchanged.

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
deleted after each file's diarization and ASR are complete, and the per-file work file (detection, ASR words, cues, translations) once its last
`.srt` is written. An interrupted run keeps the work files of unfinished files and resumes from them; a skipped
file remembers its verdict so a repeated run does not re-read it (`--source`, `--overwrite` or `--audio-track`
retries it). `--keep-work` keeps the ASR cache to re-translate with another model; `mlsubgen clean` wipes leftovers.

## Configuration

| variable | default | |
|---|---|---|
| `MLSUBGEN_HOME` | `~/mlsubgen` (`/data` in Docker) | state: work files, logs, the queue database |
| `MLSUBGEN_MEDIA_ROOTS` | *(none — set it)* | colon-separated folders the worker and the picker may use |
| `MLSUBGEN_TARGETS` | `en` | default subtitle languages — a saved setting (`mlsubgen config targets …` or the web form's *make these the default*, kept in `settings.json` under `MLSUBGEN_HOME`) beats it; `--target` on a run beats both. English need not be among them: every route, rule and file name is per target. |
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
