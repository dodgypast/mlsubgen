"""mlsubgen — Japanese audio in video files → English .srt subtitles, entirely local.

Pipeline: ffprobe (pick the Japanese audio track) → ffmpeg (16 kHz mono wav) → Silero VAD (speech regions,
≤4-minute chunks) → Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B (Japanese text with word timestamps) →
cue segmentation + hallucination filters → LLM translation with rolling context (Ollama / any OpenAI-compatible
server) → subtitle typesetting rules → <video>.en.srt beside the video.

Nothing leaves the machine: models are pulled once, audio and text never go to a cloud API, and the work
files (transcripts, translations) live under ~/mlsubgen/work, not beside the videos.

Long jobs run under the mlsubgen-worker service (`mlsubgen serve`): `mlsubgen` in a folder queues the job, the worker runs
queued jobs one at a time, and a reboot only pauses them — every stage is checkpointed in the work files.
`mlsubgen pause ID` / `mlsubgen resume ID` (and the web buttons) hold a job on purpose, across reboots; an --overwrite
job resumes where it stopped because the worker hands the run the job's creation time (`--since`).
"""

__version__ = "0.5.10"
