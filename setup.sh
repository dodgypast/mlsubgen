#!/usr/bin/env bash
# mlsubgen — one-time host install (the alternative to Docker). Needs: Linux, an NVIDIA GPU with a current driver,
# ffmpeg, and uv (https://docs.astral.sh/uv/). Creates ./.venv with Python 3.12, CUDA torch and the pinned deps,
# links the `mlsubgen` command into ~/.local/bin and installs the two user units (not enabled).
#   git clone <repo> ~/mlsubgen && cd ~/mlsubgen && bash setup.sh
# The checkout has to live at ~/mlsubgen: the units and the launcher expect it there (%h/mlsubgen).
set -euo pipefail
cd "$(dirname "$0")"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

command -v ffmpeg >/dev/null || { echo "ffmpeg missing (apt install ffmpeg / pacman -S ffmpeg / dnf install ffmpeg)"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe missing (ships with ffmpeg)"; exit 1; }
[ -x "$UV" ] || { echo "uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ "$PWD" = "$HOME/mlsubgen" ] || echo "note: this checkout is $PWD, not ~/mlsubgen — the units and ~/.local/bin/mlsubgen expect ~/mlsubgen"

"$UV" venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
"$UV" pip install --upgrade pip wheel setuptools
# CUDA 12.8 wheels — the same pins as the Dockerfile
"$UV" pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install -r requirements.txt

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "— no GPU seen: check the NVIDIA driver")
import qwen_asr, silero_vad, soundfile, numpy, faster_whisper  # noqa
print("imports OK")
EOF
python -m mlsubgen selftest
# a symlink, not a copy: a copy goes stale when bin/mlsubgen changes
mkdir -p "$HOME/.local/bin" "$HOME/.config/systemd/user"
ln -sfn "$PWD/bin/mlsubgen" "$HOME/.local/bin/mlsubgen"
install -m 644 mlsubgen-worker.service mlsubgen-web.service "$HOME/.config/systemd/user/"

cat <<'EOF'

Installed: the `mlsubgen` command (~/.local/bin/mlsubgen) and the mlsubgen-worker / mlsubgen-web user units.
Next:
  1. Edit the units — the folders your videos live in, your default languages:
       ~/.config/systemd/user/mlsubgen-worker.service   (MLSUBGEN_MEDIA_ROOTS=..., MLSUBGEN_TARGETS=...)
       ~/.config/systemd/user/mlsubgen-web.service      (the same two lines)
  2. A translator model in Ollama (https://ollama.com), e.g.
       ollama pull qwen3.8:27b          # 18 GB — the default route for Japanese → English (24 GB GPU)
       ollama pull gemma4:31b-it-qat    # 19 GB — the default for every other language pair
       ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M   # ~18 GB on disk, ~3 B active — the fast fallback preset
     then `mlsubgen models` shows which presets are ready; `-t NAME` / `--model TAG` pick others.
  3. Start the services (they survive logouts and reboots):
       loginctl enable-linger "$USER"
       systemctl --user daemon-reload && systemctl --user enable --now mlsubgen-worker mlsubgen-web
  4. In any folder of videos:   mlsubgen     — queued for the worker; http://<this host>:8790 shows it.

The ASR models (Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B, whisper large-v3 — about 8 GB) download from
Hugging Face on first use. After that, runs work with no internet.
EOF
