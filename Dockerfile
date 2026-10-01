# mlsubgen — one image with the dependencies AND the code (/app). State lives in /data (MLSUBGEN_HOME), the
# Hugging Face model cache in /models (HF_HOME); both are volumes, so the image is replaceable.
#   docker compose up -d --build          (see compose.yaml and .env.example)
#   docker build -t mlsubgen:latest .     dependencies only change with requirements.txt; code is the last layer
# Developers who bind-mount a checkout over /app (or anywhere, with PYTHONPATH pointing at it) get edits on the
# next job with no rebuild.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# CUDA 12.8 wheels (any reasonably current NVIDIA driver is fine); the versions the project was validated with
RUN pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
# run as an ordinary user whose uid/gid match the files it writes (the .srt beside the video, /data, /models):
#   docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) ...   (compose does this from PUID/PGID)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} mlsubgen && useradd -m -u ${UID} -g ${GID} -s /bin/bash mlsubgen
COPY --chown=mlsubgen:mlsubgen . /app
USER mlsubgen
ENV HOME=/home/mlsubgen PYTHONPATH=/app MLSUBGEN_HOME=/data HF_HOME=/models
WORKDIR /app
VOLUME ["/data", "/models"]
EXPOSE 8790
ENTRYPOINT ["python", "-m", "mlsubgen"]
CMD ["serve"]
