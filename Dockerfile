# CPU image. For GPU, start from an nvidia/cuda runtime image and install the CUDA builds of torch instead.
# NOTE: written and syntax-checked, but the image has not been built end to end in the development environment.
FROM python:3.9-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Non-root user; /models holds downloaded models and is meant to be a volume.
RUN useradd --create-home app && mkdir -p /models /data && chown app:app /models /data
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Versions that were run together during development (CPU wheels).
RUN pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[models]"

USER app
ENV HF_HOME=/models/hf XDG_CACHE_HOME=/models/cache NLTK_DATA=/models/nltk
# HF_TOKEN is passed at runtime (compose / `docker run -e HF_TOKEN`), never baked into the image.
VOLUME ["/models"]
ENTRYPOINT ["profanity-diarization"]
CMD ["--help"]
