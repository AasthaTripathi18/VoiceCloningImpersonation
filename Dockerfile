# CPU-only base image. If you need GPU inference, swap the base for
# nvidia/cuda:12.1.0-runtime-ubuntu22.04 and install torch with CUDA wheels
# (see requirements.txt comment).
FROM python:3.11-slim

# System deps: libsndfile for soundfile/librosa, ffmpeg for broad audio
# format support (mp3/m4a decoding).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

# Pre-download the Wav2Vec2 base weights and Whisper model into the image
# at build time, so the container doesn't hit Hugging Face on every cold
# start in production. Comment this out if you'd rather fetch at runtime.
RUN python -c "from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor; \
    Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base'); \
    Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')"
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Your trained checkpoint. Place best.pt in deploy/checkpoints/ before
# building, or mount it as a volume at runtime instead (see docker-compose.yml).
COPY checkpoints/ /app/checkpoints/

ENV CHECKPOINT_PATH=/app/checkpoints/best.pt
ENV PRETRAINED_MODEL=facebook/wav2vec2-base
ENV FAKE_PROB_THRESHOLD=0.6
ENV WHISPER_MODEL_SIZE=base

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
