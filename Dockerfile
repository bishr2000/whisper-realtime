FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE=float16 \
    WHISPER_MODEL=large-v3

# System deps (espeak-ng needed by Kokoro TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn uvloop httptools

# Copy application
COPY app.py .
COPY templates/ templates/
COPY octopus_repo/ octopus_repo/

# Pre-download models at build time (optional, makes startup faster)
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

EXPOSE 8000

# Gunicorn with Uvicorn workers — production ASGI server
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:8000", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--timeout", "300", \
     "--keep-alive", "65", \
     "--access-logfile", "-"]
