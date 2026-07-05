# Whisper ASR — Real-time Speech Recognition & Translation

A real-time web application for **speech recognition**, **translation**, and **text-to-speech** supporting English and Arabic. Streams microphone audio via WebSocket, transcribes with Whisper on GPU, translates with NLLB-200, and speaks the result back.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Real-time streaming ASR** — Browser mic → WebSocket → GPU transcription with sub-second latency
- **Semantic turn detection** — Silero VAD detects natural speech pauses instead of fixed-duration chunking
- **Multiple Whisper models** — Switch between 7 faster-whisper models + Arabic-finetuned variants on the fly
- **Neural machine translation** — NLLB-200 (600M) for high-quality EN⇄AR translation
- **Text-to-speech** — Kokoro-82M for English, Edge TTS for Arabic, with auto-speak toggle
- **Noise cancellation** — 3-stage pipeline: spectral gating + stationary noise removal + high-pass filter
- **Live waveform visualizer** — Canvas-based waveform with VU meter and VAD state indicator
- **File upload** — Drag-and-drop audio files (MP3, WAV, FLAC, OGG, WebM, M4A)
- **Single-file backend** — Everything in one `app.py` — no microservices, no Docker required

## Architecture

```
Browser (Web Audio API)
  │  16-bit PCM @ 16kHz
  ▼
WebSocket (/ws/stream)
  │
  ├─► Silero VAD ──► Turn boundary detection
  │                    │
  ▼                    ▼
Noise Cancellation ◄── Complete utterance
  │
  ▼
Whisper ASR (faster-whisper / HF transformers)
  │
  ▼
NLLB-200 Translation (EN⇄AR)
  │
  ▼
JSON result → Browser
  │
  ▼
TTS (Kokoro / Edge TTS) → Audio playback
```

## Models

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| ASR | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (multiple) | 1–10 GB VRAM | CTranslate2, default: `large-v3` |
| ASR (Arabic) | [whisper-large-v3-turbo-arabic](https://huggingface.co/mboushaba/whisper-large-v3-turbo-arabic) | ~3 GB | Arabic-finetuned, WER 31 |
| ASR (Arabic) | [whisper-large-arabic-cv-11](https://huggingface.co/KalamTech/whisper-large-arabic-cv-11) | ~3 GB | Arabic CV11, WER 12.6 |
| ASR (Arabic) | [ArabicSpeech/Octopus](https://huggingface.co/ArabicSpeech/Octopus) | ~4 GB | Multi-task Arabic speech LLM |
| Translation | [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | ~2.5 GB | 200-language neural MT |
| TTS (English) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | ~350 MB | Fast, natural-sounding |
| TTS (Arabic) | Edge TTS | Cloud | Microsoft Edge voices (free) |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) | ~2 MB | Turn detection at 16kHz |

## Requirements

- **Python** 3.10+
- **NVIDIA GPU** with CUDA (6+ GB VRAM for `large-v3`, less for smaller models)
- **eSpeak NG** — required by Kokoro TTS ([download](https://github.com/espeak-ng/espeak-ng/releases))

## Quick Start

### Windows (one-click)

```bash
# Double-click run.bat — it creates a venv, installs everything, and starts the server
run.bat
```

### Manual Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/whisper-realtime.git
cd whisper-realtime

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Install TTS engines
pip install kokoro>=0.9.4 edge-tts

# Start the server
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_DEVICE` | `cuda` | Device for ASR (`cuda` or `cpu`) |
| `WHISPER_COMPUTE` | `float16` | Compute type (`float16`, `int8`, `float32`) |
| `WHISPER_MODEL` | `large-v3` | Default Whisper model to load |
| `NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | Translation model |
| `NLLB_CACHE` | `~/.cache/huggingface` | Model cache directory |
| `TTS_VOICE` | `af_heart` | Kokoro voice for English TTS |
| `TTS_SPEED` | `1.0` | TTS speaking speed |
| `AR_TTS_VOICE` | `ar-SA-HamedNeural` | Edge TTS voice for Arabic |
| `VAD_THRESHOLD` | `0.40` | VAD speech probability threshold |
| `TURN_SILENCE_MS` | `1400` | Silence duration to end a turn (ms) |
| `MIN_SPEECH_MS` | `1000` | Minimum speech before allowing turn end |
| `MAX_TURN_DURATION` | `30` | Maximum turn length in seconds |
| `CHUNK_DURATION` | `3` | Fixed chunk duration in seconds (when not using semantic mode) |

## Usage

1. **Start Real-time** — Click the mic button, grant permission, and speak
2. **Select Model** — Choose from the dropdown (models load on-demand)
3. **Language** — Pick English, Arabic, or Auto-detect
4. **Chunk Mode** — "Semantic (auto turn)" uses VAD; fixed durations (2s–8s) also available
5. **TTS Toggle** — 🔊/🔇 button auto-speaks translations
6. **Upload File** — Click 📁 or drag-and-drop any audio file

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/models` | List available models and current selection |
| `POST` | `/switch-model` | Switch ASR model (`{"model": "large-v3"}`) |
| `POST` | `/transcribe` | Upload audio file for transcription |
| `POST` | `/tts` | Generate TTS audio (`{"text": "...", "lang": "en"}`) |
| `WS` | `/ws/stream` | Real-time audio streaming WebSocket |

## WebSocket Protocol

**Client → Server:**
- Binary: Raw 16-bit PCM audio at 16kHz
- JSON: `{"type": "config", "language": "en", "chunk_duration": "semantic"}`
- JSON: `{"type": "stop"}` / `{"type": "reset"}`

**Server → Client:**
- `{"type": "vad", "state": "speaking" | "processing" | "idle"}`
- `{"type": "result", "text": "...", "translated": "...", "language": "en", "target_language": "ar", "segments": [...]}`

## VRAM Usage

Typical GPU memory with `large-v3` + NLLB-200 + Kokoro + VAD:

| Component | VRAM |
|-----------|------|
| Whisper large-v3 | ~6 GB |
| NLLB-200-distilled-600M | ~2.5 GB |
| Kokoro-82M | ~0.5 GB |
| Silero VAD | ~0.01 GB |
| **Total** | **~9 GB** |

Use `tiny` or `base` models to run on GPUs with less memory.

## License

[MIT](LICENSE)
