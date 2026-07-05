"""
Whisper ASR – Real-time streaming speech recognition
  • WebSocket streams mic audio from browser
  • Noise cancellation (spectral gating + high-pass filter)
  • Faster Whisper GPU transcription on rolling audio buffer
  • Multiple model support (faster-whisper models + ArabicSpeech/Octopus)
  • NLLB-200 neural machine translation (EN⇄AR)
  • Kokoro-82M text-to-speech for English output
"""

import os

import asyncio
import io
import json as json_mod
import struct
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import torch


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor

# ── Noise cancellation imports ───────────────────────────────────────
try:
    import noisereduce as nr
    import torchaudio
    NC_AVAILABLE = True
except ImportError:
    NC_AVAILABLE = False
    print("[WARN] Noise cancellation not available")
    print("       Install with: pip install noisereduce torch torchaudio")

app = FastAPI(title="Whisper ASR")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── Available models ─────────────────────────────────────────────────
WHISPER_MODELS = {
    "tiny":           "Tiny (~1GB VRAM, fastest)",
    "base":           "Base (~1GB VRAM)",
    "small":          "Small (~2GB VRAM)",
    "medium":         "Medium (~5GB VRAM)",
    "large-v2":       "Large v2 (~10GB VRAM)",
    "large-v3":       "Large v3 (~6GB VRAM, best Whisper accuracy)",
    "large-v3-turbo": "Large v3 Turbo (~4.5GB VRAM, fast + accurate)",
}

# HuggingFace Whisper models (loaded via transformers, not faster-whisper)
HF_WHISPER_MODELS = {
    "arabic-v3-turbo": {
        "name": "Arabic-finetuned Turbo (WER 31, best Arabic)",
        "repo": "mboushaba/whisper-large-v3-turbo-arabic",
    },
    "arabic-cv11": {
        "name": "Arabic CV11 Finetuned (WER 12.6 on CV)",
        "repo": "KalamTech/whisper-large-arabic-cv-11",
    },
}

# ── Config ───────────────────────────────────────────────────────────
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "float16")
SAMPLE_RATE = 16000
CHUNK_DURATION = float(os.getenv("CHUNK_DURATION", "3"))

# ── Translation config ───────────────────────────────────────────────
NLLB_MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_CACHE_DIR = os.getenv("NLLB_CACHE", os.path.join(Path.home(), ".cache", "huggingface"))
nllb_model = None
nllb_tokenizer = None

# Language codes for NLLB
NLLB_LANGS = {"en": "eng_Latn", "ar": "arb_Arab"}

# ── TTS config ────────────────────────────────────────────────────────
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_SAMPLE_RATE = 24000
AR_TTS_VOICE = os.getenv("AR_TTS_VOICE", "ar-SA-HamedNeural")
kokoro_pipeline = None

# ── Semantic turn detection config ────────────────────────────────────
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.40"))
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "1400"))   # ms of silence to end a turn
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "1000"))        # min speech before allowing turn end
MAX_TURN_DURATION = float(os.getenv("MAX_TURN_DURATION", "30"))  # safety cap (seconds)
PRE_ROLL_MS = 300   # ms of audio to keep before speech onset
VAD_DEBOUNCE = 3    # consecutive silence frames needed to start silence counter
vad_model = None

# ── Model state ──────────────────────────────────────────────────────
current_model_name: str = os.getenv("WHISPER_MODEL", "large-v3")
whisper_model: WhisperModel | None = None
hf_whisper_model = None       # transformers WhisperForConditionalGeneration
hf_whisper_processor = None   # transformers WhisperProcessor
octopus_model = None
octopus_processor = None
model_lock = threading.Lock()


def _load_whisper(name: str):
    global whisper_model, hf_whisper_model, hf_whisper_processor
    global octopus_model, octopus_processor, current_model_name
    print(f"  Loading Faster Whisper '{name}' on {DEVICE} ({COMPUTE_TYPE})…")
    whisper_model = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
    hf_whisper_model = None
    hf_whisper_processor = None
    octopus_model = None
    octopus_processor = None
    current_model_name = name
    print(f"  ✓ {name} ready")


def _load_hf_whisper(model_id: str, repo: str):
    global hf_whisper_model, hf_whisper_processor, whisper_model
    global octopus_model, octopus_processor, current_model_name
    print(f"  Loading HF Whisper '{repo}' on {DEVICE} (float16)…")
    hf_whisper_processor = WhisperProcessor.from_pretrained(
        repo, cache_dir=NLLB_CACHE_DIR,
    )
    hf_whisper_model = WhisperForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.float16, cache_dir=NLLB_CACHE_DIR,
    ).cuda()
    hf_whisper_model.eval()
    whisper_model = None
    octopus_model = None
    octopus_processor = None
    current_model_name = model_id
    print(f"  ✓ {model_id} ready")


def _load_octopus():
    global octopus_model, octopus_processor, whisper_model, current_model_name

    octopus_dir = Path(__file__).parent / "octopus_repo"

    # Clone repo if needed
    if not octopus_dir.exists():
        print("  Cloning ArabicSpeech/Octopus from HuggingFace…")
        subprocess.run(
            ["git", "clone", "https://huggingface.co/ArabicSpeech/Octopus", str(octopus_dir)],
            check=True,
        )
        print("  ✓ Octopus repo cloned")

    # Ensure extra deps
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "omegaconf", "peft", "sentencepiece", "accelerate"],
        check=True,
    )

    if str(octopus_dir) not in sys.path:
        sys.path.insert(0, str(octopus_dir))

    print("  Loading Octopus model…")
    from omegaconf import OmegaConf
    from transformers import WhisperFeatureExtractor
    from models.tinyoctopus import TINYOCTOPUS

    cfg = OmegaConf.load(str(octopus_dir / "decode_config.yaml"))
    model = TINYOCTOPUS.from_config(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    octopus_model = model
    octopus_processor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3")
    whisper_model = None
    current_model_name = "octopus"
    print("  ✓ Octopus ready")


def _load_nllb():
    global nllb_model, nllb_tokenizer
    print(f"  Loading NLLB-200 translator '{NLLB_MODEL_NAME}'…")
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, cache_dir=NLLB_CACHE_DIR)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_NAME,
        torch_dtype=torch.float16,
        cache_dir=NLLB_CACHE_DIR,
    ).cuda()
    nllb_model.eval()
    print("  ✓ NLLB-200 ready")


def _load_kokoro():
    global kokoro_pipeline
    print(f"  Loading Kokoro-82M TTS (voice={TTS_VOICE})…")
    from kokoro import KPipeline
    kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    print("  ✓ Kokoro TTS ready")


def _load_vad():
    global vad_model
    print("  Loading Silero VAD…")
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
    )
    vad_model.eval()
    print("  ✓ Silero VAD ready")


class TurnDetector:
    """VAD-based semantic turn detector.

    Accumulates audio and uses Silero VAD to detect when the speaker
    has finished a natural turn (pause ≥ TURN_SILENCE_MS after speech).
    Returns the complete utterance for downstream processing.
    """

    VAD_FRAME = 512  # Silero needs 512 samples at 16 kHz (32 ms)

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.silence_limit = int(TURN_SILENCE_MS * sample_rate / 1000)
        self.min_speech = int(MIN_SPEECH_MS * sample_rate / 1000)
        self.max_samples = int(MAX_TURN_DURATION * sample_rate)
        self.pre_roll = int(PRE_ROLL_MS * sample_rate / 1000)
        self.reset()

    def reset(self):
        self.speech_buf: list[np.ndarray] = []
        self.pre_roll_buf: list[np.ndarray] = []
        self.pre_roll_n = 0
        self.silence_n = 0
        self.speech_n = 0
        self.voiced_n = 0          # frames actually classified as speech (excl. silence padding)
        self.consec_silence = 0    # consecutive silence frames for debouncing
        self.is_speaking = False
        self._pending = np.array([], dtype=np.float32)
        if vad_model is not None:
            vad_model.reset_states()

    # ------------------------------------------------------------------ #
    def feed(self, audio_np: np.ndarray) -> np.ndarray | None:
        """Feed PCM float32 samples.  Returns a complete turn or None."""
        self._pending = np.concatenate([self._pending, audio_np])

        while len(self._pending) >= self.VAD_FRAME:
            frame = self._pending[: self.VAD_FRAME]
            self._pending = self._pending[self.VAD_FRAME :]

            prob = float(
                vad_model(torch.from_numpy(frame), self.sr).item()
            )
            is_speech = prob >= VAD_THRESHOLD

            if is_speech:
                self.consec_silence = 0
                if not self.is_speaking:
                    # Speech onset – prepend pre-roll
                    self.is_speaking = True
                    if self.pre_roll_buf:
                        self.speech_buf = list(self.pre_roll_buf)
                        self.speech_n = self.pre_roll_n
                    self.pre_roll_buf.clear()
                    self.pre_roll_n = 0
                self.speech_buf.append(frame)
                self.speech_n += len(frame)
                self.voiced_n += len(frame)
                self.silence_n = 0
            else:
                self.consec_silence += 1
                if self.is_speaking:
                    self.speech_buf.append(frame)
                    self.speech_n += len(frame)
                    # Only start counting silence after debounce threshold
                    if self.consec_silence >= VAD_DEBOUNCE:
                        self.silence_n += len(frame)
                    # End turn only if enough voiced speech AND enough silence
                    if (self.silence_n >= self.silence_limit
                            and self.voiced_n >= self.min_speech):
                        return self._flush()
                else:
                    # Idle – maintain rolling pre-roll
                    self.pre_roll_buf.append(frame)
                    self.pre_roll_n += len(frame)
                    while self.pre_roll_n > self.pre_roll:
                        removed = self.pre_roll_buf.pop(0)
                        self.pre_roll_n -= len(removed)

            # Safety cap
            if self.speech_n >= self.max_samples:
                return self._flush()

        return None

    # ------------------------------------------------------------------ #
    def _flush(self) -> np.ndarray | None:
        if not self.speech_buf:
            self.reset()
            return None
        result = np.concatenate(self.speech_buf)
        # keep state but clear buffers
        self.speech_buf.clear()
        self.pre_roll_buf.clear()
        self.pre_roll_n = 0
        self.silence_n = 0
        self.speech_n = 0
        self.voiced_n = 0
        self.consec_silence = 0
        self.is_speaking = False
        if vad_model is not None:
            vad_model.reset_states()
        return result

    def flush_remaining(self) -> np.ndarray | None:
        """Flush leftover audio (called on stop). Needs ≥ 0.5 s of speech."""
        if self.speech_buf and self.speech_n > self.sr // 2:
            return self._flush()
        self.reset()
        return None


# ── Load models at startup ───────────────────────────────────────────
print(f"[1/5] Loading ASR model '{current_model_name}'…")
_load_whisper(current_model_name)

print("[2/5] Loading NLLB-200 translation model…")
_load_nllb()

print("[3/5] Loading Kokoro-82M TTS…")
_load_kokoro()

print("[4/5] Loading Silero VAD for turn detection…")
_load_vad()

if NC_AVAILABLE:
    print("[5/5] Noise cancellation ready (noisereduce + spectral gating)")
else:
    print("[5/5] Skipped noise cancellation (not installed)")


# ── Helpers ──────────────────────────────────────────────────────────
def denoise_audio(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """Apply multi-stage noise cancellation (spectral gating + high-pass filter)."""
    if not NC_AVAILABLE:
        return audio_np

    # Skip denoising if audio is too short or silent
    if len(audio_np) < 1024:
        return audio_np
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-6:
        return audio_np

    audio_float = audio_np.astype(np.float32)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Stage 1: Spectral gating noise reduction (non-stationary)
        try:
            cleaned = nr.reduce_noise(
                y=audio_float,
                sr=sr,
                stationary=False,
                prop_decrease=0.95,
                n_fft=min(2048, len(audio_float)),
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            )
        except Exception:
            cleaned = audio_float

        # Stage 2: Stationary noise (fans, hum, hiss)
        try:
            cleaned = nr.reduce_noise(
                y=cleaned,
                sr=sr,
                stationary=True,
                prop_decrease=0.8,
            )
        except Exception:
            pass

    # Stage 3: High-pass filter at 80 Hz to remove rumble
    from scipy.signal import butter, sosfilt
    sos = butter(5, 80, btype='highpass', fs=sr, output='sos')
    cleaned = sosfilt(sos, cleaned).astype(np.float32)

    return cleaned


def transcribe_buffer(audio_np: np.ndarray, language: str):
    """Transcribe a numpy audio buffer with the active model."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"whisper_{id(audio_np)}_{time.time_ns()}.wav")
    try:
        sf.write(tmp_path, audio_np, SAMPLE_RATE)

        # ── Octopus path ─────────────────────────────────────────
        if octopus_model is not None:
            return _transcribe_octopus(tmp_path, language)

        # ── HF Whisper path (Arabic-finetuned models) ────────────
        if hf_whisper_model is not None:
            return _transcribe_hf_whisper(audio_np, language)

        # ── Faster Whisper path ──────────────────────────────────
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language or None,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        parts = []
        seg_list = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                parts.append(text)
                seg_list.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                })
        return " ".join(parts), seg_list
    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass  # Windows file locking; will be cleaned up later


def pcm_bytes_to_np(raw_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM LE bytes to float32 numpy array."""
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using NLLB-200."""
    if not text.strip() or nllb_model is None:
        return ""
    src_code = NLLB_LANGS.get(source_lang, "eng_Latn")
    tgt_code = NLLB_LANGS.get(target_lang, "arb_Arab")
    nllb_tokenizer.src_lang = src_code
    inputs = nllb_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(nllb_model.device) for k, v in inputs.items()}
    tgt_token_id = nllb_tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.no_grad():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_new_tokens=1024,
        )
    return nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_tts(text: str) -> bytes | None:
    """Generate TTS audio for English text via Kokoro, return WAV bytes."""
    if not text.strip() or kokoro_pipeline is None:
        return None
    chunks = []
    for _gs, _ps, audio in kokoro_pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED):
        chunks.append(audio)
    if not chunks:
        return None
    full_audio = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, full_audio, TTS_SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def generate_tts_arabic(text: str) -> bytes | None:
    """Generate TTS audio for Arabic text via Edge TTS, return MP3 bytes."""
    if not text.strip():
        return None
    import edge_tts
    buf = io.BytesIO()
    async def _run():
        comm = edge_tts.Communicate(text, AR_TTS_VOICE)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
    # Run in a fresh event loop since we're in a thread executor
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
    data = buf.getvalue()
    return data if data else None


def _transcribe_hf_whisper(audio_np: np.ndarray, language: str) -> tuple[str, list]:
    """Transcribe using a HuggingFace Whisper model (Arabic-finetuned)."""
    inputs = hf_whisper_processor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt",
    )
    input_features = inputs.input_features.to(hf_whisper_model.device, dtype=torch.float16)

    with torch.no_grad():
        predicted_ids = hf_whisper_model.generate(
            input_features,
            language=language or "ar",
            task="transcribe",
            max_new_tokens=440,
        )

    text = hf_whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True,
    )[0].strip()

    seg_list = [{"start": 0.0, "end": 0.0, "text": text}] if text else []
    return text, seg_list


def _transcribe_octopus(wav_path: str, language: str) -> tuple[str, list]:
    """Run Octopus model inference on a WAV file."""
    from utils import prepare_one_sample

    # Pick task based on language direction
    if language == "ar":
        task = "asr"  # Arabic ASR
    else:
        task = "translation"  # Translate Arabic audio → English

    task_prompts = {
        "asr": "تعرف على الكلام وأعطني النص.",
        "translation": "الرجاء ترجمة هذا المقطع الصوتي إلى اللغة الإنجليزية.",
        "dialect": "What is the dialect of the speaker?",
    }
    prompt_text = task_prompts[task]
    samples = prepare_one_sample(wav_path, octopus_processor)
    prompt = [f"<Speech><SpeechHere></Speech> {prompt_text}"]

    text = octopus_model.generate(samples, {"temperature": 0.7}, prompts=prompt)[0]
    text = text.replace("<s>", "").replace("</s>", "").strip()

    seg_list = [{"start": 0.0, "end": 0.0, "text": text}] if text else []
    return text, seg_list


# ── HTTP routes ──────────────────────────────────────────────────────
@app.get("/models")
async def list_models():
    """Return available models and which is active."""
    models = []
    for key, desc in WHISPER_MODELS.items():
        models.append({"id": key, "name": desc, "active": key == current_model_name})
    for key, info in HF_WHISPER_MODELS.items():
        models.append({"id": key, "name": info["name"], "active": key == current_model_name})
    models.append({
        "id": "octopus",
        "name": "Octopus – ArabicSpeech (Arabic ASR + Translation LLM)",
        "active": current_model_name == "octopus",
    })
    return JSONResponse({"models": models, "current": current_model_name})


@app.post("/switch-model")
async def switch_model(request: Request):
    """Switch the active model. Unloads the old model first."""
    body = await request.json()
    model_id = body.get("model", "").strip()

    if not model_id:
        return JSONResponse({"error": "No model specified"}, status_code=400)
    if model_id == current_model_name:
        return JSONResponse({"status": "already_loaded", "model": model_id})
    all_known = set(WHISPER_MODELS) | set(HF_WHISPER_MODELS) | {"octopus"}
    if model_id not in all_known:
        return JSONResponse({"error": f"Unknown model: {model_id}"}, status_code=400)

    loop = asyncio.get_event_loop()

    def _do_switch():
        with model_lock:
            if model_id == "octopus":
                _load_octopus()
            elif model_id in HF_WHISPER_MODELS:
                _load_hf_whisper(model_id, HF_WHISPER_MODELS[model_id]["repo"])
            else:
                _load_whisper(model_id)

    await loop.run_in_executor(None, _do_switch)
    return JSONResponse({"status": "ok", "model": current_model_name})


@app.post("/tts")
async def tts_endpoint(request: Request):
    """Generate TTS audio from text. Returns WAV (English) or MP3 (Arabic)."""
    body = await request.json()
    text = body.get("text", "").strip()
    lang = body.get("lang", "en")
    if not text:
        return JSONResponse({"error": "No text"}, status_code=400)

    loop = asyncio.get_event_loop()

    if lang == "ar":
        audio_bytes = await loop.run_in_executor(None, generate_tts_arabic, text)
        if audio_bytes is None:
            return JSONResponse({"error": "Arabic TTS failed"}, status_code=500)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    else:
        wav_bytes = await loop.run_in_executor(None, generate_tts, text)
        if wav_bytes is None:
            return JSONResponse({"error": "TTS generation failed"}, status_code=500)
        return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"df_available": NC_AVAILABLE},
    )


@app.post("/transcribe")
async def transcribe_file(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    """File upload endpoint (non-realtime fallback)."""
    suffix = Path(audio.filename or "audio.wav").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await audio.read()
        tmp.write(content)
        tmp.close()

        # Read audio
        audio_np, sr = sf.read(tmp.name, dtype="float32")
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)  # mono

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            import torchaudio, torch
            tensor = torch.from_numpy(audio_np).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE)
            audio_np = tensor.squeeze(0).numpy()

        # Noise cancellation
        audio_np = denoise_audio(audio_np, SAMPLE_RATE)

        original_text, segments = transcribe_buffer(audio_np, language)

        target = "ar" if language != "ar" else "en"
        translated = translate_text(original_text, language, target) if original_text.strip() else ""

        return JSONResponse({
            "language": language,
            "target_language": target,
            "text": original_text,
            "translated": translated,
            "segments": segments,
            "noise_cancelled": NC_AVAILABLE,
        })
    finally:
        os.unlink(tmp.name)


# ── WebSocket for real-time streaming ────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Real-time streaming:
    1. Browser sends raw 16-bit PCM audio chunks via WebSocket
    2. Server accumulates audio, applies noise cancellation
    3. Uses either fixed-duration chunks OR semantic turn detection (Silero VAD)
    4. Sends back transcription + translation as JSON
    """
    await ws.accept()

    audio_buffer = bytearray()
    language = "en"
    samples_needed = int(SAMPLE_RATE * CHUNK_DURATION) * 2  # 2 bytes per int16 sample
    use_turn_detection = False
    turn_detector: TurnDetector | None = None

    try:
        while True:
            data = await ws.receive()

            # Text message = control command
            if "text" in data:
                msg = json_mod.loads(data["text"])
                if msg.get("type") == "config":
                    language = msg.get("language", "en")
                    chunk_mode = msg.get("chunk_duration", "3")
                    if chunk_mode == "semantic":
                        use_turn_detection = True
                        turn_detector = TurnDetector(SAMPLE_RATE)
                    else:
                        use_turn_detection = False
                        turn_detector = None
                        samples_needed = int(SAMPLE_RATE * float(chunk_mode)) * 2
                    continue
                if msg.get("type") == "stop":
                    if use_turn_detection and turn_detector:
                        remaining = turn_detector.flush_remaining()
                        if remaining is not None:
                            result = await _process_audio(remaining, language)
                            if result:
                                await ws.send_json(result)
                        turn_detector.reset()
                    else:
                        if len(audio_buffer) > SAMPLE_RATE:  # at least 0.5s
                            result = await _process_chunk(bytes(audio_buffer), language)
                            if result:
                                await ws.send_json(result)
                        audio_buffer.clear()
                    await ws.send_json({"type": "stopped"})
                    continue
                if msg.get("type") == "reset":
                    audio_buffer.clear()
                    if turn_detector:
                        turn_detector.reset()
                    continue

            # Binary message = audio data
            if "bytes" in data:
                if use_turn_detection and turn_detector:
                    # ── Semantic turn detection path ──────────────
                    was_speaking = turn_detector.is_speaking
                    audio_np = pcm_bytes_to_np(data["bytes"])
                    turn_audio = turn_detector.feed(audio_np)

                    # Send VAD state changes so UI can show feedback
                    if turn_detector.is_speaking and not was_speaking:
                        await ws.send_json({"type": "vad", "state": "speaking"})
                    elif not turn_detector.is_speaking and was_speaking and turn_audio is None:
                        # silence detected but not yet long enough
                        pass

                    if turn_audio is not None:
                        dur = len(turn_audio) / SAMPLE_RATE
                        await ws.send_json({"type": "vad", "state": "processing",
                                            "duration": round(dur, 1)})
                        result = await _process_audio(turn_audio, language)
                        if result:
                            await ws.send_json(result)
                        await ws.send_json({"type": "vad", "state": "idle"})
                else:
                    # ── Fixed-duration path ───────────────────────
                    audio_buffer.extend(data["bytes"])
                    if len(audio_buffer) >= samples_needed:
                        chunk = bytes(audio_buffer)
                        audio_buffer.clear()
                        result = await _process_chunk(chunk, language)
                        if result:
                            await ws.send_json(result)

    except (WebSocketDisconnect, RuntimeError):
        pass


async def _process_audio(audio_np: np.ndarray, language: str) -> dict | None:
    """Denoise and transcribe a float32 numpy audio array (from turn detector)."""
    loop = asyncio.get_event_loop()

    def _sync():
        clean = denoise_audio(audio_np, SAMPLE_RATE)
        text, segments = transcribe_buffer(clean, language)
        if not text.strip():
            return None
        target = "ar" if language != "ar" else "en"
        translated = translate_text(text, language, target)
        return {
            "type": "result",
            "text": text,
            "translated": translated,
            "language": language,
            "target_language": target,
            "segments": segments,
            "noise_cancelled": NC_AVAILABLE,
        }

    return await loop.run_in_executor(None, _sync)


async def _process_chunk(raw_pcm: bytes, language: str) -> dict | None:
    """Denoise and transcribe a PCM audio chunk."""
    loop = asyncio.get_event_loop()

    def _sync_process():
        audio_np = pcm_bytes_to_np(raw_pcm)

        # Noise cancellation
        clean_audio = denoise_audio(audio_np, SAMPLE_RATE)

        text, segments = transcribe_buffer(clean_audio, language)
        if not text.strip():
            return None

        target = "ar" if language != "ar" else "en"
        translated = translate_text(text, language, target)

        return {
            "type": "result",
            "text": text,
            "translated": translated,
            "language": language,
            "target_language": target,
            "segments": segments,
            "noise_cancelled": NC_AVAILABLE,
        }

    return await loop.run_in_executor(None, _sync_process)
