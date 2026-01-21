"""
AiTranscribe Backend Server
===========================

This is a FastAPI server that handles speech-to-text transcription.

WHAT IS FASTAPI?
----------------
FastAPI is a modern Python web framework for building APIs (Application Programming Interfaces).
An API is like a waiter in a restaurant:
- You (the client/app) make a request ("I want transcription")
- The API (waiter) takes your request to the kitchen (the model)
- The kitchen processes it and gives back the result
- The API returns the result to you

WHY FASTAPI?
------------
1. Fast - One of the fastest Python frameworks
2. Easy - Automatic documentation, type hints
3. Modern - Async support, great for our use case

HOW THIS SERVER WORKS:
----------------------
1. When server starts -> Load the ASR model (heavy, ~3GB)
2. Model stays in memory (don't reload for each request)
3. Swift app sends audio -> Server transcribes -> Returns text
4. This is called a "microservice architecture"

ENDPOINTS (URLs the app can call):
----------------------------------
- GET  /health     -> Check if server is running
- GET  /status     -> Check if model is loaded
- POST /load       -> Load the model into memory
- POST /unload     -> Remove model from memory
- POST /transcribe -> Send audio, get text back

To run this server:
    cd backend
    uvicorn server:app --host 127.0.0.1 --port 8765

Then visit http://127.0.0.1:8765/docs to see automatic API documentation!
"""

# ============================================================================
# IMPORTS - Libraries we need
# ============================================================================

# FastAPI - The web framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Pydantic - For data validation (ensures data is correct type)
from pydantic import BaseModel

# Standard library
import os
import sys
import tempfile
import time

# Force unbuffered stdout so logs appear immediately in Swift's debug console
# This is important because Python buffers stdout by default when not connected to a TTY
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

# Model management
from model_manager import (
    AVAILABLE_MODELS,
    ModelType,
    get_storage_dir,
    get_whisper_dir,
    get_model_info,
    get_all_models_info,
    get_storage_info,
    check_model_downloaded,
    download_model,
    delete_model,
    check_nemo_available,
    is_nemo_model,
)

# Audio processing
import numpy as np
from scipy.io.wavfile import write as wav_write

# ============================================================================
# CONFIGURATION - Settings for our server
# ============================================================================

# Model configuration is now in model_manager.py
# AVAILABLE_MODELS is imported from there

# Default model
DEFAULT_MODEL = "parakeet-v2"

# Audio settings (must match what we record)
SAMPLE_RATE = 16000  # 16kHz - standard for speech recognition

# Server settings
HOST = "127.0.0.1"  # localhost - only accessible from this computer
PORT = 8765         # The port number (like a door number for the server)


# ============================================================================
# GLOBAL STATE - Variables that persist across requests
# ============================================================================

# We store the model here so it stays in memory
# None means "not loaded yet"
asr_model = None
model_device = None
current_model_id = None  # Which model is loaded (e.g., "parakeet-v2")

# Audio recorder instance
from recorder import AudioRecorder, list_audio_devices, get_default_input_device
audio_recorder = AudioRecorder()

# Streaming state
streaming_active = False
streaming_stop_requested = False


# ============================================================================
# PYDANTIC MODELS - Define the shape of data we send/receive
# ============================================================================

class StatusResponse(BaseModel):
    """
    What we return when someone asks for status.

    Pydantic models are like templates - they define what fields
    a response should have and what types they should be.
    """
    status: str                    # "ready", "loading", "not_loaded"
    model_loaded: bool             # True/False
    model_id: Optional[str]        # e.g., "parakeet-v2", "parakeet-v3"
    model_name: Optional[str]      # Full model name
    device: Optional[str]          # "cpu", "mps", etc.


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str                        # e.g., "parakeet-v2"
    name: str                      # Full HuggingFace name
    display_name: str              # Human-readable name
    author: str                    # e.g., "NVIDIA", "OpenAI"
    type: str                      # "nemo" or "whisper"
    languages: list[str]           # Language codes
    language_names: list[str]      # Human-readable language names
    description: str
    multilingual: bool
    size_mb: int                   # Download size
    ram_mb: int                    # RAM when loaded
    streaming_native: bool         # Is model optimized for streaming?
    downloaded: bool               # Is model downloaded?
    download_url: Optional[str]    # URL for manual download (Whisper only)
    path: Optional[str]            # Path to model file (if downloaded)
    nemo_required: bool            # Does this model require NeMo to be installed?
    can_use: bool = True           # Can this model be used? (False if NeMo required but not available)


class LoadModelRequest(BaseModel):
    """Request to load a specific model."""
    model_id: str = DEFAULT_MODEL  # Which model to load


class TranscriptionResponse(BaseModel):
    """What we return after transcription."""
    text: str                      # The transcribed text
    duration_seconds: float        # How long the audio was
    processing_time: float         # How long transcription took
    realtime_factor: float         # Speed (e.g., 10x = 10x faster than realtime)


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool


class AudioDeviceInfo(BaseModel):
    """Information about an audio device."""
    id: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool


class RecordingStatus(BaseModel):
    """Current recording status."""
    is_recording: bool
    duration_seconds: float
    volume: float  # 0.0 to 1.0 (roughly)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

# Whisper model instance (separate from NeMo)
whisper_model = None
current_model_type = None  # "nemo" or "whisper"

# Cached NeMo status (checked once at startup or on first request)
_nemo_status_cache = None

def get_cached_nemo_status() -> dict:
    """Get cached NeMo status (full dict)."""
    global _nemo_status_cache
    if _nemo_status_cache is None:
        print("DEBUG: Checking NeMo availability (first time)...")
        _nemo_status_cache = check_nemo_available()
        print(f"DEBUG: NeMo available: {_nemo_status_cache['available']}")
    return _nemo_status_cache

def get_nemo_available() -> bool:
    """Get cached NeMo availability status (bool only)."""
    return get_cached_nemo_status()["available"]


def load_asr_model(model_id: str = DEFAULT_MODEL):
    """
    Load the ASR (Automatic Speech Recognition) model.

    This is the "heavy" operation - takes a few seconds and uses RAM.
    We only do this once when the server starts or when explicitly requested.

    Supports both NeMo (Parakeet) and Whisper models.

    Args:
        model_id: Which model to load (e.g., "parakeet-v2", "whisper-base-en")
    """
    global asr_model, whisper_model, model_device, current_model_id, current_model_type

    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")

    model_config = AVAILABLE_MODELS[model_id]
    model_type = model_config["type"]

    print(f"Loading ASR model: {model_config['display_name']}")
    start_time = time.time()

    if model_type == ModelType.NEMO:
        # Load NeMo model (Parakeet)
        _load_nemo_model(model_id, model_config)

    elif model_type == ModelType.WHISPER:
        # Load Whisper model
        _load_whisper_model(model_id, model_config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    current_model_id = model_id
    current_model_type = model_type.value if isinstance(model_type, ModelType) else model_type

    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.1f}s on {model_device}")


def _load_nemo_model(model_id: str, model_config: dict):
    """Load a NeMo/Parakeet model."""
    global asr_model, model_device

    import torch
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    model_name = model_config["name"]

    # Load the pretrained model from NVIDIA
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Configure decoding strategy
    # "greedy_batch" = fast, picks most likely word at each step
    if hasattr(model, 'change_decoding_strategy'):
        decoding_cfg = OmegaConf.create({
            'strategy': 'greedy_batch',
            'greedy': {'max_symbols': 10, 'use_cuda_graph_decoder': False}
        })
        model.change_decoding_strategy(decoding_cfg)

    # Set to evaluation mode (not training)
    model.eval()

    # Use CPU (MPS has float64 issues with NeMo)
    model = model.to("cpu")
    model_device = "cpu"

    asr_model = model


def _load_whisper_model(model_id: str, model_config: dict):
    """Load a Whisper model using faster-whisper (CTranslate2)."""
    global whisper_model, model_device

    from faster_whisper import WhisperModel

    # Get model name for faster-whisper
    # This can be a size like "base.en", "small.en", "large-v3"
    # or a full HuggingFace path like "deepdml/faster-whisper-large-v3-turbo-ct2"
    model_name = model_config["name"]

    # Load the model (faster-whisper auto-downloads from HuggingFace if needed)
    # Use CPU with int8 for best compatibility on Mac
    whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    model_device = "cpu"


def unload_asr_model():
    """
    Unload the model to free memory.

    Python has garbage collection, but explicitly deleting
    and clearing CUDA/MPS cache ensures memory is freed.
    """
    global asr_model, whisper_model, model_device, current_model_id, current_model_type

    if asr_model is not None:
        del asr_model
        asr_model = None

    if whisper_model is not None:
        del whisper_model
        whisper_model = None

    model_device = None
    current_model_id = None
    current_model_type = None

    # Clear any cached memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Force garbage collection
    import gc
    gc.collect()

    print("Model unloaded")


def transcribe_audio(audio_path: str) -> tuple[str, float]:
    """
    Transcribe an audio file.

    Supports both NeMo and Whisper models.

    Args:
        audio_path: Path to a WAV file

    Returns:
        Tuple of (transcribed_text, processing_time)
    """
    global asr_model, whisper_model, current_model_type

    if current_model_type == "nemo":
        return _transcribe_nemo(audio_path)
    elif current_model_type == "whisper":
        return _transcribe_whisper(audio_path)
    else:
        raise RuntimeError("No model loaded")


def _transcribe_nemo(audio_path: str) -> tuple[str, float]:
    """Transcribe using NeMo model."""
    global asr_model

    if asr_model is None:
        raise RuntimeError("NeMo model not loaded")

    import torch

    start_time = time.time()

    # torch.no_grad() tells PyTorch we're not training
    # This saves memory and speeds up inference
    with torch.no_grad():
        result = asr_model.transcribe([audio_path])[0]

    processing_time = time.time() - start_time

    # Extract text from result
    if hasattr(result, 'text'):
        text = result.text
    else:
        text = str(result)

    return text, processing_time


def _transcribe_whisper(audio_path: str) -> tuple[str, float]:
    """Transcribe using faster-whisper model."""
    global whisper_model

    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded")

    start_time = time.time()

    # Transcribe with faster-whisper
    # Returns (segments_generator, transcription_info)
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)

    # Collect all segment texts
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    processing_time = time.time() - start_time

    # Join all segments into final text
    text = " ".join(text_parts)

    return text.strip(), processing_time


def is_model_loaded() -> bool:
    """Check if any model is currently loaded."""
    return asr_model is not None or whisper_model is not None


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events - code that runs when server starts/stops.

    This is a "context manager" (the 'async with' pattern).
    - Code before 'yield' runs on startup
    - Code after 'yield' runs on shutdown
    """
    # STARTUP
    print("="*50)
    print("AiTranscribe Server Starting")
    print("="*50)
    print("Server ready. No model loaded yet.")
    print("Model will be loaded when requested by the app.")

    # NOTE: Auto-load disabled for lower memory footprint
    # The Swift app will call POST /load when needed
    # (controlled by "Load model on app launch" setting)

    yield  # Server runs here

    # SHUTDOWN
    print("Server shutting down...")
    unload_asr_model()


# Create the FastAPI app
app = FastAPI(
    title="AiTranscribe API",
    description="Speech-to-text transcription service using NVIDIA Nemotron",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware
# CORS = Cross-Origin Resource Sharing
# This allows our Swift app (different "origin") to talk to the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],      # Allow all HTTP methods
    allow_headers=["*"],      # Allow all headers
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    @app.get("/health") is a "decorator" - it tells FastAPI:
    "When someone visits GET /health, run this function"

    Used by other services to check if this server is alive.
    """
    return {"status": "healthy", "service": "AiTranscribe"}


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get the current status of the transcription service.

    response_model=StatusResponse tells FastAPI to validate
    our response matches the StatusResponse shape.
    """
    model_name = None
    if current_model_id and current_model_id in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[current_model_id]["name"]

    return StatusResponse(
        status="ready" if is_model_loaded() else "not_loaded",
        model_loaded=is_model_loaded(),
        model_id=current_model_id,
        model_name=model_name,
        device=model_device
    )


@app.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """
    List all available ASR models.

    Returns information about each model including supported languages
    and download status.
    """
    models_info = get_all_models_info()
    nemo_available = get_nemo_available()

    # DEBUG: Log all models and their downloaded status
    print("=" * 60)
    print("DEBUG: /models endpoint called")
    print(f"DEBUG: Total models: {len(models_info)}, NeMo available: {nemo_available}")
    for info in models_info:
        print(f"  - {info['id']}: downloaded={info['downloaded']}")
    print("=" * 60)

    result = []
    for info in models_info:
        requires_nemo = is_nemo_model(info["id"])
        # Model can be used if it doesn't require NeMo, OR if NeMo is available
        can_use = not requires_nemo or nemo_available

        result.append(ModelInfo(
            id=info["id"],
            name=info["name"],
            display_name=info["display_name"],
            author=info["author"],
            type=info["type"],
            languages=info["languages"],
            language_names=info["language_names"],
            description=info["description"],
            multilingual=info["multilingual"],
            size_mb=info["size_mb"],
            ram_mb=info["ram_mb"],
            streaming_native=info.get("streaming_native", False),
            downloaded=info["downloaded"],
            download_url=info.get("download_url"),
            path=info.get("path"),
            nemo_required=requires_nemo,
            can_use=can_use,
        ))

    return result


# ============================================================================
# NEMO STATUS ENDPOINT
# ============================================================================

class NemoStatusResponse(BaseModel):
    """NeMo availability status."""
    nemo_available: bool
    nemo_version: Optional[str]
    torch_version: Optional[str]
    device: str
    backend_mode: str


@app.get("/nemo/status", response_model=NemoStatusResponse)
async def get_nemo_status() -> NemoStatusResponse:
    """
    Check if NeMo is available and get version info.

    Returns:
    - nemo_available: Whether NeMo can be imported
    - nemo_version: NeMo version string if available
    - torch_version: PyTorch version string if available
    - device: Available device (cpu, mps, cuda)
    - backend_mode: Current backend mode (pyinstaller, nemo_venv, development)

    Note: Uses cached status to avoid slow repeated import attempts.
    """
    nemo_status = get_cached_nemo_status()

    # Determine backend mode from environment variable
    backend_mode = os.environ.get("AITRANSCRIBE_BACKEND_MODE", "development")

    return NemoStatusResponse(
        nemo_available=nemo_status["available"],
        nemo_version=nemo_status["version"],
        torch_version=nemo_status["torch_version"],
        device=nemo_status["device"],
        backend_mode=backend_mode
    )


@app.post("/load", response_model=MessageResponse)
async def load_model(request: LoadModelRequest = None) -> MessageResponse:
    """
    Load an ASR model into memory.

    POST is used for actions that change state (loading a model).
    GET is used for reading data without changing anything.

    Args:
        request: Optional request body with model_id. Defaults to parakeet-v2.
    """
    model_id = request.model_id if request else DEFAULT_MODEL

    # Check if requested model is already loaded
    if is_model_loaded() and current_model_id == model_id:
        return MessageResponse(
            message=f"Model {model_id} already loaded",
            success=True
        )

    # Unload current model if a different one is loaded
    if is_model_loaded():
        unload_asr_model()

    try:
        load_asr_model(model_id)
        display_name = AVAILABLE_MODELS[model_id]["display_name"]
        return MessageResponse(
            message=f"{display_name} loaded successfully",
            success=True
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload", response_model=MessageResponse)
async def unload_model() -> MessageResponse:
    """Unload the model to free memory."""
    unload_asr_model()
    return MessageResponse(
        message="Model unloaded",
        success=True
    )


# ============================================================================
# STORAGE & MODEL DOWNLOAD ENDPOINTS
# ============================================================================

class StorageInfo(BaseModel):
    """Information about model storage."""
    storage_path: str
    whisper_path: str
    huggingface_cache: str
    total_size_mb: int
    model_count: int


@app.get("/storage", response_model=StorageInfo)
async def get_storage() -> StorageInfo:
    """
    Get information about model storage locations and sizes.

    This helps users understand where models are stored
    and how much disk space is being used.
    """
    info = get_storage_info()
    return StorageInfo(**info)


@app.get("/models/{model_id}")
async def get_model(model_id: str) -> ModelInfo:
    """
    Get detailed information about a specific model.

    Includes download status and file path if downloaded.
    """
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    info = get_model_info(model_id)
    return ModelInfo(
        id=info["id"],
        name=info["name"],
        display_name=info["display_name"],
        author=info["author"],
        type=info["type"],
        languages=info["languages"],
        language_names=info["language_names"],
        description=info["description"],
        multilingual=info["multilingual"],
        size_mb=info["size_mb"],
        ram_mb=info["ram_mb"],
        streaming_native=info.get("streaming_native", False),
        downloaded=info["downloaded"],
        download_url=info.get("download_url"),
        path=info.get("path"),
        nemo_required=is_nemo_model(model_id),
    )


@app.post("/models/{model_id}/download")
async def download_model_endpoint(model_id: str):
    """
    Download a model with progress updates via Server-Sent Events.

    Returns a stream of events:
    - {"status": "downloading", "progress": 0.5, "downloaded_mb": 256, "total_mb": 512}
    - {"status": "verifying", "progress": 1.0}
    - {"status": "complete", "path": "/path/to/model"}
    - {"status": "error", "message": "..."}
    """
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    async def progress_generator():
        for update in download_model(model_id):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/models/{model_id}", response_model=MessageResponse)
async def delete_model_endpoint(model_id: str) -> MessageResponse:
    """
    Delete a downloaded model from HuggingFace cache to free disk space.

    Works for both NeMo and Whisper models as they all use HuggingFace cache.
    Note: Cannot delete the currently loaded model - unload it first.
    """
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Don't allow deleting the currently loaded model
    if model_id == current_model_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete currently loaded model. Unload it first via POST /unload"
        )

    try:
        deleted = delete_model(model_id)
        if deleted:
            return MessageResponse(
                message=f"Model {model_id} deleted successfully",
                success=True
            )
        else:
            return MessageResponse(
                message=f"Model {model_id} was not downloaded",
                success=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    """
    Transcribe an uploaded audio file.

    UploadFile = File(...) means:
    - Expect a file upload in the request
    - File(...) means it's required (not optional)

    The client sends audio as a file, we save it temporarily,
    transcribe it, then delete the temp file.
    """
    # Check model is loaded
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Model not loaded. Call POST /load first."
        )

    # Save uploaded file to temp location
    # We need a file on disk because NeMo expects a file path
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()  # 'await' because it's async
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save audio: {e}")

    try:
        # Get audio duration
        from scipy.io.wavfile import read as wav_read
        sample_rate, audio_data = wav_read(tmp_path)
        duration = len(audio_data) / sample_rate

        # Transcribe
        text, processing_time = transcribe_audio(tmp_path)

        # Calculate realtime factor (how many times faster than realtime)
        realtime_factor = duration / processing_time if processing_time > 0 else 0

        return TranscriptionResponse(
            text=text.strip() if text else "",
            duration_seconds=duration,
            processing_time=processing_time,
            realtime_factor=realtime_factor
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/transcribe-bytes")
async def transcribe_bytes(
    audio_data: bytes = File(...),
    sample_rate: int = SAMPLE_RATE
) -> TranscriptionResponse:
    """
    Transcribe raw audio bytes (alternative to file upload).

    This is useful when the Swift app records audio directly
    and wants to send raw bytes instead of a file.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    duration = len(audio_array) / sample_rate

    # Save to temp file (NeMo needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav_write(tmp_path, sample_rate, audio_array)

    try:
        text, processing_time = transcribe_audio(tmp_path)
        realtime_factor = duration / processing_time if processing_time > 0 else 0

        return TranscriptionResponse(
            text=text.strip() if text else "",
            duration_seconds=duration,
            processing_time=processing_time,
            realtime_factor=realtime_factor
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================================
# AUDIO DEVICE ENDPOINTS
# ============================================================================

@app.get("/devices", response_model=list[AudioDeviceInfo])
async def get_audio_devices() -> list[AudioDeviceInfo]:
    """
    List all available audio input devices.

    The Swift app can use this to show a microphone picker.
    """
    devices = list_audio_devices()
    return [AudioDeviceInfo(**d) for d in devices]


@app.get("/devices/default")
async def get_default_device() -> dict:
    """Get the default audio input device."""
    device = get_default_input_device()
    if device is None:
        raise HTTPException(status_code=404, detail="No audio input device found")
    return device


# ============================================================================
# RECORDING ENDPOINTS
# ============================================================================

@app.get("/recording/status", response_model=RecordingStatus)
async def get_recording_status() -> RecordingStatus:
    """
    Get current recording status.

    The Swift app can poll this to update the UI
    (show recording indicator, volume meter, etc.)
    """
    return RecordingStatus(
        is_recording=audio_recorder.is_recording,
        duration_seconds=audio_recorder.get_duration(),
        volume=audio_recorder.get_volume()
    )


@app.post("/recording/start", response_model=MessageResponse)
async def start_recording() -> MessageResponse:
    """
    Start recording audio from the microphone.

    The recording will continue until /recording/stop is called.
    """
    if audio_recorder.is_recording:
        return MessageResponse(
            message="Already recording",
            success=False
        )

    success = audio_recorder.start()

    if success:
        return MessageResponse(
            message="Recording started",
            success=True
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to start recording")


@app.post("/recording/stop", response_model=TranscriptionResponse)
async def stop_recording_and_transcribe() -> TranscriptionResponse:
    """
    Stop recording and transcribe the audio.

    This is the main workflow:
    1. User presses hotkey -> Swift calls /recording/start
    2. User releases hotkey -> Swift calls /recording/stop
    3. Server transcribes -> Returns text
    4. Swift pastes text or shows in UI
    """
    if not audio_recorder.is_recording:
        raise HTTPException(status_code=400, detail="Not currently recording")

    if not is_model_loaded():
        # Stop recording anyway to not lose audio
        audio_recorder.stop()
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Stop and get audio data
    audio_data = audio_recorder.stop()

    if audio_data is None or len(audio_data) < SAMPLE_RATE * 0.2:
        raise HTTPException(status_code=400, detail="No audio recorded or too short")

    # Calculate duration
    duration = len(audio_data) / SAMPLE_RATE

    # Save to temp file for transcription
    tmp_path = audio_recorder.save_to_wav(audio_data)

    try:
        text, processing_time = transcribe_audio(tmp_path)
        realtime_factor = duration / processing_time if processing_time > 0 else 0

        return TranscriptionResponse(
            text=text.strip() if text else "",
            duration_seconds=duration,
            processing_time=processing_time,
            realtime_factor=realtime_factor
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/recording/cancel", response_model=MessageResponse)
async def cancel_recording() -> MessageResponse:
    """
    Cancel recording without transcribing.

    Useful when user wants to abort (e.g., pressed wrong key).
    """
    if not audio_recorder.is_recording:
        return MessageResponse(
            message="Not recording",
            success=False
        )

    audio_recorder.stop()  # Discard the audio
    return MessageResponse(
        message="Recording cancelled",
        success=True
    )


# ============================================================================
# STREAMING TRANSCRIPTION ENDPOINTS
# ============================================================================

def transcribe_audio_buffer(audio_data: np.ndarray) -> str:
    """
    Transcribe audio data from a numpy array.

    This is used for streaming - transcribes current buffer without
    waiting for recording to stop.
    """
    if not is_model_loaded() or len(audio_data) < SAMPLE_RATE * 0.3:
        return ""

    # Save to temp file
    tmp_path = audio_recorder.save_to_wav(audio_data)

    try:
        text, _ = transcribe_audio(tmp_path)
        return text.strip() if text else ""
    except Exception as e:
        print(f"Streaming transcription error: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def stream_transcription_generator():
    """
    Generator that yields transcription updates as Server-Sent Events.

    This runs while recording is active, sending partial transcriptions
    every ~1 second so the Swift app can display them in real-time.
    """
    global streaming_active, streaming_stop_requested

    last_text = ""

    while streaming_active and not streaming_stop_requested:
        # Wait a bit before next transcription
        await asyncio.sleep(1.0)

        if not audio_recorder.is_recording:
            break

        # Get current audio buffer
        if len(audio_recorder.recording_buffer) > 0:
            current_audio = np.concatenate(audio_recorder.recording_buffer).flatten()

            # Transcribe current buffer
            text = transcribe_audio_buffer(current_audio)

            # Only send if text changed
            if text and text != last_text:
                last_text = text
                event_data = {
                    "type": "partial",
                    "text": text,
                    "duration": audio_recorder.get_duration()
                }
                yield f"data: {json.dumps(event_data)}\n\n"

    # Send final result
    if audio_recorder.is_recording:
        audio_data = audio_recorder.stop()
        if audio_data is not None and len(audio_data) >= SAMPLE_RATE * 0.2:
            final_text = transcribe_audio_buffer(audio_data)
            event_data = {
                "type": "final",
                "text": final_text,
                "duration": len(audio_data) / SAMPLE_RATE
            }
            yield f"data: {json.dumps(event_data)}\n\n"

    streaming_active = False
    streaming_stop_requested = False


@app.get("/recording/stream")
async def stream_recording():
    """
    Start recording with streaming transcription.

    Returns a Server-Sent Events stream that sends partial transcriptions
    in real-time as the user speaks.

    Events:
    - {"type": "partial", "text": "...", "duration": 1.5}  - Partial result
    - {"type": "final", "text": "...", "duration": 3.2}    - Final result

    The Swift app should:
    1. Open this endpoint as an SSE stream
    2. Listen for events and type text as it arrives
    3. Call /recording/stream/stop when done
    """
    global streaming_active, streaming_stop_requested

    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if streaming_active or audio_recorder.is_recording:
        raise HTTPException(status_code=400, detail="Already recording")

    # Start recording
    success = audio_recorder.start()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start recording")

    streaming_active = True
    streaming_stop_requested = False

    return StreamingResponse(
        stream_transcription_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering if behind proxy
        }
    )


@app.post("/recording/stream/stop", response_model=MessageResponse)
async def stop_streaming() -> MessageResponse:
    """
    Stop the streaming transcription.

    This signals the stream generator to stop and send the final result.
    """
    global streaming_stop_requested, streaming_active

    if not streaming_active:
        return MessageResponse(
            message="No streaming session active",
            success=False
        )

    streaming_stop_requested = True

    return MessageResponse(
        message="Streaming stop requested",
        success=True
    )


# ============================================================================
# MAIN - Run the server
# ============================================================================

if __name__ == "__main__":
    """
    This block runs when you execute: python server.py

    uvicorn is the ASGI server that runs FastAPI.
    ASGI = Asynchronous Server Gateway Interface
    (It's the modern, async version of WSGI)
    """
    import uvicorn

    print("Starting AiTranscribe server...")
    print(f"API docs will be at: http://{HOST}:{PORT}/docs")

    uvicorn.run(
        app,               # Pass the app object directly (required for PyInstaller)
        host=HOST,
        port=PORT,
        reload=False,      # Set True during development to auto-reload on changes
        log_level="info"
    )
