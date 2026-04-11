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
import threading

# Force unbuffered stdout so logs appear immediately in Swift's debug console
# This is important because Python buffers stdout by default when not connected to a TTY
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import queue
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from collections import deque
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
from summary_manager import (
    SUMMARY_MODELS,
    get_all_models_info as get_all_summary_models_info,
    get_model_info as get_summary_model_info,
    get_runtime_status as get_summary_runtime_status,
    download_summary_model,
    delete_summary_model,
    get_summary_setup_resource_path,
    get_summary_python_path,
    get_summary_worker_resource_path,
    get_summary_model_dir,
    get_summary_file_name,
    should_quantize_kv_cache,
    estimate_summary_memory,
    normalize_summary_request,
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

# Summary worker state
active_summary_process = None
active_summary_cancel_requested = False

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
    model_url: Optional[str] = None  # URL to model page (HuggingFace/GitHub)
    path: Optional[str]            # Path to model file (if downloaded)
    nemo_required: bool            # Does this model require NeMo to be installed?
    session_compatible: bool       # Can this model do batch/session transcription?
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


class SummaryRuntimeResponse(BaseModel):
    installed: bool
    ready: bool
    venv_path: str
    python_path: Optional[str]
    worker_path: Optional[str]
    requirements_path: Optional[str]
    mlx_vlm_available: bool


class SummaryModelInfo(BaseModel):
    id: str
    display_name: str
    provider: str
    engine: str
    quantization: str
    context_tokens: int
    download_size_mb: int
    resident_model_ram_mb: int
    downloaded: bool
    recommended: bool
    path: Optional[str]
    model_url: Optional[str] = None
    kv_bytes_per_token_default: int
    kv_bytes_per_token_quantized: int


class SummaryPresetRequest(BaseModel):
    preset_id: str
    display_name: str
    file_name: str
    system_prompt: str
    target_words: int
    max_output_tokens: int


class SessionSummarizeRequest(BaseModel):
    session_dir: str
    model_id: str
    presets: list[SummaryPresetRequest]


class SummaryRuntimeInstallRequest(BaseModel):
    python_path: Optional[str] = None


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

# Whisper model path (whisper.cpp GGML file — no in-memory model, loaded per-invocation by whisper-cli)
whisper_model_path = None
current_model_type = None  # "nemo" or "whisper"

# Path to whisper-cli binary
import subprocess as _subprocess
def _find_whisper_cli() -> Optional[str]:
    """Find the whisper-cli binary path. Returns None if not found."""
    candidates = [
        Path(__file__).parent / "bin" / "whisper-cli",          # backend/bin/ (development)
        Path(__file__).parent / "whisper-cli",                  # Same dir as server.py (Xcode bundle)
        Path(__file__).parent.parent / "bin" / "whisper-cli",   # One level up
        Path(sys.executable).parent / "bin" / "whisper-cli",    # PyInstaller bundle
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    # Check system PATH as fallback
    import shutil
    system_path = shutil.which("whisper-cli")
    if system_path:
        return system_path
    return None

whisper_cli_path = _find_whisper_cli()
if whisper_cli_path:
    print(f"whisper-cli found at: {whisper_cli_path}")
else:
    print("WARNING: whisper-cli not found. Whisper models will not work.")

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
    global asr_model, whisper_model_path, model_device, current_model_id, current_model_type

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
    """Validate whisper.cpp GGML model exists and store its path.

    whisper.cpp loads the model per-invocation via whisper-cli subprocess,
    so "loading" just validates the file and stores the path. Instant.
    """
    global whisper_model_path, model_device

    filename = model_config["filename"]
    path = get_whisper_dir() / filename

    if not path.exists():
        raise FileNotFoundError(f"Whisper GGML model not found: {path}")
    if whisper_cli_path is None:
        raise FileNotFoundError("whisper-cli binary not found. Whisper models require it.")

    whisper_model_path = str(path)
    model_device = "metal"  # whisper.cpp uses Metal on Apple Silicon
    file_size_mb = int(path.stat().st_size / (1024 * 1024))
    print(f"")
    print(f"========================================")
    print(f"  WHISPER ENGINE: whisper.cpp (Metal GPU)")
    print(f"  Model: {filename} ({file_size_mb} MB)")
    print(f"  Path: {whisper_model_path}")
    print(f"  Binary: {whisper_cli_path}")
    print(f"========================================")


def unload_asr_model():
    """
    Unload the model to free memory.

    Python has garbage collection, but explicitly deleting
    and clearing CUDA/MPS cache ensures memory is freed.
    """
    global asr_model, whisper_model_path, model_device, current_model_id, current_model_type

    # Import gc early for multiple collection passes
    import gc

    # Store device type before clearing
    was_using_mps = model_device == "mps"
    was_using_cuda = model_device == "cuda"

    if asr_model is not None:
        # Move model to CPU first to release GPU/MPS memory
        try:
            asr_model.cpu()
        except Exception:
            pass
        del asr_model
        asr_model = None

    # whisper.cpp: just clear the path (no in-memory model to free)
    whisper_model_path = None

    model_device = None
    current_model_id = None
    current_model_type = None

    # Force garbage collection multiple times
    # (sometimes needed for circular references)
    for _ in range(3):
        gc.collect()

    # Clear PyTorch cached memory
    try:
        import torch

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear MPS cache (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have empty_cache in older PyTorch versions
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            # Synchronize to ensure all operations complete
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not clear GPU cache: {e}")

    # Final garbage collection pass
    gc.collect()

    print("Model unloaded and memory freed")


def transcribe_audio(audio_path: str) -> tuple[str, float]:
    """
    Transcribe an audio file.

    Supports both NeMo and Whisper models.

    Args:
        audio_path: Path to a WAV file

    Returns:
        Tuple of (transcribed_text, processing_time)
    """
    global asr_model, whisper_model_path, current_model_type

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
    import numpy as np
    from scipy.io.wavfile import read as wav_read

    start_time = time.time()

    try:
        print(f"DEBUG: Transcribing with NeMo: {audio_path}")

        # Load audio directly to bypass Lhotse dataloader compatibility issues
        sample_rate, audio_data = wav_read(audio_path)

        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed (NeMo expects 16kHz)
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            sample_rate = 16000

        # Convert to tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)

        # Move to same device as model
        device = next(asr_model.parameters()).device
        audio_tensor = audio_tensor.to(device)
        audio_length = audio_length.to(device)

        print(f"DEBUG: Audio tensor shape: {audio_tensor.shape}, length: {audio_length}")

        # Transcribe using direct forward pass to avoid Lhotse compatibility issues
        with torch.no_grad():
            text = None

            # Method 1: Direct forward pass with preprocessor
            # This bypasses the lhotse dataloader entirely
            try:
                print("DEBUG: Trying direct forward pass with preprocessor...")

                # Step 1: Use model's preprocessor to get mel spectrogram features
                if hasattr(asr_model, 'preprocessor') and asr_model.preprocessor is not None:
                    processed_signal, processed_signal_length = asr_model.preprocessor(
                        input_signal=audio_tensor,
                        length=audio_length
                    )
                    print(f"DEBUG: Preprocessed signal shape: {processed_signal.shape}")
                else:
                    raise RuntimeError("Model has no preprocessor")

                # Step 2: Get encoder output
                encoded, encoded_len = asr_model.encoder(
                    audio_signal=processed_signal,
                    length=processed_signal_length
                )
                print(f"DEBUG: Encoded shape: {encoded.shape}")

                # Step 3: Decode based on model type
                if hasattr(asr_model, 'joint') and hasattr(asr_model, 'decoding'):
                    # RNNT/Transducer model (Parakeet uses this)
                    print("DEBUG: Using RNNT decoding...")

                    # Use greedy decoding - handle different return value formats
                    decode_result = asr_model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=encoded,
                        encoded_lengths=encoded_len,
                        return_hypotheses=False
                    )

                    # Handle different return formats from NeMo versions
                    if isinstance(decode_result, tuple):
                        best_hyp = decode_result[0]
                        print(f"DEBUG: Decode returned tuple with {len(decode_result)} elements")
                    else:
                        best_hyp = decode_result
                        print(f"DEBUG: Decode returned single value of type {type(best_hyp)}")

                    print(f"DEBUG: best_hyp type: {type(best_hyp)}, value: {best_hyp}")

                    if best_hyp is not None and len(best_hyp) > 0:
                        # Decode token IDs to text
                        if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
                            # Get the first (and only) hypothesis
                            hyp = best_hyp[0]
                            print(f"DEBUG: hyp type: {type(hyp)}, value: {hyp}")

                            if isinstance(hyp, torch.Tensor):
                                token_ids = hyp.cpu().tolist()
                            elif hasattr(hyp, 'y_sequence'):
                                # NeMo Hypothesis object
                                token_ids = hyp.y_sequence if isinstance(hyp.y_sequence, list) else hyp.y_sequence.cpu().tolist()
                            elif isinstance(hyp, list):
                                token_ids = hyp
                            else:
                                token_ids = list(hyp) if hasattr(hyp, '__iter__') else [hyp]

                            print(f"DEBUG: token_ids: {token_ids}")
                            text = asr_model.tokenizer.ids_to_text(token_ids)
                        else:
                            text = str(best_hyp[0])

                elif hasattr(asr_model, 'decoder'):
                    # CTC-based model
                    print("DEBUG: Using CTC decoding...")
                    log_probs = asr_model.decoder(encoder_output=encoded)
                    greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

                    if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
                        # Remove CTC blanks and duplicates
                        pred_ids = greedy_predictions[0].cpu().tolist()
                        # Simple CTC decode (remove blanks and consecutive duplicates)
                        decoded_ids = []
                        prev_id = -1
                        blank_id = asr_model.tokenizer.blank_id if hasattr(asr_model.tokenizer, 'blank_id') else 0
                        for idx in pred_ids:
                            if idx != blank_id and idx != prev_id:
                                decoded_ids.append(idx)
                            prev_id = idx
                        text = asr_model.tokenizer.ids_to_text(decoded_ids)
                    else:
                        text = str(greedy_predictions[0].tolist())

                if text:
                    print(f"DEBUG: Method 1 (direct) succeeded: '{text}'")

            except Exception as e1:
                print(f"DEBUG: Method 1 (direct) failed: {e1}")
                import traceback
                traceback.print_exc()

                # Method 2: Try standard transcribe (will fail with lhotse bug, but try anyway)
                try:
                    print("DEBUG: Trying standard transcribe method...")
                    result = asr_model.transcribe(
                        audio=[audio_path],
                        batch_size=1,
                        verbose=False
                    )
                    if result and len(result) > 0:
                        if hasattr(result[0], 'text'):
                            text = result[0].text
                        elif isinstance(result[0], str):
                            text = result[0]
                        else:
                            text = str(result[0])
                    print(f"DEBUG: Method 2 succeeded: '{text}'")
                except Exception as e2:
                    print(f"DEBUG: Method 2 failed: {e2}")
                    # Re-raise the original error if all methods fail
                    raise e1

        processing_time = time.time() - start_time

        if text is None:
            text = ""

        print(f"DEBUG: Final transcription: '{text}' in {processing_time:.2f}s")

        return text, processing_time

    except Exception as e:
        print(f"ERROR: NeMo transcription failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def _transcribe_whisper(audio_path: str) -> tuple[str, float]:
    """Transcribe using whisper.cpp via subprocess (Metal GPU accelerated)."""
    global whisper_model_path

    if whisper_model_path is None:
        raise RuntimeError("Whisper model not loaded")
    if whisper_cli_path is None:
        raise RuntimeError("whisper-cli binary not found. Place it at backend/bin/whisper-cli")

    print(f"[whisper.cpp] Transcribing: {audio_path}")
    print(f"[whisper.cpp] Model: {whisper_model_path}")
    start_time = time.time()

    result = _subprocess.run(
        [
            whisper_cli_path,
            "-m", whisper_model_path,
            "-f", audio_path,
            "-np",          # no prints (suppress log output)
            "-nt",          # no timestamps in text output
            "-t", "4",      # threads
            "-l", "auto",   # auto-detect language
        ],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min safety net
        cwd="/tmp",    # explicit cwd — prevents getcwd crash if parent dir was deleted
    )

    processing_time = time.time() - start_time

    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"ERROR: whisper-cli failed (code {result.returncode}): {stderr}")
        raise RuntimeError(f"whisper-cli failed: {stderr}")

    text = result.stdout.strip()
    word_count = len(text.split()) if text else 0
    print(f"[whisper.cpp] Done in {processing_time:.1f}s — {word_count} words")
    return text, processing_time


# ============================================================================
# TRANSCRIPTION PROGRESS TRACKING (for SSE streaming endpoint)
# ============================================================================

_transcription_progress = {"progress": 0.0, "estimated": True}
_transcription_progress_lock = threading.Lock()


def _set_progress(progress: float, estimated: bool = True):
    """Thread-safe progress update for streaming transcription."""
    with _transcription_progress_lock:
        _transcription_progress["progress"] = progress
        _transcription_progress["estimated"] = estimated


def _get_progress() -> dict:
    """Thread-safe progress read for streaming transcription."""
    with _transcription_progress_lock:
        return dict(_transcription_progress)


def _transcribe_whisper_with_progress(audio_path: str, audio_duration: float) -> tuple[str, float]:
    """Transcribe using whisper.cpp via subprocess.

    whisper-cli doesn't support per-segment progress callbacks, so batch
    transcription handles progress at the chunk level instead.
    """
    return _transcribe_whisper(audio_path)


def is_model_loaded() -> bool:
    """Check if any model is currently loaded."""
    return asr_model is not None or whisper_model_path is not None


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
    # Set process name so it shows as "AiTranscribe Server" in Activity Monitor
    try:
        import setproctitle
        setproctitle.setproctitle("AiTranscribeServer")
    except ImportError:
        pass  # setproctitle not installed, process will show as "python"

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
    version="0.2.0",
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
            model_url=info.get("model_url"),
            path=info.get("path"),
            nemo_required=requires_nemo,
            session_compatible=info["type"] != "nemo" or info["id"] in ("parakeet-v2", "parakeet-v3"),
            can_use=can_use,
        ))

    return result


@app.get("/summary/runtime/status", response_model=SummaryRuntimeResponse)
async def get_summary_runtime() -> SummaryRuntimeResponse:
    status = get_summary_runtime_status()
    return SummaryRuntimeResponse(**status.__dict__)


@app.post("/summary/runtime/install", response_model=MessageResponse)
async def install_summary_runtime(request: SummaryRuntimeInstallRequest = None) -> MessageResponse:
    setup_script = get_summary_setup_resource_path()
    if setup_script is None or not setup_script.exists():
        raise HTTPException(status_code=404, detail="Summary setup script not found")

    python_path = request.python_path if request and request.python_path else sys.executable
    venv_path = get_summary_python_path().parent.parent

    try:
        result = _subprocess.run(
            [python_path, str(setup_script), str(venv_path)],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr or result.stdout or "Install failed")
        return MessageResponse(message="Summary runtime installed", success=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/summary/models", response_model=list[SummaryModelInfo])
async def list_summary_models() -> list[SummaryModelInfo]:
    result = []
    for info in get_all_summary_models_info():
        result.append(SummaryModelInfo(**info))
    return result


@app.get("/summary/models/{model_id}", response_model=SummaryModelInfo)
async def get_summary_model(model_id: str) -> SummaryModelInfo:
    if model_id not in SUMMARY_MODELS:
        raise HTTPException(status_code=404, detail=f"Summary model not found: {model_id}")
    return SummaryModelInfo(**get_summary_model_info(model_id))


@app.post("/summary/models/{model_id}/download")
async def download_summary_model_endpoint(model_id: str):
    if model_id not in SUMMARY_MODELS:
        raise HTTPException(status_code=404, detail=f"Summary model not found: {model_id}")

    async def progress_generator():
        for update in download_summary_model(model_id):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/summary/models/{model_id}", response_model=MessageResponse)
async def delete_summary_model_endpoint(model_id: str) -> MessageResponse:
    if model_id not in SUMMARY_MODELS:
        raise HTTPException(status_code=404, detail=f"Summary model not found: {model_id}")

    if active_summary_process is not None:
        raise HTTPException(status_code=400, detail="Cannot delete summary model while a summary is running")

    deleted = delete_summary_model(model_id)
    return MessageResponse(
        message=f"Summary model {model_id} deleted" if deleted else f"Summary model {model_id} was not downloaded",
        success=deleted,
    )


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
        model_url=info.get("model_url"),
        path=info.get("path"),
        nemo_required=is_nemo_model(model_id),
        session_compatible=info["type"] != "nemo" or model_id in ("parakeet-v2", "parakeet-v3"),
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


@app.post("/transcribe-stream")
async def transcribe_stream(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file with SSE progress updates.

    Runs transcription in a background thread and sends heartbeat events
    every 2 seconds to keep the connection alive and report progress.

    Events:
    - {"type": "heartbeat", "elapsed": 12.5, "progress": 0.45, "estimated": true}
    - {"type": "complete", "text": "...", "duration_seconds": ..., "processing_time": ..., "realtime_factor": ...}
    - {"type": "error", "message": "..."}
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Call POST /load first.")

    # Save uploaded file to temp location (same as /transcribe)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save audio: {e}")

    # Read audio duration
    from scipy.io.wavfile import read as wav_read
    sample_rate_val, audio_data_arr = wav_read(tmp_path)
    audio_duration = len(audio_data_arr) / sample_rate_val

    async def event_generator():
        result_holder = {"text": None, "processing_time": None, "error": None}
        _set_progress(0.0, estimated=True)

        def run_transcription():
            try:
                if current_model_type == "whisper":
                    text, proc_time = _transcribe_whisper_with_progress(tmp_path, audio_duration)
                else:
                    text, proc_time = transcribe_audio(tmp_path)
                result_holder["text"] = text
                result_holder["processing_time"] = proc_time
            except Exception as e:
                result_holder["error"] = str(e)

        # Start transcription in background thread
        thread = threading.Thread(target=run_transcription)
        thread.start()

        start_time = time.time()

        try:
            # Send heartbeats while transcription runs
            while thread.is_alive():
                await asyncio.sleep(2.0)
                elapsed = time.time() - start_time
                prog = _get_progress()

                # For NeMo, estimate progress based on typical realtime factor (~3.5x)
                if current_model_type == "nemo":
                    estimated_total = audio_duration / 3.5
                    progress = min(elapsed / estimated_total, 0.95) if estimated_total > 0 else 0.0
                    prog = {"progress": progress, "estimated": True}

                event = {
                    "type": "heartbeat",
                    "elapsed": round(elapsed, 1),
                    "progress": round(prog["progress"], 2),
                    "estimated": prog["estimated"]
                }
                yield f"data: {json.dumps(event)}\n\n"

            thread.join()

            # Send final result or error
            if result_holder["error"]:
                event = {"type": "error", "message": result_holder["error"]}
                yield f"data: {json.dumps(event)}\n\n"
            else:
                processing_time = result_holder["processing_time"]
                realtime_factor = audio_duration / processing_time if processing_time > 0 else 0
                event = {
                    "type": "complete",
                    "text": result_holder["text"].strip() if result_holder["text"] else "",
                    "duration_seconds": audio_duration,
                    "processing_time": processing_time,
                    "realtime_factor": realtime_factor
                }
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # Always clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
# SESSION ENDPOINTS - Batch transcription of long recordings
# ============================================================================

from batch_transcriber import (
    BatchTranscriber,
    get_sessions_dir,
    estimate_batches,
)

# Active batch transcriber (for cancellation support)
active_batch_transcriber: Optional[BatchTranscriber] = None


class BatchTranscribeRequest(BaseModel):
    """Request to batch-transcribe a session recording."""
    session_dir: str       # Directory name within Sessions/ folder
    model_id: str          # e.g., "parakeet-v2"
    ram_budget_mb: int     # Total RAM budget in MB (e.g., 4096)


class EstimateBatchesRequest(BaseModel):
    """Request to estimate batch count for a recording."""
    audio_duration_seconds: float
    model_id: str
    ram_budget_mb: int


class SessionDeleteRequest(BaseModel):
    """Request to delete parts of a session."""
    delete_audio: bool = True
    delete_transcription: bool = True


class BulkDeleteAudioRequest(BaseModel):
    """Request to bulk-delete audio files."""
    transcribed_only: bool = False


@app.post("/session/transcribe")
async def session_transcribe(request: BatchTranscribeRequest):
    """
    Batch-transcribe a session recording with SSE progress updates.

    The audio file is split into chunks based on RAM budget, transcribed
    sequentially, and the results are concatenated with overlap deduplication.

    Events:
    - {"event": "started", "total_batches": N, ...}
    - {"event": "batch_progress", "batch": N, "total": M, "percent": P}
    - {"event": "batch_complete", "batch": N, "total": M, "batch_text": "..."}
    - {"event": "stats", "cpu_percent": X, "memory_mb": Y, "eta_seconds": Z}
    - {"event": "done", "full_text": "...", "total_batches": N, "total_time": T}
    - {"event": "error", "message": "..."}
    """
    global active_batch_transcriber

    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Call POST /load first.")

    # Resolve session directory
    sessions_base = get_sessions_dir()
    session_path = sessions_base / request.session_dir

    if not session_path.exists():
        raise HTTPException(status_code=404, detail=f"Session directory not found: {request.session_dir}")

    # Find audio file (M4A or WAV)
    audio_path = None
    for ext in [".m4a", ".wav", ".mp4"]:
        candidate = session_path / f"audio{ext}"
        if candidate.exists():
            audio_path = str(candidate)
            break

    if audio_path is None:
        raise HTTPException(status_code=404, detail="No audio file found in session directory")

    print(f"Session transcribe: dir={request.session_dir}, model={request.model_id}, "
          f"budget={request.ram_budget_mb}MB, audio={audio_path}")

    # Create the batch transcriber
    transcriber = BatchTranscriber(
        audio_path=audio_path,
        model_id=request.model_id,
        ram_budget_mb=request.ram_budget_mb,
        transcribe_fn=transcribe_audio,
    )
    active_batch_transcriber = transcriber

    async def event_generator():
        """
        Stream transcription events without blocking the event loop.

        The synchronous transcriber.transcribe() generator is polled from a
        background thread so other endpoints (/models, /status, /cancel) remain
        responsive during long whisper-large-v3 transcriptions.
        """
        global active_batch_transcriber
        event_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()

        def _run_transcription():
            try:
                for event in transcriber.transcribe():
                    event_queue.put(event)
            except Exception as e:
                event_queue.put({"event": "error", "message": str(e)})
            finally:
                event_queue.put(_SENTINEL)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_transcription)

        try:
            while True:
                # Poll the queue without blocking the event loop
                while True:
                    try:
                        event = event_queue.get_nowait()
                        break
                    except queue.Empty:
                        await asyncio.sleep(0.5)
                        continue

                if event is _SENTINEL:
                    break

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("event") == "done":
                    full_text = event.get("full_text", "")
                    _save_session_transcription(session_path, full_text, event)
        finally:
            active_batch_transcriber = None

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/session/cancel-transcription", response_model=MessageResponse)
async def session_cancel_transcription():
    """Cancel an active batch transcription."""
    global active_batch_transcriber

    if active_batch_transcriber is None:
        return MessageResponse(message="No active transcription to cancel", success=False)

    active_batch_transcriber.cancel()
    return MessageResponse(message="Transcription cancellation requested", success=True)


@app.post("/session/summarize")
async def session_summarize(request: SessionSummarizeRequest):
    global active_summary_process, active_summary_cancel_requested

    if request.model_id not in SUMMARY_MODELS:
        raise HTTPException(status_code=404, detail=f"Summary model not found: {request.model_id}")

    if not request.presets:
        raise HTTPException(status_code=400, detail="At least one summary preset is required")

    try:
        normalized_presets = [normalize_summary_request(preset.model_dump()) for preset in request.presets]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    runtime_status = get_summary_runtime_status()
    if not runtime_status.installed:
        raise HTTPException(status_code=503, detail="Summary runtime not installed")
    if runtime_status.worker_path is None or runtime_status.python_path is None:
        raise HTTPException(status_code=503, detail="Summary runtime is incomplete")

    sessions_base = get_sessions_dir()
    session_path = sessions_base / request.session_dir
    if not session_path.exists():
        raise HTTPException(status_code=404, detail=f"Session directory not found: {request.session_dir}")

    transcription_path = session_path / "transcription.txt"
    if not transcription_path.exists():
        raise HTTPException(status_code=404, detail="Session transcription not found")

    model_path = get_summary_model_dir(request.model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Summary model is not downloaded")

    if active_summary_process is not None:
        raise HTTPException(status_code=400, detail="Another summary is already running")

    transcript_text = transcription_path.read_text(encoding="utf-8")
    word_count = len(transcript_text.split())
    peak_output_tokens = max(preset["max_output_tokens"] for preset in normalized_presets)
    quantize_kv = should_quantize_kv_cache(request.model_id, word_count, peak_output_tokens)
    memory_estimate = estimate_summary_memory(
        request.model_id,
        word_count,
        reserved_output_tokens=peak_output_tokens,
        quantized=quantize_kv,
    )
    context_limit = SUMMARY_MODELS[request.model_id]["context_tokens"]
    if memory_estimate["required_context_tokens"] > context_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Transcript is too large for {request.model_id}: "
                f"{memory_estimate['required_context_tokens']} > {context_limit} context tokens"
            ),
        )

    async def event_generator():
        global active_summary_process, active_summary_cancel_requested
        event_queue: queue.Queue = queue.Queue()
        _SENTINEL = object()
        active_summary_cancel_requested = False
        stderr_lines: deque[str] = deque(maxlen=20)
        terminal_event_seen = False
        requests_file_path = None

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as request_file:
            json.dump({"presets": normalized_presets}, request_file)
            requests_file_path = request_file.name

        command = [
            runtime_status.python_path,
            runtime_status.worker_path,
            "--model-path",
            str(model_path),
            "--transcription-file",
            str(transcription_path),
            "--requests-file",
            requests_file_path,
            "--word-count",
            str(word_count),
        ]
        if quantize_kv:
            command.extend(
                [
                    "--kv-bits",
                    "8",
                    "--kv-quant-scheme",
                    "turboquant",
                    "--kv-group-size",
                    "64",
                    "--quantized-kv-start",
                    "0",
                ]
            )

        process = None
        process = _subprocess.Popen(
            command,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        active_summary_process = process

        def _pump_stdout():
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event_queue.put(json.loads(line))
                    except json.JSONDecodeError:
                        event_queue.put({"event": "error", "message": f"Invalid worker output: {line}"})
                        break
            finally:
                event_queue.put(_SENTINEL)

        def _pump_stderr():
            try:
                assert process.stderr is not None
                for line in process.stderr:
                    line = line.strip()
                    if line:
                        stderr_lines.append(line)
                        print(f"[summary-worker] {line}")
            except Exception:
                pass

        threading.Thread(target=_pump_stdout, daemon=True).start()
        threading.Thread(target=_pump_stderr, daemon=True).start()

        try:
            initial = {
                "event": "preparing_runtime",
                "model_id": request.model_id,
                "preset_ids": [preset["preset_id"] for preset in normalized_presets],
                "total_presets": len(normalized_presets),
                "memory_estimate": memory_estimate,
            }
            yield f"data: {json.dumps(initial)}\n\n"

            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    if process.poll() is not None and event_queue.empty():
                        if not terminal_event_seen:
                            if active_summary_cancel_requested:
                                terminal_event_seen = True
                                yield f"data: {json.dumps({'event': 'cancelled'})}\n\n"
                            else:
                                terminal_event_seen = True
                                message = f"Summary worker exited unexpectedly (code {process.returncode})"
                                if stderr_lines:
                                    message = f"{message}: {stderr_lines[-1]}"
                                yield f"data: {json.dumps({'event': 'error', 'message': message})}\n\n"
                        break
                    continue

                if event is _SENTINEL:
                    break

                yield f"data: {json.dumps(event)}\n\n"
                event_name = event.get("event")
                if event_name == "done":
                    _save_session_summary(
                        session_path=session_path,
                        preset_id=event.get("preset_id") or "",
                        model_id=request.model_id,
                        full_text=event.get("text", ""),
                        done_event=event,
                    )
                if event_name in {"batch_complete", "error", "cancelled"}:
                    terminal_event_seen = True
                if event_name in {"error", "cancelled"}:
                    break
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
            active_summary_process = None
            if requests_file_path and os.path.exists(requests_file_path):
                try:
                    os.unlink(requests_file_path)
                except OSError:
                    pass
            active_summary_cancel_requested = False

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/session/cancel-summary", response_model=MessageResponse)
async def session_cancel_summary() -> MessageResponse:
    global active_summary_process, active_summary_cancel_requested

    if active_summary_process is None:
        return MessageResponse(message="No active summary to cancel", success=False)

    active_summary_cancel_requested = True
    try:
        active_summary_process.terminate()
    except Exception:
        pass
    return MessageResponse(message="Summary cancellation requested", success=True)


@app.post("/session/estimate", response_model=None)
async def session_estimate(request: EstimateBatchesRequest):
    """
    Estimate the number of batches and processing time for a recording.
    Used by the frontend to show estimates before the user starts transcription.
    """
    result = estimate_batches(
        audio_duration_seconds=request.audio_duration_seconds,
        model_id=request.model_id,
        ram_budget_mb=request.ram_budget_mb,
    )
    return result


@app.get("/session/list")
async def session_list():
    """
    List all sessions with their metadata.
    Reads metadata.json from each session directory.
    """
    sessions_base = get_sessions_dir()
    sessions = []

    if not sessions_base.exists():
        return {"sessions": []}

    for entry in sorted(sessions_base.iterdir(), reverse=True):
        if not entry.is_dir():
            continue

        metadata_path = entry / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                metadata["directory_name"] = entry.name

                # Check actual file state
                audio_m4a = entry / "audio.m4a"
                audio_wav = entry / "audio.wav"
                metadata["has_audio"] = audio_m4a.exists() or audio_wav.exists()

                transcription_path = entry / "transcription.txt"
                metadata["has_transcription"] = transcription_path.exists()

                if metadata["has_audio"]:
                    audio_file = audio_m4a if audio_m4a.exists() else audio_wav
                    metadata["file_size_mb"] = round(audio_file.stat().st_size / 1_000_000, 2)

                sessions.append(metadata)
            except Exception as e:
                print(f"Error reading session metadata from {entry}: {e}")

    return {"sessions": sessions}


@app.delete("/session/{session_dir}")
async def session_delete(session_dir: str, request: SessionDeleteRequest = None):
    """Delete a session or parts of it."""
    import shutil

    sessions_base = get_sessions_dir()
    session_path = sessions_base / session_dir

    if not session_path.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_dir}")

    if request is None or (request.delete_audio and request.delete_transcription):
        # Delete entire session directory
        shutil.rmtree(session_path)
        return MessageResponse(message=f"Session deleted: {session_dir}", success=True)

    if request.delete_audio:
        for ext in [".m4a", ".wav", ".mp4"]:
            audio_file = session_path / f"audio{ext}"
            if audio_file.exists():
                audio_file.unlink()

        # Update metadata
        _update_session_metadata(session_path, {"has_audio": False, "file_size_mb": 0})

    if request.delete_transcription:
        transcription_file = session_path / "transcription.txt"
        if transcription_file.exists():
            transcription_file.unlink()
        _clear_session_summaries(session_path)
        _update_session_metadata(session_path, {"has_transcription": False})

    return MessageResponse(message=f"Session updated: {session_dir}", success=True)


@app.post("/session/bulk-delete-audio", response_model=MessageResponse)
async def session_bulk_delete_audio(request: BulkDeleteAudioRequest):
    """Bulk delete audio files from sessions."""
    sessions_base = get_sessions_dir()

    if not sessions_base.exists():
        return MessageResponse(message="No sessions found", success=True)

    deleted_count = 0
    for entry in sessions_base.iterdir():
        if not entry.is_dir():
            continue

        # If transcribed_only, check for transcription
        if request.transcribed_only:
            transcription = entry / "transcription.txt"
            if not transcription.exists():
                continue

        # Delete audio files
        for ext in [".m4a", ".wav", ".mp4"]:
            audio_file = entry / f"audio{ext}"
            if audio_file.exists():
                audio_file.unlink()
                deleted_count += 1

        _update_session_metadata(entry, {"has_audio": False, "file_size_mb": 0})

    return MessageResponse(
        message=f"Deleted audio from {deleted_count} sessions",
        success=True,
    )


def _save_session_transcription(session_path: Path, full_text: str, done_event: dict):
    """Save transcription results to the session directory."""
    # Save transcription text
    transcription_path = session_path / "transcription.txt"
    transcription_path.write_text(full_text, encoding="utf-8")
    _clear_session_summaries(session_path)

    # Update metadata
    updates = {
        "has_transcription": True,
        "word_count": done_event.get("word_count", 0),
        "batch_count": done_event.get("total_batches", 0),
        "transcription_time_seconds": done_event.get("total_time", 0),
        "status": "completed",
    }
    _update_session_metadata(session_path, updates)
    print(f"Session transcription saved: {transcription_path}")


def _save_session_summary(
    session_path: Path,
    preset_id: str,
    model_id: str,
    full_text: str,
    done_event: dict,
):
    summary_file_name = done_event.get("file_name") or get_summary_file_name(preset_id)
    summary_path = session_path / summary_file_name
    summary_path.write_text(full_text, encoding="utf-8")

    metadata_path = session_path / "metadata.json"
    current_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as file:
                current_metadata = json.load(file)
        except Exception:
            current_metadata = {}

    summaries = current_metadata.get("summaries", {}) or {}
    summaries[preset_id] = {
        "has_summary": True,
        "status": "completed",
        "status_message": None,
        "file_name": summary_file_name,
        "model_id": model_id,
        "model_name": SUMMARY_MODELS[model_id]["display_name"],
        "preset_display_name": done_event.get("display_name"),
        "word_count": done_event.get("output_word_count", len(full_text.split())),
        "target_word_count": done_event.get("target_words"),
        "max_output_tokens": done_event.get("max_output_tokens"),
        "processing_time_seconds": done_event.get("processing_time_seconds", 0),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    _update_session_metadata(session_path, {"summaries": summaries})
    print(f"Session summary saved: {summary_path}")


def _clear_session_summaries(session_path: Path):
    metadata_path = session_path / "metadata.json"
    current_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as file:
                current_metadata = json.load(file)
        except Exception:
            current_metadata = {}

    summaries = {}
    current_summaries = current_metadata.get("summaries", {}) or {}
    default_presets = {
        "general": ("General Summary", "summary-general.md"),
        "meeting_notes": ("Meeting Notes", "summary-meeting-notes.md"),
        "action_items": ("Action Items", "summary-action-items.md"),
        "technical": ("Technical Summary", "summary-technical.md"),
    }

    for preset_id, default_data in default_presets.items():
        current_summaries.setdefault(
            preset_id,
            {
                "file_name": default_data[1],
                "preset_display_name": default_data[0],
            },
        )

    for preset_id, metadata in current_summaries.items():
        summary_path = session_path / metadata.get("file_name", get_summary_file_name(preset_id))
        if summary_path.exists():
            summary_path.unlink()
        summaries[preset_id] = {
            "has_summary": False,
            "status": "idle",
            "status_message": None,
            "file_name": metadata.get("file_name", get_summary_file_name(preset_id)),
            "model_id": None,
            "model_name": None,
            "preset_display_name": metadata.get("preset_display_name"),
            "word_count": None,
            "target_word_count": None,
            "max_output_tokens": None,
            "processing_time_seconds": None,
            "generated_at": None,
        }
    _update_session_metadata(session_path, {"summaries": summaries})


def _update_session_metadata(session_path: Path, updates: dict):
    """Update fields in a session's metadata.json."""
    metadata_path = session_path / "metadata.json"
    if not metadata_path.exists():
        return

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)

        metadata.update(updates)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error updating session metadata: {e}")


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
