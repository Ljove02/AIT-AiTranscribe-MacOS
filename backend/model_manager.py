"""
Model Manager
=============

Centralized management for all ASR models including:
- Parakeet (NVIDIA NeMo) - uses HuggingFace cache
- Whisper (whisper.cpp) - uses custom storage directory

This module handles:
- Model registry with metadata
- Download status checking
- Model downloading with progress
- Storage path management
"""

import os
import sys
import json
import hashlib
import requests
import time
import threading
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelType(str, Enum):
    """Types of models we support."""
    NEMO = "nemo"       # NVIDIA NeMo (Parakeet)
    WHISPER = "whisper" # whisper.cpp (ggml format)


# Storage directory for our models
def get_storage_dir() -> Path:
    """
    Get the application's model storage directory.

    On macOS: ~/Library/Application Support/AiTranscribe/models
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "AiTranscribe"
    else:
        # Linux/Windows fallback
        base = Path.home() / ".aitranscribe"

    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_whisper_dir() -> Path:
    """Get the directory for whisper.cpp models."""
    whisper_dir = get_storage_dir() / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    return whisper_dir


# ============================================================================
# MODEL REGISTRY
# ============================================================================

# All available models
AVAILABLE_MODELS = {
    # -------------------------------------------------------------------------
    # NVIDIA Nemotron Streaming Model (Optimized for real-time)
    # -------------------------------------------------------------------------
    "nemotron-streaming": {
        "type": ModelType.NEMO,
        "name": "nvidia/nemotron-speech-streaming-en-0.6b",
        "display_name": "Nemotron Streaming",
        "author": "NVIDIA",
        "languages": ["en"],
        "language_names": ["English"],
        "description": "Cache-aware streaming model with sub-100ms latency. Optimized for real-time transcription.",
        "multilingual": False,
        "size_mb": 2000,
        "ram_mb": 3000,
        "streaming_native": True,  # This model is designed for streaming
        "download_url": None,
        "filename": None,
        "sha256": None,
    },

    # -------------------------------------------------------------------------
    # NVIDIA Parakeet Models (NeMo) - Batch transcription
    # -------------------------------------------------------------------------
    "parakeet-v2": {
        "type": ModelType.NEMO,
        "name": "nvidia/parakeet-tdt-0.6b-v2",
        "display_name": "Parakeet TDT 0.6B v2",
        "author": "NVIDIA",
        "languages": ["en"],
        "language_names": ["English"],
        "description": "Fast English-only transcription. Best for batch processing.",
        "multilingual": False,
        "size_mb": 2000,
        "ram_mb": 3000,
        "streaming_native": False,
        "download_url": None,
        "filename": None,
        "sha256": None,
    },
    "parakeet-v3": {
        "type": ModelType.NEMO,
        "name": "nvidia/parakeet-tdt-0.6b-v3",
        "display_name": "Parakeet TDT 0.6B v3",
        "author": "NVIDIA",
        "languages": ["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
                      "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
                      "sl", "es", "sv", "ru", "uk"],
        "language_names": ["Bulgarian", "Croatian", "Czech", "Danish", "Dutch",
                           "English", "Estonian", "Finnish", "French", "German",
                           "Greek", "Hungarian", "Italian", "Latvian", "Lithuanian",
                           "Maltese", "Polish", "Portuguese", "Romanian", "Slovak",
                           "Slovenian", "Spanish", "Swedish", "Russian", "Ukrainian"],
        "description": "Multilingual transcription with automatic language detection. Supports 25 European languages.",
        "multilingual": True,
        "size_mb": 2000,
        "ram_mb": 3000,
        "streaming_native": False,
        "download_url": None,
        "filename": None,
        "sha256": None,
    },

    # -------------------------------------------------------------------------
    # OpenAI Whisper Models (faster-whisper / CTranslate2 format)
    # These are auto-downloaded by faster-whisper from HuggingFace
    # -------------------------------------------------------------------------
    "whisper-base-en": {
        "type": ModelType.WHISPER,
        "name": "base.en",  # faster-whisper model size
        "display_name": "Whisper Base (English)",
        "author": "OpenAI",
        "languages": ["en"],
        "language_names": ["English"],
        "description": "Fast, lightweight English transcription. Great for quick dictation.",
        "multilingual": False,
        "size_mb": 142,
        "ram_mb": 150,
        "streaming_native": False,
        "download_url": None,
        "filename": None,
        "sha256": None,
    },
    "whisper-small-en": {
        "type": ModelType.WHISPER,
        "name": "small.en",  # faster-whisper model size
        "display_name": "Whisper Small (English)",
        "author": "OpenAI",
        "languages": ["en"],
        "language_names": ["English"],
        "description": "Balanced accuracy and speed for English. Recommended for most users.",
        "multilingual": False,
        "size_mb": 466,
        "ram_mb": 200,
        "streaming_native": False,
        "download_url": None,
        "filename": None,
        "sha256": None,
    },
    "whisper-large-v3-turbo": {
        "type": ModelType.WHISPER,
        "name": "deepdml/faster-whisper-large-v3-turbo-ct2",  # HuggingFace model
        "display_name": "Whisper Large v3 Turbo",
        "author": "OpenAI",
        "languages": ["af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
                      "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
                      "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
                      "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
                      "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
                      "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
                      "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
                      "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
                      "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
                      "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"],
        "language_names": ["99+ languages"],
        "description": "Best quality multilingual transcription. Supports 99+ languages with automatic detection.",
        "multilingual": True,
        "size_mb": 1500,
        "ram_mb": 650,
        "streaming_native": False,
        "download_url": None,
        "filename": None,
        "sha256": None,
    },
    "whisper-large-v3": {
        "type": ModelType.WHISPER,
        "name": "large-v3",  # faster-whisper model size
        "display_name": "Whisper Large v3",
        "author": "OpenAI",
        "languages": ["af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
                      "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
                      "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
                      "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
                      "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
                      "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
                      "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
                      "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
                      "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
                      "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"],
        "language_names": ["99+ languages"],
        "description": "Highest quality multilingual model. Best accuracy, larger size.",
        "multilingual": True,
        "size_mb": 2900,
        "ram_mb": 2000,
        "streaming_native": False,
        "download_url": None,
        "filename": None,
        "sha256": None,
    },
}


# ============================================================================
# MODEL STATUS
# ============================================================================

@dataclass
class ModelStatus:
    """Status information for a model."""
    id: str
    downloaded: bool
    size_mb: int
    size_on_disk_mb: Optional[int]
    path: Optional[str]


def check_model_downloaded(model_id: str) -> ModelStatus:
    """
    Check if a model is downloaded and available.

    For NeMo models: Check HuggingFace cache
    For Whisper models: Check HuggingFace cache (faster-whisper auto-downloads)
    """
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    config = AVAILABLE_MODELS[model_id]
    model_type = config["type"]

    if model_type == ModelType.NEMO:
        # NeMo models use HuggingFace cache
        downloaded = check_huggingface_model_cached(config["name"])
        path = get_huggingface_cache_path(config["name"]) if downloaded else None
        size_on_disk = get_directory_size_mb(path) if path else None

    elif model_type == ModelType.WHISPER:
        # Faster-whisper models are auto-downloaded to HuggingFace cache
        # Check if model exists in cache
        model_name = config["name"]

        # For built-in model sizes (base.en, small.en, etc.), check Systran models
        if "/" not in model_name:
            # Standard model sizes are at Systran/faster-whisper-{size}
            hf_name = f"Systran/faster-whisper-{model_name}"
        else:
            hf_name = model_name

        downloaded = check_huggingface_model_cached(hf_name)
        path = get_huggingface_cache_path(hf_name) if downloaded else None
        size_on_disk = get_directory_size_mb(path) if path else None

    else:
        downloaded = False
        path = None
        size_on_disk = None

    return ModelStatus(
        id=model_id,
        downloaded=downloaded,
        size_mb=config["size_mb"],
        size_on_disk_mb=size_on_disk,
        path=path
    )



def check_huggingface_model_cached(model_name: str) -> bool:
    """Check if a HuggingFace model is in the cache.
    
    Checks multiple locations:
    1. HuggingFace hub cache (standard location)
    2. Custom whisper directory (for faster-whisper models)
    """
    import os
    from pathlib import Path
    
    print(f"DEBUG: Checking if model is cached: {model_name}")
    home = str(Path.home())
    
    # Method 1: Check HuggingFace hub cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                print(f"DEBUG: Found {model_name} in HuggingFace cache at {repo.repo_path}")
                return True
    except Exception as e:
        print(f"DEBUG: scan_cache_dir failed: {e}")
    
    # Method 2: Check HuggingFace hub directory manually
    try:
        hub_path = os.path.join(home, ".cache", "huggingface", "hub")
        
        # HuggingFace stores models as "models--{org}--{model}"
        # Replace / with -- for the directory name
        folder_name = f"models--{model_name.replace('/', '--')}"
        model_path = os.path.join(hub_path, folder_name)
        
        print(f"DEBUG: Checking path: {model_path}")
        
        if os.path.exists(model_path):
            # Check if directory has actual model files
            has_files = any(os.scandir(model_path))
            print(f"DEBUG: Path exists, has files: {has_files}")
            if has_files:
                return True
    except Exception as e:
        print(f"DEBUG: Hub path check failed: {e}")
    
    # Method 3: For whisper models, check if files exist in faster-whisper cache
    try:
        if "faster-whisper" in model_name:
            # Extract model size from name (e.g., "base.en" from "Systran/faster-whisper-base.en")
            size = model_name.replace("Systran/faster-whisper-", "").split("/")[0]
            
            # Check multiple possible locations
            possible_paths = [
                os.path.join(home, ".cache", "huggingface", "hub", "models--Systran--faster-whisper", size),
                os.path.join(home, ".cache", "huggingface", "hub", f"models--Systran--faster-whisper-{size}"),
            ]
            
            for path in possible_paths:
                print(f"DEBUG: Checking whisper path: {path}")
                if os.path.exists(path) and any(os.scandir(path)):
                    print(f"DEBUG: Found whisper model at {path}")
                    return True
    except Exception as e:
        print(f"DEBUG: Whisper path check failed: {e}")
    
    print(f"DEBUG: Model {model_name} NOT found in cache")
    return False


def get_huggingface_cache_path(model_name: str) -> Optional[str]:
    """Get the path to a cached HuggingFace model."""
    import os
    from pathlib import Path
    
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                return str(repo.repo_path)
        return None
    except Exception:
        # Fallback: check directory manually
        try:
            home = str(Path.home())
            hub_path = os.path.join(home, ".cache", "huggingface", "hub")
            folder_name = f"models--{model_name.replace('/', '--')}"
            model_path = os.path.join(hub_path, folder_name)
            
            if os.path.exists(model_path):
                return model_path
        except:
            pass
        return None
    except Exception:
        # Fallback: check directory manually
        try:
            home = Path.home()
            hub_path = home / ".cache" / "huggingface" / "hub"
            folder_name = f"models--{model_name.replace('/', '--')}"
            model_path = hub_path / folder_name
            
            if model_path.exists():
                return str(model_path)
        except:
            pass
        return None


def get_directory_size_mb(path: Optional[str]) -> Optional[int]:
    """Get the total size of a directory in MB."""
    if path is None:
        return None

    try:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total += os.path.getsize(filepath)
        return int(total / (1024 * 1024))
    except Exception:
        return None


# ============================================================================
# MODEL DOWNLOADING
# ============================================================================

def download_model(model_id: str) -> Generator[dict, None, None]:
    """
    Download a model with progress updates.

    Yields progress dicts:
    {"status": "downloading", "progress": 0.5, "downloaded_mb": 256, "total_mb": 512}
    {"status": "verifying", "progress": 1.0}
    {"status": "complete", "path": "/path/to/model"}
    {"status": "error", "message": "..."}

    For NeMo models: Triggers HuggingFace download (no granular progress)
    For Whisper models: Downloads with detailed progress
    """
    if model_id not in AVAILABLE_MODELS:
        yield {"status": "error", "message": f"Unknown model: {model_id}"}
        return

    config = AVAILABLE_MODELS[model_id]
    model_type = config["type"]

    if model_type == ModelType.NEMO:
        # NeMo models - trigger HuggingFace download
        yield from download_nemo_model(model_id, config)

    elif model_type == ModelType.WHISPER:
        # Whisper models - direct download with progress
        yield from download_whisper_model(model_id, config)


def download_nemo_model(model_id: str, config: dict) -> Generator[dict, None, None]:
    """Download a NeMo model via HuggingFace with progress tracking via folder size polling."""
    import queue
    
    model_name = config["name"]
    total_mb = config["size_mb"]
    
    # Get cache path for size tracking
    cache_path = get_huggingface_cache_path(model_name)
    initial_size_mb = get_directory_size_mb(cache_path) if cache_path else 0
    
    yield {"status": "downloading", "progress": 0.0, "downloaded_mb": initial_size_mb, "total_mb": total_mb, "message": "Preparing download..."}
    
    # Queue for thread communication
    progress_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def download_thread_func():
        """Background thread that performs the actual download."""
        try:
            import nemo.collections.asr as nemo_asr
            print(f"DEBUG: Starting NeMo model download for {model_name}")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=config["name"])
            print(f"DEBUG: NeMo model {model_name} downloaded successfully")
            del model  # Don't keep in memory
            import gc
            gc.collect()
            result_queue.put(("success", None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_queue.put(("error", str(e)))
    
    def poll_folder_size():
        """Background thread that polls folder size and sends to queue."""
        while True:
            try:
                current_size = get_directory_size_mb(cache_path) if cache_path else 0
                progress_queue.put(("progress", current_size))
                time.sleep(1)  # Poll every 1 second
            except:
                progress_queue.put(("progress", 0))
                time.sleep(1)
    
    # Start download thread
    download_thread = threading.Thread(target=download_thread_func)
    download_thread.start()
    
    # Start polling thread
    poll_thread = threading.Thread(target=poll_folder_size, daemon=True)
    poll_thread.start()
    
    # Wait for download to complete, yielding progress updates
    download_finished = False
    last_reported_size = initial_size_mb
    
    while not download_finished:
        # Check if download thread is done
        download_thread.join(timeout=0.1)
        if not download_thread.is_alive():
            download_finished = True
        
        # Check for progress updates from polling thread
        try:
            while not progress_queue.empty():
                msg_type, value = progress_queue.get_nowait()
                if msg_type == "progress":
                    current_size = value
                    if current_size > last_reported_size:
                        last_reported_size = current_size
                        progress = min(float(current_size) / float(total_mb), 0.95)  # Cap at 95% until complete
                        yield {
                            "status": "downloading", 
                            "progress": progress, 
                            "downloaded_mb": current_size, 
                            "total_mb": total_mb, 
                            "message": f"Downloading... {current_size}/{total_mb} MB"
                        }
        except:
            pass
    
    # Check result
    if not result_queue.empty():
        status, value = result_queue.get()
        if status == "error":
            yield {"status": "error", "message": value or "Download failed"}
            return
    
    # Get final size
    final_size_mb = get_directory_size_mb(cache_path) if cache_path else total_mb
    print(f"DEBUG: NeMo model download complete. Cache path: {cache_path}, size: {final_size_mb} MB")
    
    yield {"status": "complete", "progress": 1.0, "downloaded_mb": final_size_mb, "total_mb": total_mb, "path": cache_path}


def download_whisper_model(model_id: str, config: dict) -> Generator[dict, None, None]:
    """
    Download a Whisper model using faster-whisper with progress tracking via folder size polling.
    
    faster-whisper automatically downloads models from HuggingFace,
    so we track progress by polling the cache folder size.
    """
    import queue
    
    model_name = config["name"]
    total_mb = config["size_mb"]

    # Get cache path for size tracking
    if "/" not in model_name:
        hf_name = f"Systran/faster-whisper-{model_name}"
    else:
        hf_name = model_name
    
    cache_path = get_huggingface_cache_path(hf_name)
    initial_size_mb = get_directory_size_mb(cache_path) if cache_path else 0
    
    yield {"status": "downloading", "progress": 0.0, "downloaded_mb": initial_size_mb, "total_mb": total_mb, "message": "Preparing download..."}

    # Queues for thread communication
    progress_queue = queue.Queue()
    result_queue = queue.Queue()
    
    def download_thread_func():
        """Background thread that performs the actual download."""
        try:
            from faster_whisper import WhisperModel
            print(f"DEBUG: Starting Whisper model download for {model_name}")
            # This will download the model if not cached
            # Use CPU to avoid GPU memory issues during download
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print(f"DEBUG: Whisper model {model_name} downloaded successfully")
            del model
            import gc
            gc.collect()
            result_queue.put(("success", None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_queue.put(("error", str(e)))
    
    def poll_folder_size():
        """Background thread that polls folder size and sends to queue."""
        while True:
            try:
                current_size = get_directory_size_mb(cache_path) if cache_path else 0
                progress_queue.put(("progress", current_size))
                time.sleep(1)  # Poll every 1 second
            except:
                progress_queue.put(("progress", 0))
                time.sleep(1)
    
    # Start download thread
    download_thread = threading.Thread(target=download_thread_func)
    download_thread.start()
    
    # Start polling thread
    poll_thread = threading.Thread(target=poll_folder_size, daemon=True)
    poll_thread.start()
    
    # Wait for download to complete, yielding progress updates
    download_finished = False
    last_reported_size = initial_size_mb
    
    while not download_finished:
        # Check if download thread is done
        download_thread.join(timeout=0.1)
        if not download_thread.is_alive():
            download_finished = True
        
        # Check for progress updates from polling thread
        try:
            while not progress_queue.empty():
                msg_type, value = progress_queue.get_nowait()
                if msg_type == "progress":
                    current_size = value
                    if current_size > last_reported_size:
                        last_reported_size = current_size
                        progress = min(float(current_size) / float(total_mb), 0.95)
                        yield {
                            "status": "downloading", 
                            "progress": progress, 
                            "downloaded_mb": current_size, 
                            "total_mb": total_mb, 
                            "message": f"Downloading... {current_size}/{total_mb} MB"
                        }
        except:
            pass
    
    # Check result
    if not result_queue.empty():
        status, value = result_queue.get()
        if status == "error":
            yield {"status": "error", "message": value or "Download failed"}
            return
    
    # Get final size
    final_size_mb = get_directory_size_mb(cache_path) if cache_path else total_mb
    
    yield {"status": "complete", "progress": 1.0, "downloaded_mb": final_size_mb, "total_mb": total_mb, "path": cache_path}


def delete_model(model_id: str) -> bool:
    """
    Delete a downloaded model from HuggingFace cache.

    Both NeMo and faster-whisper models use HuggingFace cache,
    so they can be deleted the same way.

    Returns True if model was deleted, False if not found.
    """
    import shutil

    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    config = AVAILABLE_MODELS[model_id]

    # Get the HuggingFace model name
    model_name = config["name"]

    # For standard whisper sizes, convert to full HF name
    if config["type"] == ModelType.WHISPER and "/" not in model_name:
        model_name = f"Systran/faster-whisper-{model_name}"

    # The cache is organized as: ~/.cache/huggingface/hub/models--{model_name}
    # Replace slashes with double dashes for the folder name
    cache_folder_name = f"models--{model_name.replace('/', '--')}"
    hf_cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_folder_name

    print(f"Looking for model cache at: {hf_cache_path}")

    if hf_cache_path.exists():
        try:
            # Use protected access to get blob storage path too
            blobs_path = hf_cache_path.parent / f"{cache_folder_name}.blobs"
            refs_path = hf_cache_path.parent / f"{cache_folder_name}.refs"

            # Delete main model folder
            shutil.rmtree(hf_cache_path)
            print(f"Deleted {hf_cache_path}")

            # Delete blobs if exists
            if blobs_path.exists():
                shutil.rmtree(blobs_path)
                print(f"Deleted {blobs_path}")

            # Delete refs if exists
            if refs_path.exists():
                shutil.rmtree(refs_path)
                print(f"Deleted {refs_path}")

            return True
        except Exception as e:
            print(f"Error deleting model cache: {e}")
            raise RuntimeError(f"Failed to delete model files: {e}")
    else:
        print(f"Model cache not found at {hf_cache_path}")
        return False


# ============================================================================
# STORAGE INFO
# ============================================================================

def get_storage_info() -> dict:
    """Get information about model storage."""
    hf_cache_path = Path.home() / ".cache" / "huggingface" / "hub"

    # Calculate total size and model count from HuggingFace cache
    total_size_mb = 0
    model_count = 0

    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()

        # Only count our models
        our_model_names = set()
        for model_id, config in AVAILABLE_MODELS.items():
            model_name = config["name"]
            # Convert standard whisper sizes to full HF names
            if config["type"] == ModelType.WHISPER and "/" not in model_name:
                model_name = f"Systran/faster-whisper-{model_name}"
            our_model_names.add(model_name)

        for repo in cache_info.repos:
            if repo.repo_id in our_model_names:
                total_size_mb += int(repo.size_on_disk / (1024 * 1024))
                model_count += 1

    except Exception:
        pass  # If we can't scan cache, return zeros

    return {
        "storage_path": str(hf_cache_path),
        "whisper_path": str(hf_cache_path),  # Same as storage path now
        "total_size_mb": total_size_mb,
        "model_count": model_count,
        "huggingface_cache": str(hf_cache_path),
    }


# ============================================================================
# MODEL INFO HELPERS
# ============================================================================

def get_model_info(model_id: str) -> dict:
    """Get full information about a model including download status."""
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    config = AVAILABLE_MODELS[model_id].copy()
    status = check_model_downloaded(model_id)

    # Convert enum to string for JSON serialization
    config["type"] = config["type"].value

    return {
        **config,
        "id": model_id,
        "downloaded": status.downloaded,
        "path": status.path,
        "size_on_disk_mb": status.size_on_disk_mb,
    }


def get_all_models_info() -> list[dict]:
    """Get information about all available models."""
    return [get_model_info(model_id) for model_id in AVAILABLE_MODELS]


# ============================================================================
# NEMO AVAILABILITY CHECK
# ============================================================================

def check_nemo_available() -> dict:
    """
    Check if NeMo library can be imported.

    Returns a dict with:
    - available: bool - whether NeMo is available
    - version: str or None - NeMo version if available
    - torch_version: str or None - PyTorch version if available
    - device: str - "mps", "cuda", or "cpu"

    Note: In PyInstaller builds, NeMo import may fail with OSError due to
    TorchScript requiring source code access. This is expected behavior.
    """
    result = {
        "available": False,
        "version": None,
        "torch_version": None,
        "device": "cpu"
    }

    try:
        import nemo
        import nemo.collections.asr as nemo_asr
        result["available"] = True
        result["version"] = nemo.__version__
    except ImportError:
        return result
    except (OSError, RuntimeError, Exception) as e:
        # PyInstaller builds can't load NeMo due to TorchScript source requirements
        print(f"DEBUG: NeMo not available in bundled mode: {type(e).__name__}")
        return result

    try:
        import torch
        result["torch_version"] = torch.__version__

        # Check available devices
        if torch.cuda.is_available():
            result["device"] = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result["device"] = "mps"
        else:
            result["device"] = "cpu"
    except ImportError:
        pass

    return result


def is_nemo_model(model_id: str) -> bool:
    """Check if a model requires NeMo."""
    if model_id not in AVAILABLE_MODELS:
        return False
    return AVAILABLE_MODELS[model_id]["type"] == ModelType.NEMO
