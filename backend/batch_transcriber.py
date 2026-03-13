"""
Batch Transcriber
=================

Handles chunked transcription of long audio recordings.

The core problem: Long recordings (30min-2hr+) are too large to transcribe in one shot.
Models like Parakeet need ~3GB RAM just to be loaded. Processing a 60-minute audio file
on top of that could spike RAM beyond what most Macs have.

Solution: Chop the audio into smaller time-based chunks, transcribe each sequentially,
and concatenate the text results. This keeps RAM usage flat regardless of recording length.

Chunk boundaries use a small overlap (3 seconds) to avoid cutting words mid-utterance.
The overlap text is deduplicated using longest common subsequence matching.
"""

import os
import gc
import time
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from scipy.io.wavfile import read as wav_read, write as wav_write

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================================
# CHUNK DURATION LOOKUP TABLE
# ============================================================================
# Maps available_ram_mb (after subtracting model RAM) to chunk_duration_seconds.
# These are conservative defaults for parakeet-v2. Refine via benchmarking.
# The tradeoff: more available RAM = bigger chunks = fewer batches = faster.

CHUNK_DURATION_TABLE = {
    "parakeet-v2": {
        # available_ram_mb: chunk_duration_seconds
        256:  90,    # 1.5 minutes - very tight RAM
        512:  120,   # 2 minutes
        1024: 180,   # 3 minutes
        2048: 300,   # 5 minutes
        3072: 420,   # 7 minutes
        4096: 600,   # 10 minutes
        6144: 900,   # 15 minutes
        8192: 1200,  # 20 minutes
    },
    "whisper-large-v3": {
        # Whisper Large v3 is ~4-6x slower than parakeet — keep chunks small
        # (~120s) for responsive cancellation, stable RAM, and better quality.
        # Each 120s chunk takes ~3-4 min to transcribe on M-series Macs.
        256:  60,    # 1 minute
        512:  90,    # 1.5 minutes
        1024: 120,   # 2 minutes  (default for most configs)
        2048: 120,   # 2 minutes  (cap here — bigger chunks don't help much)
        3072: 120,   # 2 minutes
        4096: 120,   # 2 minutes
        6144: 150,   # 2.5 minutes
        8192: 180,   # 3 minutes
    },
    "whisper-large-v3-turbo": {
        # Turbo is faster than full v3 but slower than parakeet
        256:  90,    # 1.5 minutes
        512:  120,   # 2 minutes
        1024: 180,   # 3 minutes
        2048: 300,   # 5 minutes
        3072: 420,   # 7 minutes
        4096: 600,   # 10 minutes
        6144: 900,   # 15 minutes
        8192: 1200,  # 20 minutes
    },
}

# Model RAM usage (MB) — how much RAM the loaded model occupies
MODEL_RAM_MB = {
    "parakeet-v2": 3000,
    "parakeet-v3": 3000,
    "nemotron-streaming": 3000,
    "whisper-base-en": 400,     # whisper.cpp Metal
    "whisper-small-en": 850,    # whisper.cpp Metal
    "whisper-large-v3-turbo": 1700,  # whisper.cpp Metal
    "whisper-large-v3": 4000,   # whisper.cpp Metal — more efficient than CPU-only CTranslate2
}

# Overlap between chunks in seconds
CHUNK_OVERLAP_SECONDS = 3

# Number of words to compare for overlap deduplication
OVERLAP_DEDUP_WORDS = 20


def get_sessions_dir() -> Path:
    """Get the sessions directory path (matches Swift's SessionManager)."""
    base = Path.home() / "Library" / "Application Support" / "AiTranscribe"
    sessions_dir = base / "Sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def lookup_chunk_duration(available_ram_mb: int, model_id: str) -> int:
    """
    Look up the optimal chunk duration for a given available RAM and model.

    Uses the closest lower entry in the lookup table.
    Falls back to a conservative 120 seconds if model not in table.
    """
    table = CHUNK_DURATION_TABLE.get(model_id, CHUNK_DURATION_TABLE.get("parakeet-v2", {}))

    # Find the largest key that doesn't exceed available_ram_mb
    best_duration = 120  # conservative default: 2 minutes
    for ram_threshold, duration in sorted(table.items()):
        if ram_threshold <= available_ram_mb:
            best_duration = duration
        else:
            break

    return best_duration


def convert_m4a_to_wav(m4a_path: str, wav_path: Optional[str] = None) -> str:
    """
    Convert M4A to WAV 16kHz mono using macOS built-in afconvert.

    afconvert is available on all macOS installations — no external dependencies needed.
    """
    if wav_path is None:
        wav_path = tempfile.mktemp(suffix=".wav")

    result = subprocess.run(
        [
            "afconvert",
            "-d", "LEI16",     # Linear PCM, 16-bit signed integer, little-endian
            "-f", "WAVE",      # WAV format
            "-c", "1",         # Mono
            "-r", "16000",     # 16kHz sample rate
            m4a_path,
            wav_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"afconvert failed: {result.stderr}")

    return wav_path


def get_audio_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    sample_rate, audio_data = wav_read(wav_path)
    return len(audio_data) / sample_rate


def split_audio_chunk(wav_path: str, start_sec: float, end_sec: float) -> str:
    """
    Extract a chunk from a WAV file between start_sec and end_sec.

    Returns path to a temporary WAV file containing the chunk.
    """
    sample_rate, audio_data = wav_read(wav_path)

    start_sample = int(start_sec * sample_rate)
    end_sample = min(int(end_sec * sample_rate), len(audio_data))

    chunk_data = audio_data[start_sample:end_sample]

    chunk_path = tempfile.mktemp(suffix=".wav")
    wav_write(chunk_path, sample_rate, chunk_data)

    return chunk_path


def deduplicate_overlap(prev_text: str, curr_text: str, n_words: int = OVERLAP_DEDUP_WORDS) -> str:
    """
    Remove duplicate text at the boundary between two consecutive chunks.

    Takes the last n_words from prev_text and first n_words from curr_text,
    finds the longest common subsequence of words, and removes the overlap
    from the beginning of curr_text.
    """
    if not prev_text or not curr_text:
        return curr_text

    prev_words = prev_text.split()
    curr_words = curr_text.split()

    if not prev_words or not curr_words:
        return curr_text

    # Take the tail of prev and head of curr for comparison
    prev_tail = prev_words[-n_words:] if len(prev_words) >= n_words else prev_words
    curr_head = curr_words[:n_words] if len(curr_words) >= n_words else curr_words

    # Find the longest matching overlap
    # We look for the longest suffix of prev_tail that matches a prefix of curr_head
    best_overlap_len = 0
    for overlap_len in range(1, min(len(prev_tail), len(curr_head)) + 1):
        # Check if last overlap_len words of prev match first overlap_len of curr
        if prev_tail[-overlap_len:] == curr_head[:overlap_len]:
            best_overlap_len = overlap_len

    if best_overlap_len > 0:
        # Remove the overlapping words from the start of curr_text
        return " ".join(curr_words[best_overlap_len:])

    return curr_text


# Persistent process handle so cpu_percent() has a baseline to measure against.
# Creating a new psutil.Process() each call returns 0% because there's no prior reading.
_stats_process = None

def get_system_stats() -> dict:
    """Get current CPU and memory usage stats."""
    global _stats_process
    stats = {"cpu_percent": 0.0, "memory_mb": 0.0}

    if psutil is not None:
        try:
            if _stats_process is None:
                _stats_process = psutil.Process()
                _stats_process.cpu_percent()  # Prime the baseline (returns 0 on first call)
            stats["cpu_percent"] = _stats_process.cpu_percent()
            stats["memory_mb"] = _stats_process.memory_info().rss / (1024 * 1024)
        except Exception:
            _stats_process = None

    return stats


class BatchTranscriber:
    """
    Transcribes long audio files in chunks, yielding progress events.

    Usage:
        transcriber = BatchTranscriber(audio_path, model_id, ram_budget_mb, transcribe_fn)
        for event in transcriber.transcribe():
            # event is a dict, yield as SSE
            print(event)
    """

    def __init__(
        self,
        audio_path: str,
        model_id: str,
        ram_budget_mb: int,
        transcribe_fn,
    ):
        """
        Args:
            audio_path: Path to the audio file (M4A or WAV)
            model_id: ID of the model to use (e.g., "parakeet-v2")
            ram_budget_mb: Total RAM budget in MB
            transcribe_fn: Function that takes a WAV path and returns (text, processing_time)
        """
        self.audio_path = audio_path
        self.model_id = model_id
        self.ram_budget_mb = ram_budget_mb
        self.transcribe_fn = transcribe_fn
        self.cancelled = False

        # Calculate chunk parameters
        model_ram = MODEL_RAM_MB.get(model_id, 3000)
        available_ram = max(ram_budget_mb - model_ram, 256)
        self.chunk_duration = lookup_chunk_duration(available_ram, model_id)
        self.overlap = CHUNK_OVERLAP_SECONDS

        print(f"BatchTranscriber: model_ram={model_ram}MB, available={available_ram}MB, "
              f"chunk_duration={self.chunk_duration}s, overlap={self.overlap}s")

    def cancel(self):
        """Signal the transcriber to stop after the current chunk."""
        self.cancelled = True

    def transcribe(self) -> Generator[dict, None, None]:
        """
        Main transcription generator. Yields SSE event dicts.

        Events:
        - {"event": "started", "total_batches": N, "chunk_duration": D}
        - {"event": "batch_progress", "batch": N, "total": M, "percent": P}
        - {"event": "batch_complete", "batch": N, "total": M, "batch_text": "..."}
        - {"event": "stats", "cpu_percent": X, "memory_mb": Y, "eta_seconds": Z}
        - {"event": "done", "full_text": "...", "total_batches": N, "total_time": T}
        - {"event": "error", "message": "..."}
        - {"event": "cancelled", "partial_text": "...", "batches_completed": N}
        """
        wav_path = None
        temp_wav = False
        overall_start = time.time()

        try:
            # Step 1: Convert to WAV if needed
            if self.audio_path.lower().endswith((".m4a", ".mp4", ".aac")):
                print(f"BatchTranscriber: Converting {self.audio_path} to WAV...")
                wav_path = convert_m4a_to_wav(self.audio_path)
                temp_wav = True
                print(f"BatchTranscriber: Converted to {wav_path}")
            elif self.audio_path.lower().endswith(".wav"):
                wav_path = self.audio_path
            else:
                yield {"event": "error", "message": f"Unsupported audio format: {self.audio_path}"}
                return

            # Step 2: Get audio duration and calculate batches
            total_duration = get_audio_duration(wav_path)
            step = self.chunk_duration - self.overlap
            if step <= 0:
                step = self.chunk_duration

            num_batches = max(1, int(np.ceil(total_duration / step)))

            print(f"BatchTranscriber: audio={total_duration:.1f}s, batches={num_batches}, "
                  f"step={step}s")

            yield {
                "event": "started",
                "total_batches": num_batches,
                "chunk_duration": self.chunk_duration,
                "total_audio_seconds": round(total_duration, 1),
            }

            # Step 3: Process each chunk
            full_text = ""
            batch_times = []

            for i in range(num_batches):
                if self.cancelled:
                    yield {
                        "event": "cancelled",
                        "partial_text": full_text.strip(),
                        "batches_completed": i,
                    }
                    return

                start_sec = i * step
                end_sec = min(start_sec + self.chunk_duration, total_duration)

                # Progress event before processing
                percent = round((i / num_batches) * 100, 1)
                yield {
                    "event": "batch_progress",
                    "batch": i + 1,
                    "total": num_batches,
                    "percent": percent,
                }

                # Extract and transcribe chunk
                chunk_path = None
                try:
                    chunk_path = split_audio_chunk(wav_path, start_sec, end_sec)
                    batch_start = time.time()

                    chunk_text, _ = self.transcribe_fn(chunk_path)
                    chunk_text = chunk_text.strip() if chunk_text else ""

                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)

                finally:
                    # Clean up chunk temp file
                    if chunk_path and os.path.exists(chunk_path):
                        os.unlink(chunk_path)

                # Deduplicate overlap with previous chunk
                if i > 0 and self.overlap > 0 and chunk_text:
                    chunk_text = deduplicate_overlap(full_text, chunk_text)

                # Append to full text
                if chunk_text:
                    if full_text and not full_text.endswith(" "):
                        full_text += " "
                    full_text += chunk_text

                # Batch complete event
                yield {
                    "event": "batch_complete",
                    "batch": i + 1,
                    "total": num_batches,
                    "batch_text": chunk_text,
                }

                # Stats event with ETA
                stats = get_system_stats()
                avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                remaining_batches = num_batches - (i + 1)
                eta_seconds = avg_batch_time * remaining_batches

                yield {
                    "event": "stats",
                    "cpu_percent": round(stats["cpu_percent"], 1),
                    "memory_mb": round(stats["memory_mb"], 1),
                    "eta_seconds": round(eta_seconds, 1),
                    "avg_batch_time": round(avg_batch_time, 1),
                }

                # Free memory between chunks
                gc.collect()

            # Step 4: Done
            total_time = time.time() - overall_start
            full_text = full_text.strip()

            yield {
                "event": "done",
                "full_text": full_text,
                "total_batches": num_batches,
                "total_time": round(total_time, 1),
                "word_count": len(full_text.split()) if full_text else 0,
            }

        except Exception as e:
            print(f"BatchTranscriber error: {e}")
            import traceback
            traceback.print_exc()
            yield {"event": "error", "message": str(e)}

        finally:
            # Clean up converted WAV if we created one
            if temp_wav and wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)


def estimate_batches(audio_duration_seconds: float, model_id: str, ram_budget_mb: int) -> dict:
    """
    Estimate the number of batches and processing time for a given audio duration and budget.

    Returns a dict with estimation info for the UI.
    """
    model_ram = MODEL_RAM_MB.get(model_id, 3000)
    available_ram = max(ram_budget_mb - model_ram, 256)
    chunk_duration = lookup_chunk_duration(available_ram, model_id)
    step = chunk_duration - CHUNK_OVERLAP_SECONDS
    if step <= 0:
        step = chunk_duration

    num_batches = max(1, int(np.ceil(audio_duration_seconds / step)))

    # Realtime factor: how many seconds of audio are processed per second of wall time.
    # Higher = faster. whisper.cpp with Metal GPU is significantly faster than CPU-only.
    # These estimates are for M-series Macs — actual speed depends on GPU tier.
    REALTIME_FACTORS = {
        "parakeet-v2": 3.5,
        "parakeet-v3": 3.5,
        "nemotron-streaming": 3.5,
        "whisper-base-en": 15.0,   # whisper.cpp Metal — very fast
        "whisper-small-en": 10.0,  # whisper.cpp Metal
        "whisper-large-v3-turbo": 8.0,  # whisper.cpp Metal
        "whisper-large-v3": 4.0,   # whisper.cpp Metal (was 0.6x on CPU!)
    }
    realtime_factor = REALTIME_FACTORS.get(model_id, 1.0)
    estimated_time = audio_duration_seconds / realtime_factor

    return {
        "num_batches": num_batches,
        "chunk_duration_seconds": chunk_duration,
        "estimated_time_seconds": round(estimated_time, 1),
        "available_ram_mb": available_ram,
        "model_ram_mb": model_ram,
    }
