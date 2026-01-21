"""
Audio Recorder Module
=====================

This module handles audio recording from the microphone.

WHY SEPARATE THIS?
------------------
Good code practice: "Separation of Concerns"
- server.py handles HTTP requests
- recorder.py handles audio recording
- Each file has ONE job

This makes code:
1. Easier to test (test recording separately)
2. Easier to maintain (changes in one place)
3. Easier to understand (smaller files)

HOW AUDIO RECORDING WORKS:
--------------------------
1. We use 'sounddevice' library to access the microphone
2. Audio comes in as a stream of numbers (samples)
3. Sample rate = how many numbers per second (16000 = 16kHz)
4. We collect samples in a buffer until recording stops
5. Convert to WAV format for the model
"""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, Callable
from scipy.io.wavfile import write as wav_write
import tempfile
import os


# Audio configuration
SAMPLE_RATE = 16000      # 16kHz - standard for speech
CHANNELS = 1             # Mono (1 channel, not stereo)
DTYPE = np.float32       # Data type for audio samples


class AudioRecorder:
    """
    A class to handle audio recording.

    WHY A CLASS?
    ------------
    A class bundles related data and functions together.
    - Data: recording_buffer, is_recording, etc.
    - Functions: start(), stop(), get_audio(), etc.

    This is cleaner than having global variables scattered around.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        Initialize the recorder.

        __init__ is the "constructor" - runs when you create an instance:
            recorder = AudioRecorder()  # This calls __init__
        """
        self.sample_rate = sample_rate
        self.recording_buffer: list = []
        self.is_recording: bool = False
        self.stream: Optional[sd.InputStream] = None

        # For tracking recording stats
        self.start_time: Optional[float] = None
        self.current_volume: float = 0.0

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Called automatically when new audio data arrives.

        This is a "callback" - a function that gets called by another
        piece of code (sounddevice) when something happens (new audio).

        The underscore prefix (_audio_callback) is a Python convention
        meaning "this is internal/private, don't call directly".

        Args:
            indata: The audio data (numpy array)
            frames: Number of frames in this chunk
            time_info: Timing information
            status: Any errors/warnings
        """
        if status:
            print(f"Audio status warning: {status}")

        if self.is_recording:
            # Make a copy! indata gets reused by sounddevice
            self.recording_buffer.append(indata.copy())

            # Calculate volume using RMS (root mean square) - better loudness measure
            # Then use peak detection for more responsive feedback
            rms = np.sqrt(np.mean(indata ** 2))
            peak = np.abs(indata).max()
            # Blend RMS and peak for responsive but stable volume
            self.current_volume = (rms * 0.5 + peak * 0.5)

    def start(self) -> bool:
        """
        Start recording audio.

        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            return False  # Already recording

        # Clear previous recording
        self.recording_buffer = []
        self.is_recording = True
        self.start_time = time.time()

        # Open audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._audio_callback
            )
            self.stream.start()
            return True

        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.is_recording = False
            return False

    def stop(self) -> Optional[np.ndarray]:
        """
        Stop recording and return the audio data.

        Returns:
            Numpy array of audio samples, or None if no audio
        """
        if not self.is_recording:
            return None

        self.is_recording = False

        # Stop and close the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Combine all recorded chunks
        if len(self.recording_buffer) == 0:
            return None

        # np.concatenate joins all the chunks into one array
        audio_data = np.concatenate(self.recording_buffer).flatten()

        return audio_data

    def get_duration(self) -> float:
        """Get current recording duration in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_volume(self) -> float:
        """Get current audio volume (0.0 to 1.0 roughly)."""
        return self.current_volume

    def save_to_wav(self, audio_data: np.ndarray, path: Optional[str] = None) -> str:
        """
        Save audio data to a WAV file.

        Args:
            audio_data: Numpy array of audio samples
            path: Where to save (or None for temp file)

        Returns:
            Path to the saved WAV file
        """
        if path is None:
            # Create a temporary file
            fd, path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)

        # Convert float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
        # WAV files typically use int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        wav_write(path, self.sample_rate, audio_int16)

        return path


def list_audio_devices() -> list[dict]:
    """
    List all available audio input devices.

    Returns:
        List of device info dictionaries
    """
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        # max_input_channels > 0 means it's an input device
        if device['max_input_channels'] > 0:
            input_devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate'],
                'is_default': i == sd.default.device[0]  # [0] is input default
            })

    return input_devices


def get_default_input_device() -> Optional[dict]:
    """Get the default audio input device."""
    try:
        device = sd.query_devices(kind='input')
        return {
            'name': device['name'],
            'channels': device['max_input_channels'],
            'sample_rate': device['default_samplerate']
        }
    except Exception:
        return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Test the recorder directly.
    Run: python recorder.py
    """
    print("Audio Devices:")
    print("-" * 40)
    for device in list_audio_devices():
        default = " (DEFAULT)" if device['is_default'] else ""
        print(f"  [{device['id']}] {device['name']}{default}")
    print()

    print("Testing recorder...")
    recorder = AudioRecorder()

    input("Press ENTER to start recording...")
    recorder.start()
    print("Recording... Press ENTER to stop")

    input()
    audio = recorder.stop()

    if audio is not None:
        duration = len(audio) / SAMPLE_RATE
        print(f"Recorded {duration:.1f} seconds")

        path = recorder.save_to_wav(audio, "test_recording.wav")
        print(f"Saved to: {path}")
    else:
        print("No audio recorded")
