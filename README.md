![Banner](git-banner.png)

# AiTranscribe

A powerful macOS menu bar application for local speech-to-text transcription. Runs completely offline using state-of-the-art AI models. Your voice stays on your Mac — nothing is sent to the cloud.

![macOS](https://img.shields.io/badge/macOS-13.0+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Swift](https://img.shields.io/badge/Swift-5.9+-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Version](https://img.shields.io/badge/version-0.1.5-purple)
![Stars](https://img.shields.io/github/stars/Ljove02/AIT-AiTranscribe-MacOS)

---

## Showcase

https://github.com/user-attachments/assets/3f8da9f1-6a4b-483a-96ec-75d6239b5fa8

> Press a shortcut, speak, and the transcription lands in your clipboard — ready to paste anywhere. The recording indicator follows you across desktops, snaps to screen corners, and adapts to light and dark mode.

---

## Features

### Core

- **100% Local and Private** — All processing happens on your Mac. No internet, no cloud, no data leaves your machine.
- **Global Hotkey** — Press `Option + Space` from any app to start recording. Press again to stop. The transcription is in your clipboard before you can blink.
- **Menu Bar App** — Lives in your menu bar, always accessible, never in your way.
- **Multiple AI Models** — NVIDIA Parakeet, OpenAI Whisper (base, small, large-v3, large-v3-turbo), and NVIDIA NeMo with streaming support.
- **Auto-Paste** — Transcribed text can be pasted automatically at your cursor position right after transcription.
- **Metal GPU Acceleration** — Whisper models run on Apple Silicon GPU via whisper.cpp, delivering 8-10x faster transcription than CPU-only inference. 10 minutes of audio transcribed in ~2 minutes.

### Session Recording

- **Long-form Recording** — Record sessions of any length (meetings, lectures, interviews). Captures both microphone and system audio simultaneously.
- **Batch Transcription** — Long recordings are automatically split into RAM-aware chunks and transcribed sequentially with live progress streaming.
- **Session Management** — View, rename, delete, and re-transcribe sessions. Audio and transcription files stored locally.
- **Floating Session Indicator** — A capsule-shaped indicator with pulsing red dot and HH:MM:SS timer. Draggable, snaps to 8 screen positions.

### Smart Model Management

- **Lazy Loading** — Models are not loaded at app startup. When you press the record shortcut, the model loads in the background while you speak. By the time you stop, the transcription is ready.
- **Idle Unloading** — After 2 minutes of inactivity, the model automatically unloads to free RAM. Next time you record, it loads again seamlessly.
- **Streaming models excluded** — NeMo streaming models (Nemotron) stay loaded since they need to be instantly available for real-time transcription.

### Recording

- **Floating Indicator** — A minimal, draggable recording indicator that pulses with your voice. Snaps to screen corners for convenience. Adapts to light and dark themes.
- **Real-time Progress** — For longer transcriptions, the UI shows live progress (e.g., "Transcribing... 45%") with real segment tracking for Whisper models.
- **Audio Ducking** — Automatically lowers or mutes system audio while recording, then restores it when you stop.

### Audio Device Management

- **Microphone Selection** — Pick your preferred microphone from the dropdown. Uses native CoreAudio to set the input device.
- **AirPods Support** — When AirPods connect or disconnect, the device list refreshes automatically. If your selected microphone disappears, the app falls back to the default.
- **Persistent Selection** — Your preferred microphone is remembered across app restarts.
- **Device Change Notifications** — macOS notification when the default input device changes.

### Transcription History

- All transcriptions are saved locally with timestamp, duration, word count, and which model was used.
- Tap any entry to view the full transcription text and copy it to clipboard.
- Search through your transcription history to find past recordings.

### Customization

- Configurable keyboard shortcuts for recording and cancelling
- Sound feedback when recording starts and stops
- Two audio modes during recording: mute completely, or lower volume by a percentage
- NeMo library installation support for accessing Parakeet models (guided setup in the app)

---

## Installation

### Option 1: Download Pre-built App

1. Download the latest DMG from [Releases](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/releases)

2. Open the DMG and drag AiTranscribe to Applications

3. First launch — since the app is not signed with an Apple Developer certificate:

   Right-click AiTranscribe in Applications, select "Open", click "Open" in the dialog. You only need to do this once.

   Or run in Terminal:
   ```bash
   xattr -cr /Applications/AiTranscribe.app
   ```

4. Grant permissions when prompted:
   - Microphone access (required)
   - Accessibility access (optional — enables auto-paste at cursor)

### Option 2: Build from Source

Building locally avoids all Gatekeeper warnings since the app is built on your machine.

**Prerequisites**

| Requirement            | How to Install                                                                |
| ---------------------- | ----------------------------------------------------------------------------- |
| Xcode 15+             | [App Store](https://apps.apple.com/app/xcode/id497799835)                     |
| Python 3.10+          | `brew install python@3.11` or [python.org](https://www.python.org/downloads/) |
| Command Line Tools    | `xcode-select --install`                                                      |

**Quick Build**

```bash
git clone https://github.com/Ljove02/AIT-AiTranscribe-MacOS.git
cd AIT-AiTranscribe-MacOS
python3 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt
./build_production.sh
```

The built app and DMG will be in the `dist/` folder.

**Step-by-Step Build**

```bash
# 1. Clone
git clone https://github.com/Ljove02/AIT-AiTranscribe-MacOS.git
cd AIT-AiTranscribe-MacOS

# 2. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# 3. Build backend executable
cd backend && ./build_standalone.sh && cd ..

# 4. Build everything and create DMG
./build_production.sh

# 5. Install
cp -R dist/AiTranscribe.app /Applications/
```

---

## Usage

1. Click the menu bar icon (top-right of your screen)
2. Go to Settings and download a model:
   - **Parakeet TDT 0.6B** — Recommended. Best balance of speed and accuracy.
   - **Whisper Base (English)** — Lightweight, good for quick transcriptions.
   - **Whisper Large v3** — Highest accuracy, requires more RAM.
3. Press `Option + Space` to start recording
4. Speak into your microphone
5. Press the hotkey again to stop — text is in your clipboard

The model loads automatically the first time you record. After 2 minutes of idle, it unloads to free RAM. Next time you record, it loads again in the background while you speak.

---

## Models

| Model             | Download Size | Speed    | Accuracy  | RAM Required | GPU Accelerated |
| ----------------- | ------------- | -------- | --------- | ------------ | --------------- |
| Whisper Base (EN)  | ~148MB        | Fastest  | Good      | ~400MB       | Yes (Metal)     |
| Whisper Small (EN) | ~488MB        | Fast     | Very Good | ~850MB       | Yes (Metal)     |
| Whisper Large v3 Turbo | ~1.6GB   | Fast     | Excellent | ~1.7GB       | Yes (Metal)     |
| Whisper Large v3   | ~3.1GB        | Medium   | Best      | ~4GB         | Yes (Metal)     |
| Parakeet TDT 0.6B | ~1.2GB        | Fast     | Excellent | ~3GB         | No (CPU)        |

Whisper models use [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with Metal GPU acceleration for 8-10x faster inference on Apple Silicon. Models are downloaded on-demand and stored locally.

---

## Development

For contributors who want to run the app with hot-reload:

**Terminal 1: Start the backend**
```bash
source venv/bin/activate
./run_backend.sh --dev
```

**Terminal 2: Run the frontend**
```bash
open AiTranscribe/AiTranscribe.xcodeproj
# In Xcode: Press Cmd+R to run
```

### Project Structure

```
AIT-AiTranscribe-MacOS/
├── AiTranscribe/                 # Swift/SwiftUI frontend
│   ├── AiTranscribe.xcodeproj
│   └── AiTranscribe/
│       ├── AiTranscribeApp.swift       # App entry point
│       ├── AppState.swift              # Central state management
│       ├── APIClient.swift             # HTTP client for backend
│       ├── BackendManager.swift        # Backend process management
│       ├── AudioRecorder.swift         # Audio recording + CoreAudio device management
│       ├── MenuBarView.swift           # Menu bar UI
│       ├── RecordingIndicator.swift    # Floating recording indicator
│       ├── FloatingIndicatorWindow.swift # macOS 26 display cycle fix
│       ├── Sessions/                    # Session recording system
│       │   ├── SessionManager.swift
│       │   ├── SessionRecorder.swift
│       │   └── SessionIndicatorView.swift
│       ├── Settings/                    # Settings UI (modular tabs)
│       ├── HotkeyManager.swift         # Global keyboard shortcuts
│       └── ...
│
├── backend/                      # Python/FastAPI backend
│   ├── server.py                 # API endpoints + SSE streaming
│   ├── model_manager.py          # Model download and loading
│   ├── recorder.py               # Server-side recording utilities
│   ├── requirements.txt          # Python dependencies
│   └── build_standalone.sh       # PyInstaller build script
│
├── build_production.sh           # Full production build (backend + app + DMG)
├── run_backend.sh                # Development server script
└── README.md
```

### Tech Stack

- **Frontend**: Swift / SwiftUI (macOS native)
- **Backend**: Python / FastAPI (local HTTP server on port 8765)
- **AI Models**: OpenAI Whisper via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Metal GPU), NVIDIA NeMo (Parakeet)
- **Audio**: AVFoundation + CoreAudio (Swift), sounddevice (Python)
- **Communication**: HTTP + Server-Sent Events (SSE) for streaming progress

---

## Troubleshooting

**"App is damaged and can't be opened"**
```bash
xattr -cr /Applications/AiTranscribe.app
```

**"Microphone access denied"**
Go to System Settings > Privacy & Security > Microphone and enable AiTranscribe.

**Backend won't start**
```bash
lsof -i :8765              # Check if port is in use
lsof -ti:8765 | xargs kill # Kill existing process, then restart the app
```

**Xcode build fails**
```bash
xcode-select --install     # Make sure Command Line Tools are installed
cd AiTranscribe && xcodebuild clean
```

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Set up the development environment (see [Development](#development))
4. Make your changes and test thoroughly
5. Commit and push: `git push origin feature/your-feature`
6. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for Metal-accelerated Whisper inference
- [OpenAI Whisper](https://github.com/openai/whisper) for Whisper models
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for Parakeet models

---

If you encounter issues or have questions, check the [Troubleshooting](#troubleshooting) section or open an [issue](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/issues/new).
