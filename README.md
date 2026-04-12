![Banner](git-banner.png)

# AITranscribe

A powerful macOS menu bar application for local speech-to-text transcription and AI-powered summarization. Runs completely offline using state-of-the-art AI models. Your voice stays on your Mac — nothing is sent to the cloud.

![macOS](https://img.shields.io/badge/macOS-13.0+-blue)
![License](https://img.shields.io/badge/license-MIT%20%2B%20Commons%20Clause-green)
![Swift](https://img.shields.io/badge/Swift-5.9+-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Version](https://img.shields.io/badge/version-0.2.0-purple)
![Stars](https://img.shields.io/github/stars/Ljove02/AIT-AiTranscribe-MacOS)

---

## Quick Install

1. Download the latest **DMG** from [Releases](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/releases)
2. Open the DMG and **drag AiTranscribe to Applications**
3. Run this once in Terminal (app is not signed with Apple Developer certificate):
   ```bash
   xattr -cr /Applications/AiTranscribe.app
   ```

---

## Showcase

https://github.com/user-attachments/assets/a5f44cdb-234e-43ad-b4c2-4f566bf4b369

> Completely redesigned UI with transparent glass cards, capsule buttons, and smooth stagger animations. Navigate through Dashboard, Sessions, History, Models, and more — all from a collapsible sidebar. Summarize recordings with on-device AI, track your stats, and manage everything in one place.

### Recording Indicator

https://github.com/user-attachments/assets/543446b2-f50c-416b-acde-471ee3c46c13

> The floating indicator features GPU-synced side-wave arcs that expand with your voice, a glassy capsule design, and magnetic snap to 8 screen positions.

---

## What's New in v0.2

### AI-Powered Summaries
Summarize any transcription or session recording using on-device LLMs powered by [MLX](https://github.com/ml-explore/mlx). Choose from **Gemma 4** models (E2B 4-bit recommended, E4B for higher quality) with four built-in presets — General, Meeting Notes, Action Items, and Technical. Models download on demand (~3.6GB), run entirely on Apple Silicon, and auto-unload after use to free RAM.

### Dashboard & Analytics
A full statistics hub showing total time saved, words transcribed, average WPM, and transcription count. Dive into weekly patterns, hourly activity heatmaps, WPM trends, transcription length distribution, top words, and model usage — all rendered with animated chart grow-ins.

### Auto-Updates
AiTranscribe now checks GitHub Releases automatically every 24 hours. When a new version is available, an in-app window shows the release notes (rendered Markdown), download progress, and a one-click install that replaces the app and relaunches.

### Redesigned Recording Indicator
A glassy capsule with layered glass highlights and shadow. During recording, side-wave arcs expand outward in response to audio volume at 10Hz via GPU-synced `TimelineView`. During transcription, three dots rotate and pulse. Drag it anywhere — it snaps magnetically to corners, edges, and center with a glassy placeholder preview.

### Similarity Search & History Indexing
Index your entire transcription history and search by meaning using Apple's native `NLEmbedding` from the Natural Language framework. Keyword search is still there, but now you can also find transcriptions that are semantically similar to your query — all processed on-device with zero setup.

### Redesigned Menu Bar Panel
The menu bar dropdown is now a fully custom compact popover (280px) — no more default macOS menu. It shows the app icon, version badge, and a live status dot (connected/offline). The main area is context-aware: idle shows Record and Session buttons, recording shows a red indicator bar with duration and stop/cancel, session shows an orange bar, and transcribing shows a progress spinner. A microphone picker and settings/quit footer round it out.

### Full UI Overhaul
Every screen has been rebuilt from scratch with a transparent bubble design system — glass cards, capsule buttons, stagger animations, and smooth transitions. The settings window now has a collapsible sidebar (compact icon mode or expanded with labels) and all seven tabs (Dashboard, General, Models, Sessions, History, Shortcuts, About) have been redesigned.

---

## Features

### Core

- **100% Local and Private** — All processing happens on your Mac. No internet, no cloud, no data leaves your machine.
- **Global Hotkey** — Press `Control + P` from any app to start recording. Press again to stop. The transcription is in your clipboard before you can blink.
- **Menu Bar App** — Lives in your menu bar, always accessible, never in your way.
- **Multiple AI Models** — NVIDIA Parakeet, OpenAI Whisper (base, small, large-v3, large-v3-turbo), NVIDIA Nemotron streaming, and Gemma 4 for summarization.
- **Auto-Paste** — Transcribed text can be pasted automatically at your cursor position right after transcription.
- **Metal GPU Acceleration** — Whisper models run on Apple Silicon GPU via whisper.cpp, delivering 8-10x faster transcription than CPU-only inference. 10 minutes of audio transcribed in ~2 minutes.

### AI Summaries

- **On-Device LLM Summarization** — Summarize transcriptions and session recordings using Gemma 4 models running locally via MLX on Apple Silicon.
- **Multiple Presets** — General summary, Meeting Notes, Action Items, and Technical — each with tailored system prompts and configurable output length (short, medium, long, custom).
- **Isolated Runtime** — Summary models run in a dedicated Python venv with one-click setup. Models auto-unload after generation to reclaim RAM.
- **Streaming Generation** — Watch the summary generate in real time with token-by-token streaming.

### Session Recording

- **Long-form Recording** — Record sessions of any length (meetings, lectures, interviews). Captures both microphone and system audio simultaneously.
- **RAM-Aware Batch Transcription** — Long recordings are automatically split into chunks sized to your available RAM and transcribed sequentially with live progress streaming.
- **Session Management** — View, rename, delete, and re-transcribe sessions. Bulk actions for clearing transcriptions. Audio and transcription files stored locally.
- **Per-Session Summaries** — Generate summaries for any session with your choice of preset and model. View summaries in a tabbed interface alongside the full transcription.
- **Floating Session Indicator** — A capsule-shaped indicator with pulsing red dot and HH:MM:SS timer. Draggable, snaps to 8 screen positions.

### Dashboard

- **Hero Stats** — Time saved, total words transcribed, average WPM, and transcription count at a glance.
- **Activity Analytics** — Monthly activity bars, weekly pattern breakdown, and hourly activity heatmap.
- **Trends & Distribution** — WPM trends over 8 weeks, transcription length distribution, top words, and most-used models.
- **Animated Charts** — All visualizations animate in with staggered spring grow-ins on first load.

### Smart Model Management

- **Lazy Loading** — Models are not loaded at app startup. When you press the record shortcut, the model loads in the background while you speak. By the time you stop, the transcription is ready.
- **Idle Unloading** — After 2 minutes of inactivity, the model automatically unloads to free RAM. Next time you record, it loads again seamlessly.
- **Streaming models excluded** — NeMo streaming models (Nemotron) stay loaded since they need to be instantly available for real-time transcription.

### Recording

- **Floating Indicator** — A glassy, draggable recording indicator with GPU-synced animations. Side-wave arcs respond to your voice volume. Snaps to screen corners and edges. Adapts to light and dark themes.
- **Real-time Progress** — For longer transcriptions, the UI shows live progress (e.g., "Transcribing... 45%") with real segment tracking for Whisper models.
- **Audio Ducking** — Automatically lowers or mutes system audio while recording, then restores it when you stop.

### Audio Device Management

- **Microphone Selection** — Pick your preferred microphone from the dropdown. Uses native CoreAudio to set the input device.
- **AirPods Support** — When AirPods connect or disconnect, the device list refreshes automatically. If your selected microphone disappears, the app falls back to the default.
- **Persistent Selection** — Your preferred microphone is remembered across app restarts.
- **Device Change Notifications** — macOS notification when the default input device changes.

### Transcription History

- All transcriptions are saved locally with timestamp, duration, word count, and which model was used.
- **Keyword Search** — Filter history instantly by keyword across transcription text and model names.
- **Similarity Search** — Index your entire history with one click and search by meaning, not just keywords. Uses Apple's native `NLEmbedding` (Natural Language framework) for on-device sentence embeddings — no downloads, no cloud. Text matches appear first, then semantic matches, deduplicated.
- View full transcription text, copy to clipboard, and view inline summaries per preset.

### Auto-Updates

- **GitHub Release Checking** — Automatically checks for new versions every 24 hours via the GitHub Releases API.
- **In-App Update Flow** — See rendered release notes, download progress, and one-click install that replaces the app bundle and relaunches.

### Customization

- Configurable keyboard shortcuts for recording, cancelling, and session recording
- Sound feedback when recording starts and stops
- Two audio modes during recording: mute completely, or lower volume by a percentage
- NeMo library installation support for accessing Parakeet models (guided setup in the app)
- Factory reset option in About/System settings

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
| Python 3.10+ (arm64)  | `brew install python@3.11` or [python.org](https://www.python.org/downloads/) |
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
3. Press `Control + P` to start recording
4. Speak into your microphone
5. Press the hotkey again to stop — text is in your clipboard

The model loads automatically the first time you record. After 2 minutes of idle, it unloads to free RAM. Next time you record, it loads again in the background while you speak.

For **session recording**, use `Control + K` to start a long-form session. The floating indicator shows elapsed time. Press again to stop and transcribe.

For **AI summaries**, open any session or transcription, choose a preset (General, Meeting Notes, Action Items, Technical), and generate a summary — all processed locally on your Mac.

---

## Models

### Transcription Models

| Model                    | Download Size | Speed    | Accuracy  | RAM Required | GPU Accelerated |
| ------------------------ | ------------- | -------- | --------- | ------------ | --------------- |
| Whisper Base (EN)        | ~148MB        | Fastest  | Good      | ~400MB       | Yes (Metal)     |
| Whisper Small (EN)       | ~488MB        | Fast     | Very Good | ~850MB       | Yes (Metal)     |
| Whisper Large v3 Turbo   | ~1.6GB        | Fast     | Excellent | ~1.7GB       | Yes (Metal)     |
| Whisper Large v3         | ~3.1GB        | Medium   | Best      | ~4GB         | Yes (Metal)     |
| Parakeet TDT 0.6B       | ~1.2GB        | Fast     | Excellent | ~3GB         | No (CPU)        |
| Nemotron Streaming       | —             | Realtime | Good      | ~2GB         | No (CPU)        |

Whisper models use [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with Metal GPU acceleration for 8-10x faster inference on Apple Silicon. Models are downloaded on-demand and stored locally.

### Summary Models

| Model              | Download Size | Quality   | RAM Required |
| ------------------ | ------------- | --------- | ------------ |
| Gemma 4 E2B (4-bit)| ~3.6GB       | Great     | ~4GB         |
| Gemma 4 E4B (4-bit)| ~5.2GB       | Excellent | ~6GB         |

Summary models run locally via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Installed through a one-click setup in the app with an isolated Python environment.

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
│       ├── AudioRecorder.swift         # Audio recording + CoreAudio
│       ├── MenuBarView.swift           # Menu bar panel UI
│       ├── RecordingIndicator.swift    # GPU-synced floating indicator
│       ├── UpdateChecker.swift         # GitHub release auto-updates
│       ├── SummarySetupManager.swift   # MLX summary runtime installer
│       ├── Sessions/                   # Session recording system
│       │   ├── SessionManager.swift
│       │   ├── SessionRecorder.swift
│       │   └── SystemAudioCapture.swift
│       ├── Settings/                   # Settings UI (modular tabs)
│       │   ├── SettingsView.swift          # Main settings window
│       │   ├── SettingsSidebar.swift        # Collapsible sidebar
│       │   ├── DashboardView.swift         # Analytics dashboard
│       │   ├── DashboardStatsManager.swift # Stats computation
│       │   ├── GeneralSettingsView.swift
│       │   ├── ModelsSettingsView.swift
│       │   ├── SessionsSettingsView.swift
│       │   ├── SessionDetailView.swift
│       │   ├── HistorySettingsView.swift
│       │   ├── ShortcutsSettingsView.swift
│       │   ├── AboutSettingsView.swift
│       │   └── UpdateWindowView.swift      # Auto-update UI
│       ├── HotkeyManager.swift         # Global keyboard shortcuts
│       └── HistoryManager.swift        # Transcription history + search
│
├── backend/                      # Python/FastAPI backend
│   ├── server.py                 # API endpoints + SSE streaming
│   ├── model_manager.py          # Model download and loading
│   ├── summary_manager.py        # MLX summary model management
│   ├── summary_worker.py         # Summary generation worker
│   ├── setup_summary_venv.py     # Summary runtime installer
│   ├── recorder.py               # Server-side recording utilities
│   ├── requirements.txt          # Python dependencies
│   ├── requirements-summary.txt  # Summary runtime dependencies
│   └── build_standalone.sh       # PyInstaller build script
│
├── build_production.sh           # Full production build (backend + app + DMG)
├── run_backend.sh                # Development server script
└── README.md
```

### Tech Stack

- **Frontend**: Swift / SwiftUI (macOS native)
- **Backend**: Python / FastAPI (local HTTP server on port 8765)
- **Transcription**: OpenAI Whisper via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Metal GPU), NVIDIA NeMo (Parakeet)
- **Summarization**: Google Gemma 4 via [MLX](https://github.com/ml-explore/mlx) (Apple Silicon)
- **Audio**: AVFoundation + CoreAudio (Swift), sounddevice (Python)
- **Communication**: HTTP + Server-Sent Events (SSE) for streaming progress
- **Updates**: GitHub Releases API for version checking and in-app updates

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

**Summary setup fails**
Make sure you have native arm64 Python 3.10+ installed (not Rosetta). Run `python3 -c "import platform; print(platform.machine())"` — it should print `arm64`.

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

This project is licensed under the MIT License with the [Commons Clause](https://commonsclause.com/) — you can use, modify, and share the code freely, but you cannot sell it as a commercial product. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for Metal-accelerated Whisper inference
- [OpenAI Whisper](https://github.com/openai/whisper) for Whisper models
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for Parakeet models
- [MLX](https://github.com/ml-explore/mlx) for on-device LLM inference on Apple Silicon
- [Google Gemma](https://ai.google.dev/gemma) for open-weight summary models

---

If you encounter issues or have questions, check the [Troubleshooting](#troubleshooting) section or open an [issue](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/issues/new).
