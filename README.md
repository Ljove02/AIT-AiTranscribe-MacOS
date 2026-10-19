![Banner](git-banner.png)

# AITranscribe

A powerful macOS menu bar application for speech-to-text transcription, AI text transformation, and summarization. **Local-first**: everything runs offline on your Mac with state-of-the-art on-device models — and if you want, plug in cloud providers (OpenAI, ElevenLabs, OpenRouter) for transcription and language models. You choose, per feature, where your audio goes.

![macOS](https://img.shields.io/badge/macOS-13.0+-blue)
![License](https://img.shields.io/badge/license-GPL--3.0-green)
![Swift](https://img.shields.io/badge/Swift-5.9+-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Version](https://img.shields.io/badge/version-0.3.0-purple)
![Stars](https://img.shields.io/github/stars/Ljove02/AIT-AiTranscribe-MacOS)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20this%20project-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/veljkospasic)

---

## Quick Install

1. Download the latest **DMG** from [Releases](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/releases)
2. Open the DMG and **drag AiTranscribe to Applications**
3. Run this once in Terminal (app is not signed with an Apple Developer certificate):
   ```bash
   xattr -cr /Applications/AiTranscribe.app
   ```

---

## Showcase


https://github.com/user-attachments/assets/583bdd76-36cb-4922-b42d-7296a6272ac3

> Completely redesigned UI with transparent glass cards, capsule buttons, and smooth stagger animations. Navigate through Dashboard, Sessions, History, Models, Polish, and more — all from a collapsible sidebar. Transform text with custom AI polishes, summarize recordings with on-device AI, track your stats, and manage everything in one place.

### Recording Indicator


https://github.com/user-attachments/assets/3d57b6e5-7efa-4ca1-951d-de7df93e3556


> The floating indicator features GPU-synced side-wave arcs that expand with your voice, a glassy capsule design, magnetic snap to 8 screen positions — and during sessions, live dBFS meter arcs for mic and desktop audio.

<!-- VIDEO SLOT: Polish flow showcase (record → polish → floating markdown response) -->

---

## What's New in v0.3

### The Polish System
Custom LLM text transformations for your transcriptions. Build **polish presets** with their own prompts, models, and global shortcuts — rewrite dictation as an email, turn rambling into bullet points, anything you can describe. A **6-phase AI generator** builds polishes from a plain-language description (with eval scorecards and your approval before saving). **Polish Routes = fully automatic per-app routing** — dictate a Slack reply and your messaging polish formats it; switch to your coding agent and the same hotkey now produces a clean prompt. **Multi-step modules** compose a polish from input modules (voice, desktop audio, clipboard, screenshot, active app) and output modules (copy, paste image + text, floating window) — you can even bypass the LLM entirely for pure capture pipelines. Results appear in a floating window with a hand-rolled Markdown renderer including native LaTeX math.

### Cloud Providers
Optional API support for **OpenAI, ElevenLabs, and OpenRouter** transcription models, plus **OpenAI and OpenRouter language models** for polishes and summaries. Keys live in the macOS Keychain, custom model entries are fully manageable in Settings, and known models show per-minute cost hints. Local stays the default — the cloud is opt-in.

### Rolling Transcription
Quick recordings transcribe in ~30-second chunks **while you're still speaking** — locally or via API — so the final text often lands **in under a second** after you stop. API uploads are fully resilient: retries, sequential re-sends, full-file fallback, and a FailedRecordings rescue so a long recording can never be silently lost.

### Session Studio
Auto-gain is gone — sessions record at true level with a **manual mixer**: OBS-style dBFS meters and −40…+40 dB faders for mic and desktop audio, right in the menu bar, with a soft limiter so boosts don't clip. **Session automation** auto-transcribes (and optionally auto-summarizes) when you stop. **Session import** turns MP3/WAV/M4A/AAC/AIFF/MP4/MOV files into regular sessions.

### Performance Overhaul
Recording CPU went from **~108% to ~14%** across three optimization waves: fixed a leaked indicator window that rendered forever, isolated 10 Hz recording stats from the app-wide view tree, cached keychain checks, made settings tabs instant, removed main-thread AppleScript polling, and detached heavy stop-path work. Idle is actually idle now.

### New Onboarding
A rebuilt branching first-run flow — Local, API, or Both — with a live mic preview, Keychain API setup with dated model recommendations, shortcut setup, and a built-in test transcription.

### Bluetooth Done Right
Ducking restores volumes **per Bluetooth profile** (A2DP vs HFP), and a new default-on setting records from the built-in mic when your selected mic is Bluetooth — so your music never cuts out and your headphones stay in high quality.

### A New Look
The recording indicator now comes in **six styles** — Classic, Glow, Aurora, Prism, Graphite, Ember — with **noise floor calibration** so the meter reacts to your voice, not your room. The whole UI got a consistency pass: capsule pill dropdowns everywhere, redesigned model sheets and preset managers, and a GitHub-style **activity grid** on the Dashboard.

See the full [release notes](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/releases) for everything.

---

## Features

### Core

- **Local-first and private** — Every feature works 100% on-device: transcription, polishes, summaries. No internet required, no data leaves your machine unless you explicitly configure a cloud provider.
- **Global Hotkey** — Press `Control + P` from any app to start recording. Press again to stop. The transcription is in your clipboard before you can blink.
- **Rolling transcription** — Long dictations transcribe in chunks while you speak; stopping returns the full text almost instantly, often sub-second.
- **Menu Bar App** — Lives in your menu bar, always accessible, never in your way.
- **Multiple AI Models** — NVIDIA Parakeet, OpenAI Whisper (base, small, large-v3, large-v3-turbo), NVIDIA Nemotron streaming, Gemma 4 for summarization — plus optional API models from OpenAI, ElevenLabs, and OpenRouter.
- **Auto-Paste** — Transcribed text can be pasted automatically at your cursor position right after transcription.
- **Metal GPU Acceleration** — Whisper models run on Apple Silicon GPU via whisper.cpp, delivering 8-10x faster transcription than CPU-only inference.

### Polish — AI Text Transformation

- **Custom presets** — Each polish has its own system prompt, model (local MLX or API), and optional global shortcut. Record with the shortcut and the result arrives already transformed.
- **AI-built polishes** — Describe what you want in plain language; a 6-phase pipeline (understand → plan → write → evaluate flow → evaluate prompt → repair) builds the preset, asks clarification questions when needed, and shows you an eval scorecard before anything is saved. "Edit with AI" refines existing presets the same way.
- **Polish Routes** — Automatic per-app routing: the right polish fires based on the frontmost app, configured with a running/installed app scanner.
- **Runtime context** — Polishes can read the clipboard (only if it changed during recording), take an interactive screenshot, know the active app, and capture desktop audio to hear the other side of a call.
- **Multi-step modules** — Compose a polish from input modules (voice, desktop audio, clipboard, screenshot, active app) and output modules (copy, paste image + text, floating window). A polish can even bypass the LLM entirely and pipe raw captures straight to its outputs.
- **Floating response window** — Output renders in a hand-rolled Markdown view with native SwiftUI LaTeX math — no WebView.
- **JSON import** — Share presets as JSON.

### Cloud Providers (optional)

- **API transcription** — OpenAI (`gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1`, `gpt-4o-transcribe-diarize`), ElevenLabs (Scribe v1/v2), and OpenRouter.
- **API language models** — OpenAI and OpenRouter LLMs for polishes, the polish generator, and summaries. Add any custom model ID.
- **Keychain key storage** — API keys are stored in the macOS Keychain, never in plain text.
- **Cost hints** — Known OpenAI transcription models show audio cost per minute.
- **Resilient uploads** — Fresh connection per upload, automatic retries, sequential re-send of failed batches, full-file fallback, and a FailedRecordings rescue with an alert as the last resort.

### AI Summaries

- **On-Device LLM Summarization** — Summarize transcriptions and session recordings using Gemma 4 models running locally via MLX on Apple Silicon — or through an API language model.
- **Multiple Presets** — General summary, Meeting Notes, Action Items, and Technical — each with tailored system prompts and configurable output length (short, medium, long, custom), managed in a redesigned preset editor.
- **Isolated Runtime** — Local summary models run in a dedicated Python venv with one-click setup. Models auto-unload after generation to reclaim RAM.
- **Streaming Generation** — Watch the summary generate in real time with token-by-token streaming.

### Session Recording

- **Long-form Recording** — Record sessions of any length (meetings, lectures, interviews). Captures both microphone and system audio simultaneously.
- **Manual Mixer** — OBS-style dBFS level meters (green/yellow/red, −60 dB floor) and −40…+40 dB faders for mic and desktop audio, as channel strips in the menu bar during a session. Fader gain is written into the file through a soft limiter; tracks record at true level (no auto-gain). The floating indicator draws live meter arcs: top = mic, bottom = desktop.
- **Session Automation** — Stop a session and it auto-transcribes (local or API, with preflight checks) and optionally auto-generates a summary. One job at a time, persisted state, human-readable failure messages, dock control in the Sessions tab.
- **Session Import** — Drop MP3/WAV/M4A/AAC/AIFF/MP4/MOV files to create sessions from existing audio or video. Multi-track audio is mixed down automatically.
- **RAM-Aware Batch Transcription** — Long recordings are automatically split into chunks sized to your available RAM and transcribed sequentially with live progress streaming.
- **Session Management** — View, rename, delete, and re-transcribe sessions. Bulk actions for clearing transcriptions. Audio and transcription files stored locally.
- **Per-Session Summaries** — Generate summaries for any session with your choice of preset and model, in a tabbed interface alongside the full transcription.
- **Per-recording diagnostics** — Every session writes an `audio_levels.txt` with its mix analysis.

### Dashboard

- **Hero Stats** — Time saved, total words transcribed, average WPM, and transcription count at a glance.
- **Activity Grid** — GitHub-style 52-week × 7-day streak panel with dock-style magnification and dwell tooltips.
- **Activity Analytics** — Monthly activity bars, weekly pattern breakdown, and hourly activity heatmap.
- **Trends & Distribution** — WPM trends over 8 weeks, transcription length distribution, top words, and most-used models.
- **Animated Charts** — All visualizations animate in with staggered spring grow-ins.

### Smart Model Management

- **Lazy Loading** — Models are not loaded at app startup. When you press the record shortcut, the model loads in the background while you speak. By the time you stop, the transcription is ready.
- **Idle Unloading** — After 2 minutes of inactivity, the model automatically unloads to free RAM.
- **Streaming models excluded** — NeMo streaming models (Nemotron) stay loaded since they need to be instantly available for real-time transcription.
- **API model manager** — Add, edit, and remove cloud transcription and language models in redesigned glass sheets with capsule pill provider switching.

### Recording

- **Floating Indicator** — A glassy, draggable recording indicator with GPU-synced animations. Side-wave arcs respond to your voice volume. Snaps to 8 screen positions. Adapts to light and dark themes.
- **Six indicator styles** — Classic, Glow, Aurora, Prism, Graphite, and Ember.
- **Noise floor calibration** — One click teaches the indicator your room's ambient noise so the level display responds to speech, not background hum — with reset to automatic anytime.
- **Real-time Progress** — For longer transcriptions, the UI shows live progress with real segment tracking for Whisper models.
- **Audio Ducking** — Automatically lowers or mutes system audio while recording, then restores it — correctly per Bluetooth profile (A2DP/HFP).

### Audio Device Management

- **Microphone Selection** — Pick your preferred microphone from the dropdown. Uses native CoreAudio to set the input device.
- **Prefer built-in mic over Bluetooth** — New default-on setting records from the built-in mic when the selected mic is a Bluetooth device, keeping your headphones in high-quality A2DP so music never cuts out.
- **Accurate level monitoring** — The indicator and onboarding meters measure the device the recording actually uses.
- **AirPods Support** — When AirPods connect or disconnect, the device list refreshes automatically. If your selected microphone disappears, the app falls back to the default.
- **Persistent Selection** — Your preferred microphone is remembered across app restarts.

### Transcription History

- All transcriptions are saved locally with timestamp, duration, word count, and which model was used.
- **Keyword Search** — Filter history instantly by keyword across transcription text and model names.
- **Similarity Search** — Index your entire history with one click and search by meaning using Apple's native `NLEmbedding` — on-device, no downloads. Text matches first, then semantic matches, deduplicated.
- View full transcription text, copy to clipboard, and view inline summaries per preset.

### Onboarding

- **Branching first-run flow** — Welcome → live microphone preview → setup choice (Local / API / Both with hover previews) → model downloads or Keychain API setup with dated model recommendations → shortcuts → test transcription → next steps.
- **Live step rail** — Always know where you are; finishing opens Settings focused and ready.

### Auto-Updates

- **GitHub Release Checking** — Automatically checks for new versions every 24 hours via the GitHub Releases API.
- **In-App Update Flow** — Rendered release notes, download progress, and one-click install that replaces the app bundle and relaunches.

### Customization

- Configurable keyboard shortcuts for recording, cancelling, session recording, and every polish preset
- Sound feedback when recording starts and stops
- Two audio modes during recording: mute completely, or lower volume by a percentage
- Rolling batch transcription controls
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
   - Screen recording (optional — enables desktop audio capture for sessions and polish context)

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

On first launch, the onboarding walks you through everything: microphone check, choosing Local / API / Both, downloading models or entering API keys, shortcuts, and a test transcription.

### Quick transcription

1. Press `Control + P` to start recording
2. Speak into your microphone
3. Press the hotkey again to stop — text is in your clipboard (and pasted at your cursor if auto-paste is on)

Long dictations transcribe in rolling chunks while you speak, so the result is nearly instant when you stop. The model loads automatically the first time you record and unloads after 2 minutes of idle.

### Polish

Open **Settings → Polish** and create a preset — or describe one in plain language and let the AI generator build it. Give it a shortcut, and recording with that shortcut returns the transformed text instead of the raw transcription. Set up **Routes** to pick polishes automatically per app.

### Sessions

Use `Control + K` to start a long-form session (meetings, lectures, interviews). The floating indicator shows elapsed time and live level meters; the menu bar shows the mixer with faders for mic and desktop audio. Press again to stop — with automation on, transcription (and optionally a summary) starts by itself. You can also **import** existing audio/video files as sessions.

### Summaries

Open any session or transcription, choose a preset (General, Meeting Notes, Action Items, Technical, or your own), pick a local or API model, and generate.

---

## Models

### Local Transcription Models

| Model                    | Download Size | Speed    | Accuracy  | RAM Required | GPU Accelerated |
| ------------------------ | ------------- | -------- | --------- | ------------ | --------------- |
| Whisper Base (EN)        | ~148MB        | Fastest  | Good      | ~400MB       | Yes (Metal)     |
| Whisper Small (EN)       | ~488MB        | Fast     | Very Good | ~850MB       | Yes (Metal)     |
| Whisper Large v3 Turbo   | ~1.6GB        | Fast     | Excellent | ~1.7GB       | Yes (Metal)     |
| Whisper Large v3         | ~3.1GB        | Medium   | Best      | ~4GB         | Yes (Metal)     |
| Parakeet TDT 0.6B       | ~1.2GB        | Fast     | Excellent | ~3GB         | No (CPU)        |
| Nemotron Streaming       | —             | Realtime | Good      | ~2GB         | No (CPU)        |

Whisper models use [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with Metal GPU acceleration for 8-10x faster inference on Apple Silicon. Models are downloaded on-demand and stored locally.

### API Transcription Models (optional)

| Provider   | Models                                                                              |
| ---------- | ----------------------------------------------------------------------------------- |
| OpenAI     | `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1`, `gpt-4o-transcribe-diarize` |
| ElevenLabs | Scribe v2, Scribe v1                                                                 |
| OpenRouter | Any transcription-capable model                                                      |

### Language Models

| Model              | Where it runs | Download Size | RAM Required |
| ------------------ | ------------- | ------------- | ------------ |
| Gemma 4 E2B (4-bit)| Local (MLX)   | ~3.6GB        | ~4GB         |
| Gemma 4 E4B (4-bit)| Local (MLX)   | ~5.2GB        | ~6GB         |
| GPT-4o Mini + any custom model | OpenAI / OpenRouter API | — | — |

Local language models run via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon, installed through a one-click setup with an isolated Python environment. They power summaries, polishes, and the polish generator.

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
│       ├── APIClient.swift             # HTTP client for local backend
│       ├── BackendManager.swift        # Backend process management
│       ├── AudioRecorder.swift         # Audio recording + CoreAudio
│       ├── AudioManager.swift          # Ducking + per-profile volume restore
│       ├── MenuBarView.swift           # Menu bar panel UI + session mixer host
│       ├── RecordingIndicator.swift    # GPU-synced floating indicator
│       ├── RollingBasicBatchSession.swift  # Rolling local transcription
│       ├── RollingAPIBatchSession.swift    # Rolling API transcription + retries
│       ├── TranscriptionProvider.swift     # Provider abstraction (local + API)
│       ├── TranscriptionUploadSession.swift# Fresh-connection API uploads
│       ├── OpenAI*/OpenRouter*/ElevenLabs* # API transcription + LLM clients
│       ├── APIKeychainStore.swift      # Keychain-backed API keys
│       ├── API*ModelStore.swift        # API model management
│       ├── PolishPresetStore.swift     # Polish presets
│       ├── PolishGeneratorEngine.swift # 6-phase AI polish builder
│       ├── PolishContextReader.swift   # Clipboard/screenshot/app context
│       ├── PolishMarkdownView.swift    # Markdown + LaTeX renderer
│       ├── QuickSystemAudioRecorder.swift  # Desktop audio for quick polishes
│       ├── *OnboardingView.swift       # Branching onboarding flow
│       ├── UpdateChecker.swift         # GitHub release auto-updates
│       ├── SummarySetupManager.swift   # MLX summary runtime installer
│       ├── Sessions/                   # Session recording system
│       │   ├── SessionManager.swift        # Sessions + automation queue
│       │   ├── SessionRecorder.swift       # Recording + gain pipeline
│       │   ├── SystemAudioCapture.swift
│       │   ├── SessionImporter.swift       # Audio/video file import
│       │   ├── SessionMixerView.swift      # Menu bar channel strips
│       │   ├── AudioLevelMeter.swift       # dBFS meters (OBS ballistics)
│       │   └── GainStage.swift             # Fader gain + soft limiter
│       ├── Settings/                   # Settings UI (modular tabs)
│       │   ├── SettingsView.swift          # Main settings window
│       │   ├── SettingsSidebar.swift       # Collapsible sidebar
│       │   ├── DashboardView.swift         # Analytics dashboard + activity grid
│       │   ├── PolishSettingsView.swift    # Polish list/editor/generator/routes
│       │   ├── ModelsSettingsView.swift    # Local + API model management
│       │   ├── SessionsSettingsView.swift
│       │   ├── SessionDetailView.swift
│       │   ├── SessionImportView.swift
│       │   ├── HistorySettingsView.swift
│       │   ├── GeneralSettingsView.swift
│       │   ├── ShortcutsSettingsView.swift
│       │   ├── AboutSettingsView.swift
│       │   └── UpdateWindowView.swift      # Auto-update UI
│       ├── HotkeyManager.swift         # Global keyboard shortcuts
│       └── HistoryManager.swift        # Transcription history + search
│
├── backend/                      # Python/FastAPI backend
│   ├── server.py                 # API endpoints + SSE streaming
│   ├── model_manager.py          # Model download and loading
│   ├── batch_transcriber.py      # RAM-aware chunked transcription
│   ├── summary_manager.py        # MLX summary + polish model management
│   ├── summary_worker.py         # Generation worker
│   ├── setup_summary_venv.py     # Summary runtime installer
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
- **Backend**: Python / FastAPI (local HTTP server on port 8770)
- **Transcription**: OpenAI Whisper via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (Metal GPU), NVIDIA NeMo (Parakeet) — plus optional OpenAI / ElevenLabs / OpenRouter APIs
- **Language models**: Google Gemma 4 via [MLX](https://github.com/ml-explore/mlx) (Apple Silicon) — plus optional OpenAI / OpenRouter APIs
- **Audio**: AVFoundation + CoreAudio (Swift), sounddevice (Python)
- **Communication**: HTTP + Server-Sent Events (SSE) for streaming progress
- **Secrets**: macOS Keychain for API keys
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
lsof -i :8770              # Check if port is in use
lsof -ti:8770 | xargs kill # Kill existing process, then restart the app
```

**Local port note**
AiTranscribe uses `127.0.0.1:8770` for its private backend. It intentionally avoids `8765` because Claude Science keeps a persistent local server on `127.0.0.1:8765` while it is open, with `8766` used by its sandbox flow. Keeping AiTranscribe on `8770` lets both apps coexist and prevents AiTranscribe's backend cleanup from touching Claude Science.

**Summary setup fails**
Make sure you have native arm64 Python 3.10+ installed (not Rosetta). Run `python3 -c "import platform; print(platform.machine())"` — it should print `arm64`.

**Music cuts out when recording starts**
Your selected microphone is probably a Bluetooth device — opening its mic switches it to the HFP call profile. Leave **"Prefer built-in mic over Bluetooth"** enabled in General settings (default), and the app records from the built-in mic while your Bluetooth audio stays in high quality.

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

This project is licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)** — see the [LICENSE](LICENSE) file for details.

In plain terms:

- **You can** download, use, modify, and share AiTranscribe freely — personally or at work — and contribute back to the project.
- **You can't** take this code and ship it (or a derivative) as a closed-source product. Anything you distribute that's built on AiTranscribe must stay open under the GPL, source included — which means nobody can turn it into their own paid proprietary app.
- **Commercial licensing** — separate commercial terms may be offered for material the author controls; reach out if that's you.

The official DMG is pay-what-you-want — if AiTranscribe saves you time, [supporting it](https://ko-fi.com/veljkospasic) keeps it alive.

---

## Support

If you find AITranscribe useful, consider supporting the project — it helps keep development going.

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/veljkospasic)

---

## Acknowledgments

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for Metal-accelerated Whisper inference
- [OpenAI Whisper](https://github.com/openai/whisper) for Whisper models
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for Parakeet models
- [MLX](https://github.com/ml-explore/mlx) for on-device LLM inference on Apple Silicon
- [Google Gemma](https://ai.google.dev/gemma) for open-weight summary models

---
