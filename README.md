![Banner](git-banner.png)

# AiTranscribe

A powerful macOS menu bar application for local speech-to-text transcription. Runs completely offline using state-of-the-art AI models.

![macOS](https://img.shields.io/badge/macOS-13.0+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Swift](https://img.shields.io/badge/Swift-5.9+-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

## Features

- **100% Local & Private** - All processing happens on your Mac, no data sent to the cloud
- **Multiple AI Models** - Support for NVIDIA Parakeet, OpenAI Whisper, and more
- **Global Hotkey** - Press `Option + Space` to start recording from anywhere
- **Menu Bar App** - Stays out of your way, always accessible
- **Real-time Transcription** - See your words as you speak
- **Clipboard Integration** - Transcribed text automatically copied to clipboard

## Requirements

- macOS 13.0 (Ventura) or later
- ~4GB RAM (more for larger models)
- ~3GB disk space for models

---

## Installation

### Option 1: Download Pre-built App (Easiest)

1. **Download** the latest DMG from [Releases](https://github.com/Ljove02/AIT-AiTranscribe-MacOS/releases)

2. **Open the DMG** and drag AiTranscribe to Applications

3. **First launch** - Since the app is not signed with an Apple Developer certificate:

   **Method A: Right-click to open**

   - Right-click (or Control-click) on AiTranscribe in Applications
   - Select "Open" from the menu
   - Click "Open" in the dialog that appears
   - You only need to do this once

   **Method B: Terminal command**

   ```bash
   xattr -cr /Applications/AiTranscribe.app
   ```

   Then double-click to open normally.

4. **Grant permissions** when prompted:
   - Microphone access (required for recording)
   - Accessibility access (optional, for typing text directly into apps)

---

### Option 2: Build from Source

Building locally avoids all Gatekeeper warnings since the app is built on your machine.

#### Prerequisites

| Requirement            | How to Install                                                                |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Xcode 15+**          | [App Store](https://apps.apple.com/app/xcode/id497799835)                     |
| **Python 3.10+**       | `brew install python@3.11` or [python.org](https://www.python.org/downloads/) |
| **Command Line Tools** | `xcode-select --install`                                                      |

#### Quick Build (One Command)

```bash
# Clone, setup, and build everything
git clone https://github.com/Ljove02/AiTranscribe.git
cd AiTranscribe
python3 -m venv venv && source venv/bin/activate && pip install -r backend/requirements.txt
./build_production.sh
```

The built app will be in `dist/AiTranscribe.app`.

#### Step-by-Step Build

**Step 1: Clone the repository**

```bash
git clone https://github.com/Ljove02/AiTranscribe.git
cd AiTranscribe
```

**Step 2: Set up Python environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies (this may take a few minutes)
pip install -r backend/requirements.txt
```

**Step 3: Build the backend executable**

```bash
cd backend
./build_standalone.sh
cd ..
```

This creates `backend/dist/aitranscribe-server`.

**Step 4: Build the Swift app**

```bash
cd AiTranscribe
xcodebuild -scheme AiTranscribe -configuration Release -derivedDataPath build
cd ..
```

**Step 5: Create final app bundle**

```bash
# The build_production.sh script does steps 3-4 automatically and creates a DMG
./build_production.sh
```

**Step 6: Install**

```bash
# Copy to Applications
cp -R dist/AiTranscribe.app /Applications/

# Or just open directly
open dist/AiTranscribe.app
```

---

## Development Mode

For contributors and developers who want to run the app with hot-reload:

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

The `--dev` flag enables auto-reload when you modify Python files.

---

## Project Structure

```
AiTranscribe/
├── AiTranscribe/              # Swift/SwiftUI frontend
│   ├── AiTranscribe.xcodeproj
│   └── AiTranscribe/
│       ├── AiTranscribeApp.swift    # App entry point
│       ├── AppState.swift           # Central state management
│       ├── APIClient.swift          # HTTP client for backend
│       ├── BackendManager.swift     # Backend process management
│       ├── AudioRecorder.swift      # Audio recording (AVFoundation)
│       ├── MenuBarView.swift        # Menu bar UI
│       ├── SettingsView.swift       # Settings interface
│       └── ...
│
├── backend/                   # Python/FastAPI backend
│   ├── server.py              # API endpoints
│   ├── model_manager.py       # Model download/loading
│   ├── recorder.py            # Server-side recording
│   ├── requirements.txt       # Python dependencies
│   └── build_standalone.sh    # Build executable
│
├── build_production.sh        # Full production build script
├── run_backend.sh             # Development server script
└── README.md
```

---

## Usage

1. **Click the menu bar icon** (microphone icon in the top-right of your screen)

2. **Download a model** - Go to Settings → Models and download your preferred model:

   - **Parakeet TDT 0.6B** (Recommended) - Best balance of speed and accuracy
   - **Whisper Small** - Good for older Macs
   - **Whisper Large** - Highest accuracy, requires more RAM

3. **Start transcribing**
   - Press `Option + Space` or click "Start Recording"
   - Speak into your microphone
   - Press the hotkey again to stop
   - Text is automatically copied to your clipboard

---

## Models

| Model             | Download Size | Speed   | Accuracy  | RAM Required |
| ----------------- | ------------- | ------- | --------- | ------------ |
| Parakeet TDT 0.6B | ~1.2GB        | Fast    | Excellent | ~3GB         |
| Whisper Tiny      | ~150MB        | Fastest | Good      | ~1GB         |
| Whisper Small     | ~500MB        | Fast    | Very Good | ~2GB         |
| Whisper Medium    | ~1.5GB        | Medium  | Excellent | ~4GB         |
| Whisper Large     | ~3GB          | Slow    | Best      | ~6GB         |

Models are downloaded on-demand and stored in `~/.cache/huggingface/hub/`.

---

## Troubleshooting

### "App is damaged and can't be opened"

This happens because the app isn't signed with an Apple Developer certificate:

```bash
xattr -cr /Applications/AiTranscribe.app
```

### "Microphone access denied"

Go to **System Settings → Privacy & Security → Microphone** and enable AiTranscribe.

### Backend won't start

```bash
# Check if port 8765 is in use
lsof -i :8765

# Kill existing process
lsof -ti:8765 | xargs kill -9

# Restart the app
```

### Build fails with "pip not found"

Make sure you activated the virtual environment:

```bash
source venv/bin/activate
```

### Xcode build fails

```bash
# Make sure Command Line Tools are installed
xcode-select --install

# If still failing, try cleaning
cd AiTranscribe && xcodebuild clean
```

### Models not downloading

Ensure you have internet access and enough disk space (~3GB for larger models).

---

## Tech Stack

- **Frontend**: Swift/SwiftUI (macOS native)
- **Backend**: Python/FastAPI (local HTTP server on port 8765)
- **AI Models**: NVIDIA NeMo (Parakeet), OpenAI Whisper via faster-whisper
- **Audio**: AVFoundation (Swift), sounddevice (Python)

---

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AiTranscribe.git
   ```
3. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Set up** development environment (see [Development Mode](#development-mode))
5. **Make** your changes
6. **Test** thoroughly
7. **Commit** your changes:
   ```bash
   git commit -m 'Add some amazing feature'
   ```
8. **Push** to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
9. **Open** a Pull Request

### Areas for Contribution

- Adding new transcription models
- UI/UX improvements
- Performance optimizations
- Documentation improvements
- Bug fixes

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for Parakeet models
- [OpenAI Whisper](https://github.com/openai/whisper) for Whisper models
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized inference

---

## Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search [existing issues](https://github.com/Ljove02/AiTranscribe/issues)
3. Open a [new issue](https://github.com/Ljove02/AiTranscribe/issues/new)

---

**Made with love for the open source community**
