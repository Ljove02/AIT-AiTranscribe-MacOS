#!/bin/bash
#
# build_standalone.sh
# ===================
#
# Builds the AiTranscribe backend into a standalone executable using PyInstaller.
# The resulting executable can be bundled with the macOS app.
#
# Usage:
#   ./build_standalone.sh
#
# Requirements:
#   - Python 3.8+ with pip
#   - All backend dependencies installed
#
# Output:
#   - dist/aitranscribe-server (standalone executable)
#

set -e  # Exit on error

# Change to the backend directory
cd "$(dirname "$0")"

echo "========================================"
echo "AiTranscribe Backend Build Script"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python: $PYTHON_VERSION"

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
else
    echo "PyInstaller: $(python3 -c 'import PyInstaller; print(PyInstaller.__version__)')"
fi

echo ""
echo "Building standalone executable..."
echo ""

# Clean previous builds
rm -rf build dist *.spec 2>/dev/null || true

# Build with PyInstaller
# Note: This creates a single-file executable which is slower to start
# but easier to distribute. For faster startup, remove --onefile.
python3 -m PyInstaller \
    --name aitranscribe-server \
    --onefile \
    --console \
    --noconfirm \
    --clean \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import uvicorn.lifespan.off \
    --hidden-import fastapi \
    --hidden-import pydantic \
    --hidden-import sounddevice \
    --hidden-import scipy \
    --hidden-import scipy.io \
    --hidden-import scipy.io.wavfile \
    --hidden-import numpy \
    --hidden-import torch \
    --hidden-import torchaudio \
    --hidden-import faster_whisper \
    --hidden-import ctranslate2 \
    --hidden-import huggingface_hub \
    --collect-all torch \
    --collect-all torchaudio \
    --collect-all faster_whisper \
    --collect-all ctranslate2 \
    --collect-all sounddevice \
    --copy-metadata torch \
    --copy-metadata torchaudio \
    --copy-metadata faster-whisper \
    server.py

echo ""
echo "Build complete!"
echo ""

# Check if build succeeded
if [ -f "dist/aitranscribe-server" ]; then
    SIZE=$(du -h "dist/aitranscribe-server" | cut -f1)
    echo "Output: dist/aitranscribe-server ($SIZE)"
    echo ""

    # Create Resources directory in Xcode project if needed
    RESOURCES_DIR="../AiTranscribe/AiTranscribe/Resources"
    if [ ! -d "$RESOURCES_DIR" ]; then
        echo "Creating Resources directory..."
        mkdir -p "$RESOURCES_DIR"
    fi

    # Copy to Xcode project
    echo "Copying to Xcode project..."
    cp "dist/aitranscribe-server" "$RESOURCES_DIR/"

    echo ""
    echo "========================================"
    echo "SUCCESS!"
    echo "========================================"
    echo ""
    echo "The standalone backend has been built and copied to:"
    echo "  $RESOURCES_DIR/aitranscribe-server"
    echo ""
    echo "Next steps:"
    echo "1. Open the Xcode project"
    echo "2. Add 'Resources/aitranscribe-server' to the target"
    echo "3. Ensure it's copied to the app bundle (Build Phases > Copy Bundle Resources)"
    echo "4. Build and archive the app for distribution"
    echo ""
else
    echo "ERROR: Build failed - executable not found"
    exit 1
fi
