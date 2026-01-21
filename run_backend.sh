#!/bin/bash
# =============================================================================
# AiTranscribe Backend Runner
# =============================================================================
#
# This script starts the backend server.
#
# Usage:
#   ./run_backend.sh           # Normal start
#   ./run_backend.sh --dev     # Development mode (auto-reload)
#
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=============================================="
echo "AiTranscribe Backend"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR"
    echo "Please create it first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r backend/requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "uvicorn not found. Installing dependencies..."
    pip install -r "$BACKEND_DIR/requirements.txt"
fi

# Change to backend directory
cd "$BACKEND_DIR"

# Check for development mode
if [ "$1" == "--dev" ]; then
    echo "Starting in DEVELOPMENT mode (auto-reload enabled)..."
    echo "API docs: http://127.0.0.1:8765/docs"
    echo ""
    uvicorn server:app --host 127.0.0.1 --port 8765 --reload
else
    echo "Starting server..."
    echo "API docs: http://127.0.0.1:8765/docs"
    echo ""
    python server.py
fi
