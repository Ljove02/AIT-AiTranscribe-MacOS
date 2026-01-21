#!/bin/bash
# =============================================================================
# AiTranscribe Production Build Script
# =============================================================================
#
# This script builds a production-ready version of AiTranscribe:
# 1. Builds the Python backend as a standalone executable
# 2. Builds the Swift app in Release configuration
# 3. Creates a DMG installer
#
# Usage:
#   ./build_production.sh           # Full build
#   ./build_production.sh --skip-backend  # Skip backend build (if already built)
#   ./build_production.sh --help    # Show help
#
# =============================================================================

set -e  # Exit on error

# Configuration
# Get the directory where this script is located (works even when called from different directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
VERSION="0.1.0"
APP_NAME="AiTranscribe"
SKIP_BACKEND=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-backend)
            SKIP_BACKEND=true
            shift
            ;;
        --help)
            echo "AiTranscribe Production Build Script"
            echo ""
            echo "Usage:"
            echo "  ./build_production.sh              Full build"
            echo "  ./build_production.sh --skip-backend   Skip backend build"
            echo "  ./build_production.sh --help       Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start
print_header "AiTranscribe Production Build v${VERSION}"

cd "$PROJECT_ROOT"

# =============================================================================
# Step 1: Build Backend
# =============================================================================

if [ "$SKIP_BACKEND" = true ]; then
    print_warning "Skipping backend build (--skip-backend flag)"
else
    print_header "Step 1: Building Backend Executable"

    # Check if venv exists
    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Please run:"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r backend/requirements.txt"
        exit 1
    fi

    # Activate venv and build
    source venv/bin/activate

    cd backend

    # Check if build script exists
    if [ ! -f "build_standalone.sh" ]; then
        print_error "build_standalone.sh not found in backend/"
        exit 1
    fi

    # Run build
    echo "Building backend (this may take 5-15 minutes)..."
    ./build_standalone.sh

    cd "$PROJECT_ROOT"

    # Verify backend was built
    BACKEND_PATH="AiTranscribe/AiTranscribe/Resources/aitranscribe-server"
    if [ ! -f "$BACKEND_PATH" ]; then
        print_error "Backend executable not found at $BACKEND_PATH"
        exit 1
    fi

    # Ensure it's executable
    chmod +x "$BACKEND_PATH"

    BACKEND_SIZE=$(du -h "$BACKEND_PATH" | cut -f1)
    print_success "Backend built successfully (${BACKEND_SIZE})"

    # Copy NeMo setup script and requirements to Resources
    NEMO_SCRIPT="backend/setup_nemo_venv.py"
    NEMO_REQS="backend/requirements-nemo.txt"
    RESOURCES_DIR="AiTranscribe/AiTranscribe/Resources"

    if [ -f "$NEMO_SCRIPT" ]; then
        cp "$NEMO_SCRIPT" "$RESOURCES_DIR/"
        print_success "NeMo setup script copied to Resources"
    else
        print_warning "NeMo setup script not found at $NEMO_SCRIPT"
    fi

    if [ -f "$NEMO_REQS" ]; then
        cp "$NEMO_REQS" "$RESOURCES_DIR/"
        print_success "NeMo requirements copied to Resources"
    else
        print_warning "NeMo requirements not found at $NEMO_REQS"
    fi
fi

# =============================================================================
# Step 2: Build Xcode Project
# =============================================================================

print_header "Step 2: Building Xcode Project (Release)"

cd AiTranscribe

# Clean and build
echo "Cleaning previous build..."
xcodebuild -scheme AiTranscribe -configuration Release clean 2>/dev/null || true

echo "Building Release configuration..."
xcodebuild -scheme AiTranscribe -configuration Release build

# Find the built app
APP_PATH=$(find ~/Library/Developer/Xcode/DerivedData -name "AiTranscribe.app" -path "*/Release/*" -type d 2>/dev/null | head -1)

if [ -z "$APP_PATH" ]; then
    print_error "Built app not found in DerivedData!"
    echo "Try building manually in Xcode: Product → Build"
    exit 1
fi

print_success "App built at: $APP_PATH"

cd "$PROJECT_ROOT"

# =============================================================================
# Step 3: Copy to Distribution Folder
# =============================================================================

print_header "Step 3: Preparing Distribution"

# Create dist folder
mkdir -p dist

# Remove old app if exists
rm -rf "dist/${APP_NAME}.app"

# Copy new app
echo "Copying app to dist folder..."
cp -R "$APP_PATH" "dist/"

# Verify backend is in bundle
BUNDLED_BACKEND="dist/${APP_NAME}.app/Contents/Resources/aitranscribe-server"
if [ -f "$BUNDLED_BACKEND" ]; then
    print_success "Backend executable is bundled in app"
else
    print_warning "Backend executable NOT found in app bundle!"
    echo "You may need to add it to Xcode's 'Copy Bundle Resources' phase"
fi

# Verify NeMo setup script is in bundle
BUNDLED_NEMO="dist/${APP_NAME}.app/Contents/Resources/setup_nemo_venv.py"
if [ -f "$BUNDLED_NEMO" ]; then
    print_success "NeMo setup script is bundled in app"
else
    print_warning "NeMo setup script NOT found in app bundle!"
    echo "NeMo model installation may not work without this script."
fi

# Verify NeMo requirements is in bundle
BUNDLED_NEMO_REQS="dist/${APP_NAME}.app/Contents/Resources/requirements-nemo.txt"
if [ -f "$BUNDLED_NEMO_REQS" ]; then
    print_success "NeMo requirements is bundled in app"
else
    print_warning "NeMo requirements NOT found in app bundle!"
    echo "NeMo model installation may not work without this file."
fi

print_success "App copied to: dist/${APP_NAME}.app"

# =============================================================================
# Step 4: Create DMG
# =============================================================================

print_header "Step 4: Creating DMG Installer"

# Remove old DMG
rm -f "dist/${APP_NAME}-v${VERSION}.dmg"

# Create DMG
echo "Creating DMG..."
hdiutil create -volname "${APP_NAME}" \
    -srcfolder "dist/${APP_NAME}.app" \
    -ov -format UDZO \
    "dist/${APP_NAME}-v${VERSION}.dmg"

DMG_SIZE=$(du -h "dist/${APP_NAME}-v${VERSION}.dmg" | cut -f1)
print_success "DMG created: dist/${APP_NAME}-v${VERSION}.dmg (${DMG_SIZE})"

# =============================================================================
# Summary
# =============================================================================

print_header "Build Complete!"

echo "Output files:"
echo "  App: $PROJECT_ROOT/dist/${APP_NAME}.app"
echo "  DMG: $PROJECT_ROOT/dist/${APP_NAME}-v${VERSION}.dmg"
echo ""

# Check if signed
CODESIGN_STATUS=$(codesign -dv "dist/${APP_NAME}.app" 2>&1 | grep "Signature=" || echo "not signed")
if [[ "$CODESIGN_STATUS" == *"not signed"* ]]; then
    print_warning "App is NOT code signed"
    echo ""
    echo "For distribution, you should:"
    echo "  1. Sign with Developer ID"
    echo "  2. Notarize with Apple"
    echo ""
    echo "For local testing, remove quarantine:"
    echo "  xattr -cr dist/${APP_NAME}.app"
else
    print_success "App is code signed"
fi

echo ""
echo "To test the build:"
echo "  open dist/${APP_NAME}.app"
echo ""
