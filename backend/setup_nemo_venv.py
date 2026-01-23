#!/usr/bin/env python3
"""
NeMo Virtual Environment Setup Script
======================================

This script creates a Python virtual environment and installs NeMo dependencies.
It outputs JSON progress events that can be parsed by the Swift frontend.

Usage:
    python3 setup_nemo_venv.py /path/to/venv

Output format (JSON lines):
    {"step": "checking_python", "progress": 0.0, "message": "Checking Python..."}
    {"step": "creating_venv", "progress": 0.1, "message": "Creating virtual environment..."}
    {"step": "installing_packages", "progress": 0.2, "message": "Installing packages...", "package": "torch"}
    {"step": "verifying", "progress": 0.9, "message": "Verifying installation..."}
    {"step": "complete", "progress": 1.0, "message": "NeMo installed successfully"}
    {"step": "error", "progress": -1, "message": "Error message here"}
"""

import sys
import os
import subprocess
import json
import venv
import shutil
from pathlib import Path


def emit_progress(step: str, progress: float, message: str, **kwargs):
    """Emit a JSON progress event to stdout."""
    event = {
        "step": step,
        "progress": progress,
        "message": message,
        **kwargs
    }
    print(json.dumps(event), flush=True)


def emit_error(message: str, details: str = None):
    """Emit a JSON error event to stdout."""
    event = {
        "step": "error",
        "progress": -1,
        "message": message
    }
    if details:
        event["details"] = details
    print(json.dumps(event), flush=True)


def check_python_version():
    """Check that Python version is adequate."""
    if sys.version_info < (3, 9):
        emit_error(
            f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}",
            "Please install Python 3.9 or higher from python.org"
        )
        return False
    return True


def create_venv(venv_path: Path):
    """Create a virtual environment at the specified path."""
    emit_progress("creating_venv", 0.1, f"Creating virtual environment at {venv_path}...")

    try:
        # Remove existing venv if present
        if venv_path.exists():
            emit_progress("creating_venv", 0.05, "Removing existing virtual environment...")
            shutil.rmtree(venv_path)

        # Create new venv
        venv.create(venv_path, with_pip=True)

        # Verify pip exists
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            emit_error("Failed to create virtual environment with pip")
            return False

        return True

    except Exception as e:
        emit_error(f"Failed to create virtual environment: {str(e)}")
        return False


def upgrade_pip(venv_path: Path):
    """Upgrade pip in the virtual environment."""
    emit_progress("upgrading_pip", 0.15, "Upgrading pip...")

    pip_path = venv_path / "bin" / "pip"

    try:
        result = subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            emit_error("Failed to upgrade pip", result.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        emit_error("Timeout while upgrading pip")
        return False
    except Exception as e:
        emit_error(f"Failed to upgrade pip: {str(e)}")
        return False


def install_packages(venv_path: Path, requirements_path: Path):
    """Install packages from requirements-nemo.txt with progress."""
    emit_progress("installing_packages", 0.2, "Installing NeMo dependencies...")

    pip_path = venv_path / "bin" / "pip"

    # Read packages from requirements file
    packages = []
    try:
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    packages.append(line)
    except Exception as e:
        emit_error(f"Failed to read requirements file: {str(e)}")
        return False

    if not packages:
        emit_error("No packages found in requirements-nemo.txt")
        return False

    # Install packages one by one for better progress feedback
    total_packages = len(packages)
    for i, package in enumerate(packages):
        progress = 0.2 + (0.6 * (i / total_packages))
        emit_progress(
            "installing_packages",
            progress,
            f"Installing {package}... ({i+1}/{total_packages})",
            package=package
        )

        try:
            result = subprocess.run(
                [str(pip_path), "install", package],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes for large packages like torch
            )

            if result.returncode != 0:
                # Some errors are warnings, check if it's fatal
                if "error" in result.stderr.lower() and "warning" not in result.stderr.lower():
                    emit_error(f"Failed to install {package}", result.stderr)
                    return False

        except subprocess.TimeoutExpired:
            emit_error(f"Timeout while installing {package}")
            return False
        except Exception as e:
            emit_error(f"Failed to install {package}: {str(e)}")
            return False

    # Note: We no longer downgrade lhotse as it causes import errors.
    # The lhotse sampler bug (in 1.24.0+) only affects transcription,
    # which is handled by fallback methods in server.py

    return True


def verify_installation(venv_path: Path):
    """Verify that NeMo was installed correctly by importing it."""
    emit_progress("verifying", 0.9, "Verifying NeMo installation (this may take a few minutes)...")

    python_path = venv_path / "bin" / "python3"

    # More robust verification that handles warnings and non-fatal errors
    verify_script = """
import sys
import warnings
warnings.filterwarnings('ignore')

info = {}
errors = []

# Check torch
try:
    import torch
    info['torch_version'] = torch.__version__
    info['cuda_available'] = str(torch.cuda.is_available())
    info['mps_available'] = str(torch.backends.mps.is_available())
except ImportError as e:
    errors.append(f"torch import failed: {e}")

# Check nemo core
try:
    import nemo
    info['nemo_version'] = nemo.__version__
except ImportError as e:
    errors.append(f"nemo import failed: {e}")

# Check nemo.collections.asr - this may have warnings but still work
try:
    import nemo.collections.asr as nemo_asr
    info['nemo_asr'] = 'available'
except ImportError as e:
    errors.append(f"nemo.collections.asr import failed: {e}")
except Exception as e:
    # Non-import errors (like lhotse issues) are warnings, not failures
    info['nemo_asr'] = 'available_with_warnings'
    info['asr_warning'] = str(e)

# Print results
for key, value in info.items():
    print(f"{key}={value}")

# Only fail if critical imports failed
if 'nemo_version' not in info or 'torch_version' not in info:
    print(f"errors={'; '.join(errors)}")
    sys.exit(1)

sys.exit(0)
"""

    try:
        # Use a very long timeout (10 minutes) or no timeout for verification
        # NeMo imports can be slow on some machines as it loads many dependencies
        result = subprocess.run(
            [str(python_path), "-c", verify_script],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes - NeMo import can be very slow on some systems
        )

        if result.returncode != 0:
            error_msg = result.stdout + result.stderr
            emit_error("NeMo installation verification failed", error_msg)
            return None

        # Parse verification output
        info = {}
        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key] = value

        return info

    except subprocess.TimeoutExpired:
        emit_error("Verification timed out after 10 minutes. NeMo may still be loading. Please try again.")
        return None
    except Exception as e:
        emit_error(f"Failed to verify installation: {str(e)}")
        return None


def main():
    if len(sys.argv) < 2:
        emit_error("Usage: python3 setup_nemo_venv.py /path/to/venv")
        sys.exit(1)

    venv_path = Path(sys.argv[1]).expanduser().resolve()

    # Find requirements-nemo.txt relative to this script
    script_dir = Path(__file__).parent
    requirements_path = script_dir / "requirements-nemo.txt"

    if not requirements_path.exists():
        emit_error(f"requirements-nemo.txt not found at {requirements_path}")
        sys.exit(1)

    # Step 1: Check Python version
    emit_progress("checking_python", 0.0, f"Checking Python version... ({sys.version_info.major}.{sys.version_info.minor})")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Create virtual environment
    if not create_venv(venv_path):
        sys.exit(1)

    # Step 3: Upgrade pip
    if not upgrade_pip(venv_path):
        sys.exit(1)

    # Step 4: Install packages
    if not install_packages(venv_path, requirements_path):
        sys.exit(1)

    # Step 5: Verify installation
    info = verify_installation(venv_path)
    if info is None:
        sys.exit(1)

    # Success!
    emit_progress(
        "complete",
        1.0,
        "NeMo installed successfully",
        nemo_version=info.get("nemo_version", "unknown"),
        torch_version=info.get("torch_version", "unknown"),
        cuda_available=info.get("cuda_available", "false"),
        mps_available=info.get("mps_available", "false"),
        venv_path=str(venv_path)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
