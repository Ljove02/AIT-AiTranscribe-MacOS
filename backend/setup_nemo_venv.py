#!/usr/bin/env python3
"""
setup_nemo_venv.py
==================

Creates a Python virtual environment with NeMo dependencies for AiTranscribe.

This script is bundled with the macOS app and run when users install NeMo support.
It outputs JSON progress updates to stdout for the Swift UI to display.

Usage:
    python3 setup_nemo_venv.py <target_path>

Output:
    JSON progress events to stdout, one per line.
"""

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def emit_progress(step: str, progress: float, message: str, package: str = None,
                  nemo_version: str = None, torch_version: str = None):
    """Print a JSON progress event to stdout."""
    event = {
        "step": step,
        "progress": progress,
        "message": message,
    }
    if package:
        event["package"] = package
    if nemo_version:
        event["nemo_version"] = nemo_version
    if torch_version:
        event["torch_version"] = torch_version
    print(json.dumps(event), flush=True)


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def main():
    if len(sys.argv) != 2:
        emit_progress("error", 0, "Usage: python3 setup_nemo_venv.py <target_path>")
        sys.exit(1)

    target_path = Path(sys.argv[1]).expanduser().resolve()

    emit_progress("checking_python", 0.0, "Checking Python installation...")

    python_exe = sys.executable
    if not python_exe or not Path(python_exe).exists():
        emit_progress("error", 0, "Python executable not found")
        sys.exit(1)

    emit_progress("checking_python", 0.05, f"Using Python: {python_exe}")

    emit_progress("creating_venv", 0.1, f"Creating virtual environment at {target_path}...")

    if target_path.exists():
        shutil.rmtree(target_path)

    try:
        venv.create(target_path, with_pip=True)
    except Exception as e:
        emit_progress("error", 0, f"Failed to create virtual environment: {e}")
        sys.exit(1)

    venv_python = target_path / "bin" / "python3"

    emit_progress("upgrading_pip", 0.15, "Upgrading pip...")

    try:
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            emit_progress("error", 0, f"Failed to upgrade pip: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        emit_progress("error", 0, "Pip upgrade timed out")
        sys.exit(1)

    emit_progress("upgrading_pip", 0.2, "Pip upgraded successfully")

    requirements_path = get_script_dir() / "requirements-nemo.txt"

    if not requirements_path.exists():
        emit_progress("error", 0, f"requirements-nemo.txt not found at {requirements_path}")
        sys.exit(1)

    emit_progress("installing_packages", 0.25, "Installing NeMo dependencies...")

    total_packages = 10
    current_package = 0

    packages = [
        ("setuptools", "setuptools"),
        ("wheel", "wheel"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sounddevice", "sounddevice"),
        ("huggingface_hub", "huggingface_hub"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("omegaconf", "omegaconf"),
        ("nemo_toolkit[asr]", "nemo_toolkit[asr]"),
    ]

    nemo_version = None
    torch_version = None

    for package_spec, display_name in packages:
        current_package += 1
        progress = 0.25 + (current_package / total_packages) * 0.65
        emit_progress("installing_packages", progress,
                     f"Installing {display_name}...", package=display_name)

        try:
            result = subprocess.run(
                [str(venv_python), "-m", "pip", "install", package_spec],
                capture_output=True,
                text=True,
                timeout=1800
            )
            if result.returncode != 0:
                emit_progress("error", progress, f"Failed to install {display_name}: {result.stderr}")
                sys.exit(1)

            if package_spec.startswith("torch") and not torch_version:
                try:
                    version_result = subprocess.run(
                        [str(venv_python), "-c", "import torch; print(torch.__version__)"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if version_result.returncode == 0:
                        torch_version = version_result.stdout.strip()
                except:
                    pass

            if package_spec.startswith("nemo") and not nemo_version:
                try:
                    version_result = subprocess.run(
                        [str(venv_python), "-c", "import nemo; print(nemo.__version__)"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if version_result.returncode == 0:
                        nemo_version = version_result.stdout.strip()
                except:
                    pass

        except subprocess.TimeoutExpired:
            emit_progress("error", progress, f"Installation of {display_name} timed out")
            sys.exit(1)

    emit_progress("verifying", 0.9, "Verifying installation...")

    try:
        result = subprocess.run(
            [str(venv_python), "-c", "import nemo; import torch; print('OK')"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            emit_progress("error", 0.95, f"Verification failed: {result.stderr}")
            sys.exit(1)
    except Exception as e:
        emit_progress("error", 0.95, f"Verification failed: {e}")
        sys.exit(1)

    emit_progress("complete", 1.0, "NeMo installed successfully!",
                 nemo_version=nemo_version, torch_version=torch_version)


if __name__ == "__main__":
    main()
