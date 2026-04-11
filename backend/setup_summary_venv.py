#!/usr/bin/env python3
"""
Summary runtime setup script.

Creates a dedicated virtual environment for MLX-based summarization and emits
JSON progress events for the Swift frontend.
"""

import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def emit(step: str, progress: float, message: str, **extra):
    print(json.dumps({"step": step, "progress": progress, "message": message, **extra}), flush=True)


def fail(message: str, details: str | None = None):
    payload = {"step": "error", "progress": -1, "message": message}
    if details:
        payload["details"] = details
    print(json.dumps(payload), flush=True)


def create_venv(venv_path: Path) -> bool:
    emit("creating_venv", 0.1, f"Creating summary runtime at {venv_path}...")
    try:
        if venv_path.exists():
            shutil.rmtree(venv_path)
        venv.create(venv_path, with_pip=True)
        return True
    except Exception as exc:
        fail(f"Failed to create summary runtime: {exc}")
        return False


def upgrade_pip(venv_path: Path) -> bool:
    emit("upgrading_pip", 0.2, "Upgrading pip...")
    pip_path = venv_path / "bin" / "pip"
    try:
        result = subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            fail("Failed to upgrade pip", result.stderr)
            return False
        return True
    except Exception as exc:
        fail(f"Failed to upgrade pip: {exc}")
        return False


def install_requirements(venv_path: Path, requirements_path: Path) -> bool:
    pip_path = venv_path / "bin" / "pip"
    packages = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            packages.append(line)

    if not packages:
        fail("No packages found in requirements-summary.txt")
        return False

    total = len(packages)
    for index, package in enumerate(packages, start=1):
        progress = 0.3 + (0.5 * ((index - 1) / max(total, 1)))
        emit("installing_packages", progress, f"Installing {package}... ({index}/{total})", package=package)
        try:
            result = subprocess.run(
                [str(pip_path), "install", package],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                fail(f"Failed to install {package}", result.stderr)
                return False
        except Exception as exc:
            fail(f"Failed to install {package}: {exc}")
            return False
    return True


def verify(venv_path: Path) -> bool:
    emit("verifying", 0.9, "Verifying summary runtime...")
    python_path = venv_path / "bin" / "python3"
    script = (
        "import mlx, mlx_vlm, huggingface_hub, psutil; "
        "from mlx_vlm.generate import stream_generate; "
        "print(f'mlx={mlx.__version__}'); "
        "print(f'mlx_vlm={mlx_vlm.__version__}'); "
        "print(f'psutil={psutil.__version__}'); "
        "print(f'stream_generate={callable(stream_generate)}')"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            fail("Summary runtime verification failed", result.stderr)
            return False
        versions = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                versions[key] = value
        emit(
            "complete",
            1.0,
            "Summary runtime installed successfully",
            mlx_version=versions.get("mlx", "unknown"),
            mlx_vlm_version=versions.get("mlx_vlm", "unknown"),
            venv_path=str(venv_path),
        )
        return True
    except Exception as exc:
        fail(f"Failed to verify summary runtime: {exc}")
        return False


def main():
    if len(sys.argv) < 2:
        fail("Usage: python3 setup_summary_venv.py /path/to/venv")
        sys.exit(1)

    venv_path = Path(sys.argv[1]).expanduser().resolve()
    requirements_path = Path(__file__).parent / "requirements-summary.txt"
    if not requirements_path.exists():
        fail(f"requirements-summary.txt not found at {requirements_path}")
        sys.exit(1)

    emit("checking_python", 0.0, f"Checking Python version... ({sys.version_info.major}.{sys.version_info.minor})")
    if sys.version_info < (3, 10):
        fail("Python 3.10+ required for summary runtime")
        sys.exit(1)

    if not create_venv(venv_path):
        sys.exit(1)
    if not upgrade_pip(venv_path):
        sys.exit(1)
    if not install_requirements(venv_path, requirements_path):
        sys.exit(1)
    if not verify(venv_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
