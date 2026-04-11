"""
Summary model management for session summarization.

This module keeps summarization concerns separate from ASR:
- Summary runtime detection (summary-venv)
- Summary model registry and download status
- Model download/storage paths
- Token and memory estimation helpers
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from huggingface_hub import snapshot_download

from model_manager import get_storage_dir


SUMMARY_RUNTIME_NAME = "summary-venv"
SUMMARY_MODELS_DIRNAME = "summary"
SUMMARY_PROMPT_OVERHEAD_TOKENS = 512


class SummaryPreset:
    GENERAL = "general"
    MEETING_NOTES = "meeting_notes"
    ACTION_ITEMS = "action_items"
    TECHNICAL = "technical"


SUMMARY_PRESETS = {
    SummaryPreset.GENERAL: {
        "id": SummaryPreset.GENERAL,
        "display_name": "General Summary",
        "reserved_output_tokens": 1024,
        "file_name": "summary-general.md",
        "system_prompt": (
            "You are a precise summarization assistant. "
            "Summarize the provided transcript in the same language as the transcript. "
            "Return Markdown with exactly these sections: "
            "Overview, Key Points, Notable Details."
        ),
    },
    SummaryPreset.MEETING_NOTES: {
        "id": SummaryPreset.MEETING_NOTES,
        "display_name": "Meeting Notes",
        "reserved_output_tokens": 1024,
        "file_name": "summary-meeting-notes.md",
        "system_prompt": (
            "You are a meeting-notes assistant. "
            "Turn the transcript into clean meeting notes in the same language as the transcript. "
            "Return Markdown with exactly these sections: "
            "Overview, Decisions, Open Questions, Next Steps."
        ),
    },
    SummaryPreset.ACTION_ITEMS: {
        "id": SummaryPreset.ACTION_ITEMS,
        "display_name": "Action Items",
        "reserved_output_tokens": 512,
        "file_name": "summary-action-items.md",
        "system_prompt": (
            "You extract execution-ready action items from transcripts. "
            "Write in the same language as the transcript. "
            "Return Markdown with exactly these sections: "
            "Action Items, Follow-ups, Risks or Blockers."
        ),
    },
    SummaryPreset.TECHNICAL: {
        "id": SummaryPreset.TECHNICAL,
        "display_name": "Technical Summary",
        "reserved_output_tokens": 1408,
        "file_name": "summary-technical.md",
        "system_prompt": (
            "You are a technical summarization assistant. "
            "Focus on architecture, implementation details, APIs, systems behavior, technical tradeoffs, bugs, and follow-up engineering work. "
            "Write in the same language as the transcript and keep technical terminology intact. "
            "Return Markdown with exactly these sections: "
            "Technical Overview, Key Implementation Details, Risks or Edge Cases, Recommended Follow-up."
        ),
    },
}


SUMMARY_MODELS = {
    "gemma-4-e2b-it-4bit": {
        "id": "gemma-4-e2b-it-4bit",
        "display_name": "Gemma 4 E2B Instruct 4-bit",
        "provider": "Google / MLX Community",
        "engine": "mlx-vlm",
        "quantization": "4-bit",
        "context_tokens": 131072,
        "download_size_mb": 3610,
        "resident_model_ram_mb": 3072,
        "recommended": True,
        "hf_repo_id": "mlx-community/gemma-4-e2b-it-4bit",
        "model_url": "https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit",
        "text_config": {
            "num_hidden_layers": 35,
            "num_key_value_heads": 1,
            "head_dim": 256,
        },
    },
    "gemma-4-e4b-it-4bit": {
        "id": "gemma-4-e4b-it-4bit",
        "display_name": "Gemma 4 E4B Instruct 4-bit",
        "provider": "Google / MLX Community",
        "engine": "mlx-vlm",
        "quantization": "4-bit",
        "context_tokens": 131072,
        "download_size_mb": 5220,
        "resident_model_ram_mb": 5120,
        "recommended": False,
        "hf_repo_id": "mlx-community/gemma-4-e4b-it-4bit",
        "model_url": "https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit",
        "text_config": {
            "num_hidden_layers": 42,
            "num_key_value_heads": 2,
            "head_dim": 256,
        },
    },
}


@dataclass
class SummaryRuntimeStatus:
    installed: bool
    ready: bool
    venv_path: str
    python_path: Optional[str]
    worker_path: Optional[str]
    requirements_path: Optional[str]
    mlx_vlm_available: bool


def get_app_support_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AiTranscribe"
    return Path.home() / ".aitranscribe"


def get_summary_runtime_dir() -> Path:
    path = get_app_support_dir() / SUMMARY_RUNTIME_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_summary_python_path() -> Path:
    return get_summary_runtime_dir() / "bin" / "python3"


def get_summary_models_dir() -> Path:
    path = get_storage_dir() / SUMMARY_MODELS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_summary_model_dir(model_id: str) -> Path:
    return get_summary_models_dir() / model_id


def get_summary_worker_resource_path() -> Optional[Path]:
    script_dir = Path(__file__).parent
    bundled = script_dir / "summary_worker.py"
    if bundled.exists():
        return bundled
    return None


def get_summary_requirements_resource_path() -> Optional[Path]:
    script_dir = Path(__file__).parent
    bundled = script_dir / "requirements-summary.txt"
    if bundled.exists():
        return bundled
    return None


def get_summary_setup_resource_path() -> Optional[Path]:
    script_dir = Path(__file__).parent
    bundled = script_dir / "setup_summary_venv.py"
    if bundled.exists():
        return bundled
    return None


def get_runtime_status() -> SummaryRuntimeStatus:
    python_path = get_summary_python_path()
    worker_path = get_summary_worker_resource_path()
    requirements_path = get_summary_requirements_resource_path()

    installed = python_path.exists()
    mlx_vlm_available = False

    if installed:
        try:
            import subprocess

            result = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    (
                        "import mlx_vlm, psutil; "
                        "from mlx_vlm.generate import stream_generate; "
                        "print(mlx_vlm.__version__)"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            mlx_vlm_available = result.returncode == 0
        except Exception:
            mlx_vlm_available = False

    return SummaryRuntimeStatus(
        installed=installed,
        ready=installed and worker_path is not None and mlx_vlm_available,
        venv_path=str(get_summary_runtime_dir()),
        python_path=str(python_path) if installed else None,
        worker_path=str(worker_path) if worker_path else None,
        requirements_path=str(requirements_path) if requirements_path else None,
        mlx_vlm_available=mlx_vlm_available,
    )


def estimate_transcript_tokens(word_count: int) -> int:
    return max(1, math.ceil(word_count * 1.35))


def estimate_output_tokens(target_words: int) -> int:
    clamped_target_words = max(120, min(int(target_words), 4000))
    return max(256, int(math.ceil(clamped_target_words * 1.45)))


def sanitize_summary_file_name(file_name: str, fallback_id: str) -> str:
    candidate = (file_name or "").strip().lower()
    if not candidate:
        candidate = f"summary-{fallback_id}.md"
    candidate = candidate.replace(" ", "-")
    candidate = re.sub(r"[^a-z0-9._-]+", "-", candidate)
    if not candidate.endswith(".md"):
        candidate = f"{candidate}.md"
    if not candidate.startswith("summary-"):
        candidate = f"summary-{candidate}"
    return candidate


def normalize_summary_request(request: dict) -> dict:
    preset_id = str(request.get("preset_id") or "").strip()
    fallback = SUMMARY_PRESETS.get(preset_id, {})
    display_name = str(request.get("display_name") or fallback.get("display_name") or preset_id or "Summary").strip()
    system_prompt = str(request.get("system_prompt") or fallback.get("system_prompt") or "").strip()
    if not preset_id:
        raise ValueError("Summary preset is missing an id")
    if not display_name:
        raise ValueError(f"Summary preset {preset_id} is missing a display name")
    if not system_prompt:
        raise ValueError(f"Summary preset {preset_id} is missing prompt instructions")

    target_words = int(request.get("target_words") or 0)
    if target_words <= 0:
        target_words = 520

    max_output_tokens = int(request.get("max_output_tokens") or 0)
    if max_output_tokens <= 0:
        max_output_tokens = estimate_output_tokens(target_words)

    file_name = sanitize_summary_file_name(
        str(request.get("file_name") or fallback.get("file_name") or ""),
        fallback_id=preset_id.replace("_", "-"),
    )

    return {
        "preset_id": preset_id,
        "display_name": display_name,
        "system_prompt": system_prompt,
        "target_words": target_words,
        "max_output_tokens": max_output_tokens,
        "file_name": file_name,
    }


def resolve_text_config(model_id: str, model_path: Optional[Path] = None) -> dict:
    default = SUMMARY_MODELS[model_id]["text_config"].copy()
    if model_path is None:
        return default

    config_path = model_path / "config.json"
    if not config_path.exists():
        return default

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config") or {}
        return {
            "num_hidden_layers": int(text_config.get("num_hidden_layers", default["num_hidden_layers"])),
            "num_key_value_heads": int(text_config.get("num_key_value_heads", default["num_key_value_heads"])),
            "head_dim": int(text_config.get("head_dim", default["head_dim"])),
        }
    except Exception:
        return default


def estimate_kv_cache_bytes(
    model_id: str,
    required_kv_tokens: int,
    quantized: bool,
    model_path: Optional[Path] = None,
) -> int:
    text_config = resolve_text_config(model_id, model_path=model_path)
    bytes_per_value = 1 if quantized else 2
    return (
        2
        * text_config["num_hidden_layers"]
        * text_config["num_key_value_heads"]
        * text_config["head_dim"]
        * bytes_per_value
        * required_kv_tokens
    )


def get_kv_bytes_per_token(model_id: str, quantized: bool, model_path: Optional[Path] = None) -> int:
    return estimate_kv_cache_bytes(model_id, required_kv_tokens=1, quantized=quantized, model_path=model_path)


def estimate_summary_memory(
    model_id: str,
    word_count: int,
    reserved_output_tokens: int,
    quantized: bool = False,
) -> dict:
    transcript_tokens = estimate_transcript_tokens(word_count)
    required_context_tokens = transcript_tokens + SUMMARY_PROMPT_OVERHEAD_TOKENS + reserved_output_tokens
    kv_bytes = estimate_kv_cache_bytes(
        model_id=model_id,
        required_kv_tokens=transcript_tokens + reserved_output_tokens,
        quantized=quantized,
        model_path=get_summary_model_dir(model_id) if get_summary_model_dir(model_id).exists() else None,
    )
    kv_mb = int(math.ceil(kv_bytes / (1024 * 1024)))
    resident_model_ram_mb = SUMMARY_MODELS[model_id]["resident_model_ram_mb"]
    total_estimated_mb = resident_model_ram_mb + kv_mb

    physical_ram_mb = int(get_physical_memory_mb())
    memory_budget_mb = max(0, physical_ram_mb - 4096)

    return {
        "estimated_prompt_tokens": transcript_tokens + SUMMARY_PROMPT_OVERHEAD_TOKENS,
        "estimated_transcript_tokens": transcript_tokens,
        "reserved_output_tokens": reserved_output_tokens,
        "required_context_tokens": required_context_tokens,
        "context_utilization": min(
            1.0, required_context_tokens / float(SUMMARY_MODELS[model_id]["context_tokens"])
        ),
        "resident_model_ram_mb": resident_model_ram_mb,
        "estimated_kv_cache_ram_mb": kv_mb,
        "estimated_total_ram_mb": total_estimated_mb,
        "physical_ram_mb": physical_ram_mb,
        "memory_budget_mb": memory_budget_mb,
        "will_quantize_kv": quantized,
        "fits_memory_budget": total_estimated_mb <= memory_budget_mb if memory_budget_mb > 0 else True,
    }


def should_quantize_kv_cache(model_id: str, word_count: int, reserved_output_tokens: int) -> bool:
    unquantized = estimate_summary_memory(
        model_id,
        word_count,
        reserved_output_tokens=reserved_output_tokens,
        quantized=False,
    )
    if not unquantized["fits_memory_budget"]:
        return True

    resident_model_ram_mb = SUMMARY_MODELS[model_id]["resident_model_ram_mb"]
    safe_threshold_mb = max(2048, int(get_physical_memory_mb()) - 4096 - int(resident_model_ram_mb * 0.25))
    return unquantized["estimated_total_ram_mb"] > safe_threshold_mb


def check_model_downloaded(model_id: str) -> bool:
    model_dir = get_summary_model_dir(model_id)
    return model_dir.exists() and (model_dir / "config.json").exists()


def get_model_info(
    model_id: str,
    word_count: Optional[int] = None,
    reserved_output_tokens: Optional[int] = None,
) -> dict:
    config = SUMMARY_MODELS[model_id].copy()
    model_dir = get_summary_model_dir(model_id)
    downloaded = check_model_downloaded(model_id)
    config["downloaded"] = downloaded
    config["path"] = str(model_dir) if downloaded else None
    config["kv_bytes_per_token_default"] = get_kv_bytes_per_token(
        model_id, quantized=False, model_path=model_dir if downloaded else None
    )
    config["kv_bytes_per_token_quantized"] = get_kv_bytes_per_token(
        model_id, quantized=True, model_path=model_dir if downloaded else None
    )

    if word_count is not None and reserved_output_tokens is not None:
        quantize_kv = should_quantize_kv_cache(model_id, word_count, reserved_output_tokens)
        config["memory_estimate"] = estimate_summary_memory(
            model_id=model_id,
            word_count=word_count,
            reserved_output_tokens=reserved_output_tokens,
            quantized=quantize_kv,
        )

    return config


def get_all_models_info() -> list[dict]:
    return [get_model_info(model_id) for model_id in SUMMARY_MODELS]


def download_summary_model(model_id: str) -> Generator[dict, None, None]:
    if model_id not in SUMMARY_MODELS:
        yield {"status": "error", "message": f"Unknown summary model: {model_id}"}
        return

    config = SUMMARY_MODELS[model_id]
    dest_dir = get_summary_model_dir(model_id)
    if check_model_downloaded(model_id):
        yield {
            "status": "complete",
            "progress": 1.0,
            "downloaded_mb": config["download_size_mb"],
            "total_mb": config["download_size_mb"],
            "path": str(dest_dir),
        }
        return

    try:
        yield {
            "status": "downloading",
            "progress": 0.0,
            "downloaded_mb": 0,
            "total_mb": config["download_size_mb"],
            "message": f"Downloading {config['display_name']}...",
        }
        snapshot_download(
            repo_id=config["hf_repo_id"],
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        yield {
            "status": "complete",
            "progress": 1.0,
            "downloaded_mb": config["download_size_mb"],
            "total_mb": config["download_size_mb"],
            "path": str(dest_dir),
        }
    except Exception as exc:
        yield {"status": "error", "message": str(exc)}


def delete_summary_model(model_id: str) -> bool:
    model_dir = get_summary_model_dir(model_id)
    if not model_dir.exists():
        return False
    shutil.rmtree(model_dir)
    return True


def get_physical_memory_mb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        return 16384


def get_summary_file_name(preset_id: str) -> str:
    return SUMMARY_PRESETS[preset_id]["file_name"]


def get_summary_preset(preset_id: str) -> dict:
    if preset_id not in SUMMARY_PRESETS:
        raise ValueError(f"Unknown summary preset: {preset_id}")
    return SUMMARY_PRESETS[preset_id]


def build_summary_prompt(preset_id: str, transcript_text: str) -> str:
    preset = get_summary_preset(preset_id)
    return build_custom_summary_prompt(
        system_prompt=preset["system_prompt"],
        transcript_text=transcript_text,
        target_words=max(300, int(preset["reserved_output_tokens"] / 1.45)),
        display_name=preset["display_name"],
    )


def build_custom_summary_prompt(
    system_prompt: str,
    transcript_text: str,
    target_words: int,
    display_name: Optional[str] = None,
) -> str:
    detail_hint = (
        f"Aim for roughly {max(120, min(int(target_words), 4000))} words. "
        "It is fine to be shorter when the transcript is thin, but include concrete detail when the transcript supports it."
    )
    label = f"Summary type: {display_name}\n" if display_name else ""
    return (
        f"{system_prompt.strip()}\n\n"
        f"{label}{detail_hint}\n\n"
        "Transcript:\n"
        f"{transcript_text.strip()}"
    )
