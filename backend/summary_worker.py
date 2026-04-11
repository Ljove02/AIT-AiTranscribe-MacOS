#!/usr/bin/env python3
"""
One-shot summarization worker.

This process is started by the backend, loads a single summary model, streams
generation progress as JSON lines, and exits. Exiting the process guarantees the
model weights and KV cache are fully released after each request.
"""

from __future__ import annotations

import argparse
import json
import gc
import sys
import time
from pathlib import Path

from summary_manager import build_custom_summary_prompt


def emit(event: str, **payload):
    print(json.dumps({"event": event, **payload}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--transcription-file", required=True)
    parser.add_argument("--requests-file", required=True)
    parser.add_argument("--kv-bits", type=float, default=0.0)
    parser.add_argument("--kv-quant-scheme", default="uniform")
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--word-count", type=int, default=0)
    args = parser.parse_args()

    transcription_path = Path(args.transcription_file)
    requests_path = Path(args.requests_file)
    transcript_text = transcription_path.read_text(encoding="utf-8")
    request_payload = json.loads(requests_path.read_text(encoding="utf-8"))
    preset_requests = request_payload.get("presets") or []
    if not preset_requests:
        emit("error", message="No summary preset requests were provided")
        sys.exit(1)

    emit(
        "batch_started",
        preset_ids=[preset["preset_id"] for preset in preset_requests],
        total_presets=len(preset_requests),
        model_path=args.model_path,
        word_count=args.word_count,
    )
    emit("loading_model")

    try:
        from mlx_vlm import load
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        emit("error", message=f"Summary runtime missing mlx-vlm: {exc}")
        sys.exit(1)

    try:
        model, processor = load(args.model_path)
        config = getattr(model, "config", None)
    except KeyboardInterrupt:
        emit("cancelled")
        sys.exit(130)
    except Exception as exc:
        emit("error", message=f"Failed to load summary model: {exc}")
        sys.exit(1)

    try:
        for preset_index, preset in enumerate(preset_requests, start=1):
            preset_start = time.time()
            prompt = build_custom_summary_prompt(
                system_prompt=preset["system_prompt"],
                transcript_text=transcript_text,
                target_words=int(preset["target_words"]),
                display_name=preset.get("display_name"),
            )
            formatted_prompt = apply_chat_template(processor, config, prompt, num_images=0)

            emit(
                "preset_started",
                preset_id=preset["preset_id"],
                display_name=preset["display_name"],
                file_name=preset["file_name"],
                target_words=preset["target_words"],
                max_output_tokens=preset["max_output_tokens"],
                batch_index=preset_index,
                total_presets=len(preset_requests),
            )

            emit(
                "generating",
                preset_id=preset["preset_id"],
                display_name=preset["display_name"],
                file_name=preset["file_name"],
                target_words=preset["target_words"],
                max_output_tokens=preset["max_output_tokens"],
                batch_index=preset_index,
                total_presets=len(preset_requests),
                kv_quantized=args.kv_bits > 0,
                kv_bits=args.kv_bits if args.kv_bits > 0 else None,
                kv_quant_scheme=args.kv_quant_scheme if args.kv_bits > 0 else None,
            )

            chunks: list[str] = []
            last_emit = time.time()
            stream = stream_generate(
                model,
                processor,
                formatted_prompt,
                [],
                max_tokens=int(preset["max_output_tokens"]),
                verbose=False,
                kv_bits=(args.kv_bits if args.kv_bits > 0 else None),
                kv_quant_scheme=(args.kv_quant_scheme if args.kv_bits > 0 else None),
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text is None and isinstance(chunk, dict):
                    text = chunk.get("text")
                if not text:
                    continue
                chunks.append(text)
                now = time.time()
                if now - last_emit >= 0.1:
                    emit(
                        "partial",
                        preset_id=preset["preset_id"],
                        display_name=preset["display_name"],
                        file_name=preset["file_name"],
                        batch_index=preset_index,
                        total_presets=len(preset_requests),
                        text="".join(chunks),
                    )
                    last_emit = now

            full_text = "".join(chunks).strip()
            emit(
                "done",
                preset_id=preset["preset_id"],
                display_name=preset["display_name"],
                file_name=preset["file_name"],
                target_words=preset["target_words"],
                max_output_tokens=preset["max_output_tokens"],
                batch_index=preset_index,
                total_presets=len(preset_requests),
                text=full_text,
                processing_time_seconds=(time.time() - preset_start),
                output_word_count=len(full_text.split()),
            )

            try:
                import mlx.core as mx

                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception:
                pass
            gc.collect()
    except KeyboardInterrupt:
        emit("cancelled")
        sys.exit(130)
    except Exception as exc:
        emit("error", message=str(exc))
        sys.exit(1)

    emit("batch_complete", preset_ids=[preset["preset_id"] for preset in preset_requests], total_presets=len(preset_requests))


if __name__ == "__main__":
    main()
