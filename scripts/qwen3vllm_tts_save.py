#!/usr/bin/env python3
"""
Request vLLM-Omni Qwen3-TTS PCM streaming and save the result as a WAV file.

This script is intentionally standalone: it does not import LiveTalking avatar
code, so it can be used to verify whether TTS itself returns complete audio for
long text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import requests


def read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Please provide --text, --text-file, or pipe text via stdin.")


def build_payload(args: argparse.Namespace, text: str) -> dict:
    payload = {
        "input": text,
        "voice": args.voice,
        "language": args.language,
        "task_type": args.task_type,
        "stream": True,
        "response_format": "pcm",
    }
    if args.model:
        payload["model"] = args.model
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    if args.ref_audio:
        payload["ref_audio"] = args.ref_audio
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    return payload


def save_stream(args: argparse.Namespace, payload: dict) -> tuple[int, float | None, float, float]:
    url = f"{args.server.rstrip('/')}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    first_chunk_seconds = None
    bytes_written = 0

    with requests.post(
        url,
        json=payload,
        headers=headers,
        stream=True,
        timeout=(args.connect_timeout, args.read_timeout),
    ) as response:
        headers_time = time.perf_counter()
        if response.status_code != 200:
            body = response.text
            raise RuntimeError(f"TTS request failed: status={response.status_code}, body={body}")

        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(args.channels)
            wav.setsampwidth(args.sample_width)
            wav.setframerate(args.sample_rate)

            for chunk in response.iter_content(chunk_size=args.chunk_bytes):
                if not chunk:
                    continue
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - start
                wav.writeframes(chunk)
                bytes_written += len(chunk)

    done = time.perf_counter()
    return bytes_written, first_chunk_seconds, headers_time - start, done - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save vLLM-Omni Qwen3-TTS PCM streaming output as WAV.")
    parser.add_argument("--server", default="http://127.0.0.1:8091")
    parser.add_argument("--output", "-o", default="qwen3vllm-output.wav")
    parser.add_argument("--text", "-t", default="")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--task-type", default="CustomVoice", choices=["CustomVoice", "VoiceDesign", "Base"])
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--chunk-bytes", type=int, default=960, help="960 bytes is 20ms of 24kHz s16 mono PCM.")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sample-width", type=int, default=2, help="Bytes per sample. PCM s16 is 2.")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--print-payload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    text = read_text(args).strip()
    if not text:
        raise SystemExit("Input text is empty.")

    payload = build_payload(args, text)
    if args.print_payload:
        safe_payload = dict(payload)
        if len(safe_payload["input"]) > 120:
            safe_payload["input"] = safe_payload["input"][:120] + "..."
        print(json.dumps(safe_payload, ensure_ascii=False, indent=2))

    bytes_written, first_chunk_seconds, headers_seconds, total_seconds = save_stream(args, payload)
    duration_seconds = bytes_written / (args.sample_rate * args.channels * args.sample_width)

    print(f"output: {Path(args.output).resolve()}")
    print(f"text_chars: {len(text)}")
    print(f"bytes_written: {bytes_written}")
    print(f"audio_duration_ms: {duration_seconds * 1000:.1f}")
    print(f"http_headers_ms: {headers_seconds * 1000:.1f}")
    if first_chunk_seconds is None:
        print("first_pcm_chunk_ms: -")
    else:
        print(f"first_pcm_chunk_ms: {first_chunk_seconds * 1000:.1f}")
    print(f"total_ms: {total_seconds * 1000:.1f}")

    if bytes_written == 0:
        raise SystemExit("No PCM bytes were received.")


if __name__ == "__main__":
    main()
