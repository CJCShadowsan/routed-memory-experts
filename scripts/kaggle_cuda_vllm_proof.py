#!/usr/bin/env python3
"""Kaggle CUDA vLLM proof runner for routed-memory-experts.

Run this inside a Kaggle Notebook with GPU enabled and internet enabled:

    !python scripts/kaggle_cuda_vllm_proof.py

It installs vLLM if needed, starts a local CUDA vLLM OpenAI-compatible server,
loads two Qwen3-0.6B LoRA adapters, runs the repository proof commands, and
writes CUDA-specific artifacts under runs/.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
V1_URL = f"{BASE_URL}/v1"


def run(cmd: list[str], check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, timeout=timeout)


def wait_for_health(timeout_s: int = 300) -> None:
    start = time.time()
    last_error = None
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    print("vLLM health check passed")
                    return
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become healthy within {timeout_s}s; last error={last_error!r}")


def save_models() -> None:
    with urllib.request.urlopen(f"{V1_URL}/models", timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    out = ROOT / "runs" / "cuda-vllm-models.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> int:
    print("Python:", sys.version)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"])
    run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    run([sys.executable, "-m", "pytest", "-q"])

    server_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "1024",
        "--enable-lora",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "4",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        "tldr=phh/Qwen3-0.6B-TLDR-Lora",
        "pts=codelion/Qwen3-0.6B-PTS-DPO-LoRA",
    ]
    print("Starting CUDA vLLM server with max_cpu_loras > max_loras...")
    server = subprocess.Popen(server_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_health()
        save_models()
        commands = [
            ["rme", "prove-openai", "--base-url", V1_URL, "--model", "tldr", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-tldr-proof.json", "--limit", "6", "--min-accuracy", "0.75"],
            ["rme", "prove-openai", "--base-url", V1_URL, "--model", "pts", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-pts-proof.json", "--limit", "6", "--min-accuracy", "0.50"],
            ["rme", "compare-openai-models", "--base-url", V1_URL, "--base-model", "Qwen/Qwen3-0.6B", "--expert-model", "tldr", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-base-vs-tldr.json", "--limit", "6"],
            ["rme", "benchmark-openai-concurrency", "--base-url", V1_URL, "--model", "tldr", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-concurrency.json", "--requests", "24", "--concurrency", "4"],
            ["rme", "validate-artifacts", "--path", "runs"],
        ]
        for cmd in commands:
            run(cmd, timeout=600)
        print("CUDA vLLM proof complete. Download runs/cuda-*.json from Kaggle output.")
        return 0
    finally:
        print("Stopping vLLM server...")
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        if server.stdout:
            tail = server.stdout.read()[-4000:]
            print("--- vLLM server output tail ---")
            print(tail)


if __name__ == "__main__":
    raise SystemExit(main())
