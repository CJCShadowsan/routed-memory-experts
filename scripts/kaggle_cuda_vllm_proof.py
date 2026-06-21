#!/usr/bin/env python3
"""Kaggle CUDA vLLM proof runner for routed-memory-experts.

Run this inside a Kaggle Notebook with GPU enabled and internet enabled:

    !python scripts/kaggle_cuda_vllm_proof.py

The script creates an isolated virtualenv under .kaggle-venv before installing
vLLM. This avoids mutating Kaggle/Colab's large preloaded global environment,
which otherwise produces many unrelated dependency-conflict warnings.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
V1_URL = f"{BASE_URL}/v1"
VENV_DIR = ROOT / ".kaggle-venv"
IN_VENV_ENV = "RME_KAGGLE_PROOF_IN_VENV"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, timeout=timeout)


def ensure_isolated_venv() -> None:
    """Create a clean venv and re-exec inside it.

    Kaggle/Colab images preinstall many packages with tight pins. Installing
    vLLM into that global Python can upgrade numpy/protobuf/cuda libraries and
    trigger scary but unrelated resolver warnings. A venv keeps this proof's
    dependencies separate from the notebook image.
    """
    if os.environ.get(IN_VENV_ENV) == "1":
        return

    py = venv_python()
    if not py.exists():
        create_venv_with_pip_fallback()

    try:
        run([str(py), "-m", "pip", "--version"], timeout=60)
    except subprocess.CalledProcessError:
        print("Virtualenv exists but pip is unavailable; recreating it with fallback bootstrap")
        create_venv_with_pip_fallback(force=True)

    print("Installing proof dependencies into isolated virtualenv")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], timeout=300)
    # Some hosted notebook images expose CUDA/Python packages whose optional
    # imports assume wrapt is present, while fresh isolated venvs do not include
    # it. Install it explicitly so vLLM/torch-adjacent import paths do not fail
    # with `ModuleNotFoundError: No module named 'wrapt'`.
    run([str(py), "-m", "pip", "install", "-q", "wrapt"], timeout=300)
    run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], timeout=600)
    run([str(py), "-m", "pip", "install", "-q", "vllm"], timeout=1200)

    env = os.environ.copy()
    env[IN_VENV_ENV] = "1"
    print("Re-running proof inside isolated virtualenv")
    os.execve(str(py), [str(py), str(Path(__file__).resolve())], env)


def create_venv_with_pip_fallback(force: bool = False) -> None:
    """Create the proof venv, falling back when ensurepip is unavailable.

    Some hosted notebook Python builds fail during `venv(..., with_pip=True)`
    because the inner `python -m ensurepip` command exits non-zero. In that
    case we create the venv without pip and bootstrap pip with get-pip.py.
    """
    if force and VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)

    print(f"Creating isolated virtualenv at {VENV_DIR}")
    try:
        venv.EnvBuilder(with_pip=True, clear=True, symlinks=True).create(VENV_DIR)
        return
    except subprocess.CalledProcessError as exc:
        print(f"venv ensurepip failed: {exc}; retrying with get-pip.py bootstrap")
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)

    venv.EnvBuilder(with_pip=False, clear=True, symlinks=True).create(VENV_DIR)
    get_pip = ROOT / ".kaggle-get-pip.py"
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
    run([str(venv_python()), str(get_pip)], timeout=300)


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
    ensure_isolated_venv()
    print("Python:", sys.version)
    print("Python executable:", sys.executable)
    run([sys.executable, "-m", "pytest", "-q"])
    run(["nvidia-smi"], check=False, timeout=60)

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
