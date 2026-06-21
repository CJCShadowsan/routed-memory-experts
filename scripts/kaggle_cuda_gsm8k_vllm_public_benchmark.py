#!/usr/bin/env python3
"""Kaggle CUDA vLLM runner for the GSM8K public benchmark.

Run in a Kaggle notebook with GPU and internet enabled:

    !python scripts/kaggle_cuda_gsm8k_vllm_public_benchmark.py

This script is separate from `kaggle_cuda_vllm_proof.py`: the existing proof
keeps the Qwen3 TLDR/PTS multi-LoRA cache-tier smoke test, while this script
uses a license-declared Qwen2.5 GSM8K math LoRA for public benchmark quality
measurement.
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
from typing import IO

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
V1_URL = f"{BASE_URL}/v1"
VENV_DIR = ROOT / ".kaggle-gsm8k-venv"
IN_VENV_ENV = "RME_KAGGLE_GSM8K_IN_VENV"
PYTHON_ENV_VARS_TO_DROP = ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE")
VLLM_PACKAGE = "vllm==0.10.2"
TRANSFORMERS_PACKAGE = "transformers>=4.55.0,<5"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MATH_ADAPTER = "tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora"
PUBLIC_BENCHMARK_LIMIT = "8"
PROOF_LIMIT = "8"
CONCURRENCY_REQUESTS = "16"
CONCURRENCY_LEVEL = "4"


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def sanitized_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PYTHON_ENV_VARS_TO_DROP:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run(cmd: list[str], check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, timeout=timeout, env=sanitized_env())


def patch_prometheus_fastapi_route_compat() -> None:
    """Patch vLLM's bundled FastAPI instrumentation for new Starlette routers.

    Kaggle's Python 3.12 environment currently resolves a FastAPI/Starlette stack
    where `prometheus_fastapi_instrumentator.routing._get_route_name` may see a
    Starlette `_IncludedRouter`. Older instrumentator releases assume every route
    object has `.path`, causing every vLLM OpenAI endpoint, including `/health`,
    to return HTTP 500 even after the engine starts successfully.
    """
    candidates = sorted(VENV_DIR.glob("lib/python*/site-packages/prometheus_fastapi_instrumentator/routing.py"))
    if not candidates:
        print("prometheus_fastapi_instrumentator routing.py not found; skipping compatibility patch")
        return
    routing_py = candidates[0]
    text = routing_py.read_text(encoding="utf-8")
    old = "route_name = route.path"
    new = "route_name = getattr(route, \"path\", getattr(route, \"prefix\", \"\"))"
    if new in text:
        print(f"prometheus route compatibility patch already present in {routing_py}")
        return
    if old not in text:
        print(f"Expected route.path assignment not found in {routing_py}; skipping compatibility patch")
        return
    routing_py.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched prometheus route compatibility in {routing_py}")


def create_venv(force: bool = False) -> None:
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
    get_pip = ROOT / ".kaggle-gsm8k-get-pip.py"
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
    run([str(venv_python()), str(get_pip)], timeout=300)


def ensure_venv() -> None:
    if os.environ.get(IN_VENV_ENV) == "1":
        return
    if not venv_python().exists():
        create_venv()
    py = venv_python()
    try:
        run([str(py), "-m", "pip", "--version"], timeout=60)
    except subprocess.CalledProcessError:
        print("Virtualenv exists but pip is unavailable; recreating it with fallback bootstrap")
        create_venv(force=True)
        py = venv_python()
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<80"], timeout=300)
    run([str(py), "-m", "pip", "install", "-q", "wrapt"], timeout=300)
    run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], timeout=600)
    run([str(py), "-m", "pip", "install", "-q", VLLM_PACKAGE, TRANSFORMERS_PACKAGE], timeout=1200)
    patch_prometheus_fastapi_route_compat()
    env = sanitized_env()
    env[IN_VENV_ENV] = "1"
    os.execve(str(py), [str(py), str(Path(__file__).resolve())], env)


def tail_file(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def wait_for_health(server: subprocess.Popen, log_path: Path, timeout_s: int = 300) -> None:
    start = time.time()
    last_error = None
    while time.time() - start < timeout_s:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {server.returncode}.\n{tail_file(log_path)}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    print("vLLM health check passed")
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become healthy; last error={last_error!r}.\n{tail_file(log_path)}")


def stop(server: subprocess.Popen | None, log_handle: IO[str] | None) -> None:
    if server is not None and server.poll() is None:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)
    if log_handle is not None:
        log_handle.close()


def rme_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "routed_memory_experts.cli", *args]


def save_models() -> None:
    with urllib.request.urlopen(f"{V1_URL}/models", timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    out = ROOT / "runs" / "cuda-vllm-gsm8k-models.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def run_benchmark_commands() -> None:
    commands = [
        rme_cmd("benchmark-public-openai", "--base-url", V1_URL, "--base-model", BASE_MODEL, "--expert-model", "math", "--workload", "workloads/gsm8k_public_sample.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-gsm8k-public-openai-benchmark.json", "--limit", PUBLIC_BENCHMARK_LIMIT),
        rme_cmd("prove-openai", "--base-url", V1_URL, "--model", "math", "--workload", "workloads/gsm8k_public_sample.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-gsm8k-math-proof.json", "--limit", PROOF_LIMIT, "--min-accuracy", "0.0"),
        rme_cmd("benchmark-openai-concurrency", "--base-url", V1_URL, "--model", "math", "--workload", "workloads/gsm8k_public_sample.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-gsm8k-concurrency.json", "--requests", CONCURRENCY_REQUESTS, "--concurrency", CONCURRENCY_LEVEL),
        rme_cmd("summarize-proof-gaps", "--runs", "runs", "--output", "runs/proof-gap-ledger.json"),
        rme_cmd("validate-artifacts", "--path", "runs"),
    ]
    for cmd in commands:
        run(cmd, timeout=1800)


def main() -> int:
    for key in PYTHON_ENV_VARS_TO_DROP:
        os.environ.pop(key, None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    ensure_venv()
    run([sys.executable, "-m", "pytest", "-q"], timeout=300)
    run(["nvidia-smi"], check=False, timeout=60)

    server_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        BASE_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "768",
        "--gpu-memory-utilization",
        "0.55",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "1024",
        "--enforce-eager",
        "--enable-lora",
        "--max-loras",
        "1",
        "--max-lora-rank",
        "64",
        "--lora-modules",
        f"math={MATH_ADAPTER}",
    ]
    attempts = [
        ("v0-xformers", {"VLLM_USE_V1": "0", "VLLM_ATTENTION_BACKEND": "XFORMERS", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        ("v0-torch-sdpa", {"VLLM_USE_V1": "0", "VLLM_ATTENTION_BACKEND": "TORCH_SDPA", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        ("default", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ]
    errors: list[str] = []
    for label, overrides in attempts:
        log_path = ROOT / "runs" / f"kaggle-vllm-gsm8k-startup-{label}.log"
        log_path.parent.mkdir(exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        env = sanitized_env()
        env.update(overrides)
        server = None
        try:
            print(f"Starting vLLM GSM8K attempt {label}: {overrides}")
            server = subprocess.Popen(server_cmd, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
            wait_for_health(server, log_path)
            save_models()
            run_benchmark_commands()
            print("CUDA GSM8K public benchmark complete. Download runs/cuda-vllm-gsm8k-*.json and startup logs.")
            return 0
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(f"[{label}] {exc}\n{tail_file(log_path)}")
            print(f"Attempt {label} failed; trying next fallback")
        finally:
            stop(server, log_handle)
            print(f"--- log tail {log_path} ---")
            print(tail_file(log_path))
    raise RuntimeError("All GSM8K CUDA vLLM attempts failed:\n" + "\n\n".join(errors))


if __name__ == "__main__":
    raise SystemExit(main())
