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
PYTHON_ENV_VARS_TO_DROP = ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE")
CUDA_LINK_DIR = ROOT / ".cuda-link"
VLLM_LOG = ROOT / "runs" / "kaggle-vllm-startup.log"
VLLM_PACKAGE = "vllm==0.10.2"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def sanitized_env() -> dict[str, str]:
    """Return an environment that does not leak notebook Python hooks.

    Kaggle/Colab can inject `sitecustomize` through PYTHONPATH/PYTHONHOME. In
    an isolated venv that hook may import packages, such as wrapt, before the
    venv has installed them. Dropping Python-specific environment variables
    keeps the venv isolated while preserving CUDA/PATH/LD_LIBRARY_PATH.
    """
    env = os.environ.copy()
    for key in PYTHON_ENV_VARS_TO_DROP:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def sanitize_current_process_env() -> None:
    for key in PYTHON_ENV_VARS_TO_DROP:
        os.environ.pop(key, None)
    os.environ["PYTHONNOUSERSITE"] = "1"


def prepend_env_path(key: str, path: Path) -> None:
    current = os.environ.get(key)
    value = str(path)
    if current:
        parts = current.split(os.pathsep)
        if value not in parts:
            os.environ[key] = os.pathsep.join([value, *parts])
    else:
        os.environ[key] = value


def configure_cuda_linker_env() -> None:
    """Make libcuda discoverable for hosted-notebook JIT linkers.

    Kaggle can expose the NVIDIA driver well enough for `nvidia-smi` while not
    providing an unversioned `libcuda.so` in the directories used by
    FlashInfer/vLLM's JIT linker. If only `libcuda.so.1` exists, create a local
    `libcuda.so` symlink and put it on LIBRARY_PATH/LD_LIBRARY_PATH.
    """
    candidate_dirs = [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/local/cuda/lib64/stubs"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs"),
    ]
    for directory in candidate_dirs:
        if (directory / "libcuda.so").exists():
            prepend_env_path("LIBRARY_PATH", directory)
            prepend_env_path("LD_LIBRARY_PATH", directory)
            print(f"Using existing libcuda.so from {directory}")
            return

    for directory in candidate_dirs:
        versioned = directory / "libcuda.so.1"
        if versioned.exists():
            CUDA_LINK_DIR.mkdir(exist_ok=True)
            target = CUDA_LINK_DIR / "libcuda.so"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(versioned)
            prepend_env_path("LIBRARY_PATH", CUDA_LINK_DIR)
            prepend_env_path("LD_LIBRARY_PATH", CUDA_LINK_DIR)
            print(f"Created {target} -> {versioned} for CUDA JIT linking")
            return

    print("WARNING: no libcuda.so or libcuda.so.1 found in known locations; vLLM JIT linking may fail")


def run(cmd: list[str], check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check, timeout=timeout, env=sanitized_env())


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
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<81"], timeout=300)
    # Some hosted notebook images expose CUDA/Python packages whose optional
    # imports assume wrapt is present, while fresh isolated venvs do not include
    # it. Install it explicitly so vLLM/torch-adjacent import paths do not fail
    # with `ModuleNotFoundError: No module named 'wrapt'`.
    run([str(py), "-m", "pip", "install", "-q", "wrapt"], timeout=300)
    run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], timeout=600)
    # Pin below the current latest vLLM line. On Kaggle T4, latest vLLM can
    # accept `--max-cpu-loras > --max-loras` and pass health checks, but still
    # route every backend attempt through vLLM V1 FlashInfer and crash on the
    # first LoRA prefill with `BatchPrefillWithPagedKVCache ... invalid
    # argument`. vLLM 0.10.2 still has the V0 fallback needed for this proof.
    run([str(py), "-m", "pip", "install", "-q", VLLM_PACKAGE], timeout=1200)

    env = sanitized_env()
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


def tail_file(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def wait_for_health(server: subprocess.Popen, log_path: Path, timeout_s: int = 300) -> None:
    start = time.time()
    last_error = None
    while time.time() - start < timeout_s:
        if server.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before health check passed with code {server.returncode}. "
                f"Startup log tail:\n{tail_file(log_path)}"
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    print("vLLM health check passed")
                    return
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            last_error = exc
        time.sleep(5)
    raise RuntimeError(
        f"vLLM did not become healthy within {timeout_s}s; last error={last_error!r}. "
        f"Startup log tail:\n{tail_file(log_path)}"
    )


def start_vllm_server(server_cmd: list[str], env_overrides: dict[str, str], label: str) -> tuple[subprocess.Popen, object, Path]:
    VLLM_LOG.parent.mkdir(exist_ok=True)
    log_path = VLLM_LOG.with_name(f"kaggle-vllm-startup-{label}.log")
    env = sanitized_env()
    env.update(env_overrides)
    print(f"Starting CUDA vLLM server attempt '{label}' with overrides: {env_overrides}")
    log_handle = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        server_cmd,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return server, log_handle, log_path


def stop_vllm_server(server: subprocess.Popen, log_handle: object | None) -> None:
    if server.poll() is None:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)
    if log_handle is not None:
        log_handle.close()


def save_models() -> None:
    with urllib.request.urlopen(f"{V1_URL}/models", timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    out = ROOT / "runs" / "cuda-vllm-models.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def rme_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "routed_memory_experts.cli", *args]


def run_cuda_proof_commands() -> None:
    commands = [
        rme_cmd("prove-openai", "--base-url", V1_URL, "--model", "tldr", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-tldr-proof.json", "--limit", "6", "--min-accuracy", "0.75"),
        rme_cmd("prove-openai", "--base-url", V1_URL, "--model", "pts", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-pts-proof.json", "--limit", "6", "--min-accuracy", "0.50"),
        rme_cmd("compare-openai-models", "--base-url", V1_URL, "--base-model", "Qwen/Qwen3-0.6B", "--expert-model", "tldr", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-base-vs-tldr.json", "--limit", "6"),
        rme_cmd("benchmark-openai-concurrency", "--base-url", V1_URL, "--model", "tldr", "--workload", "workloads/real_world_v1.jsonl", "--experts", "experts", "--output", "runs/cuda-vllm-concurrency.json", "--requests", "24", "--concurrency", "4"),
        rme_cmd("validate-artifacts", "--path", "runs"),
    ]
    for cmd in commands:
        run(cmd, timeout=600)


def main() -> int:
    sanitize_current_process_env()
    ensure_isolated_venv()
    print("Python:", sys.version)
    print("Python executable:", sys.executable)
    run([sys.executable, "-m", "pip", "show", "vllm"], timeout=60)
    run([sys.executable, "-m", "pytest", "-q"])
    run(["nvidia-smi"], check=False, timeout=60)
    configure_cuda_linker_env()

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
        "768",
        "--gpu-memory-utilization",
        "0.55",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "768",
        "--enforce-eager",
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
    attempts = [
        # Prefer vLLM V0 + non-FlashInfer backends on Kaggle T4. Some vLLM V1
        # builds ignore XFORMERS/TORCH_SDPA for this Qwen3 LoRA path and still
        # execute `vllm/v1/attention/backends/flashinfer.py`, which crashes with
        # `BatchPrefillWithPagedKVCache failed with error invalid argument`.
        ("v0-xformers", {"VLLM_USE_V1": "0", "VLLM_ATTENTION_BACKEND": "XFORMERS", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        ("v0-torch-sdpa", {"VLLM_USE_V1": "0", "VLLM_ATTENTION_BACKEND": "TORCH_SDPA", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        ("xformers", {"VLLM_ATTENTION_BACKEND": "XFORMERS", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        ("torch-sdpa", {"VLLM_ATTENTION_BACKEND": "TORCH_SDPA", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
        ("default", {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ]
    attempt_errors = []
    for label, overrides in attempts:
        server = None
        log_handle = None
        log_path = None
        try:
            server, log_handle, log_path = start_vllm_server(server_cmd, overrides, label)
            wait_for_health(server, log_path)
            print(f"vLLM startup attempt '{label}' succeeded; log: {log_path}")
            save_models()
            run_cuda_proof_commands()
            print("CUDA vLLM proof complete. Download runs/cuda-*.json from Kaggle output.")
            return 0
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            log_tail = tail_file(log_path) if log_path is not None else ""
            attempt_errors.append(f"[{label}] {exc}\n{log_tail}")
            print(f"vLLM proof attempt '{label}' failed; retrying if possible")
        finally:
            print("Stopping vLLM server...")
            if server is not None:
                stop_vllm_server(server, log_handle)
            if log_path is not None:
                print(f"--- vLLM server output tail ({log_path}) ---")
                print(tail_file(log_path))

    raise RuntimeError("All vLLM proof attempts failed:\n" + "\n\n".join(attempt_errors))


if __name__ == "__main__":
    raise SystemExit(main())
