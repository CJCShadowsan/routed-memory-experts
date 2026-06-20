from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RuntimeReadiness:
    ollama_available: bool
    vllm_importable: bool
    sglang_importable: bool
    nvidia_smi_available: bool
    cuda_gpu_detected: bool
    production_adapter_runtime_ready: bool
    blocker: str | None


def check_runtime_readiness(output_path: str | Path | None = None) -> RuntimeReadiness:
    ollama_available = shutil.which("ollama") is not None
    vllm_importable = importlib.util.find_spec("vllm") is not None
    sglang_importable = importlib.util.find_spec("sglang") is not None
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_smi_available = nvidia_smi is not None
    cuda_gpu_detected = False
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "-L"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
            cuda_gpu_detected = result.returncode == 0 and "GPU" in result.stdout
        except Exception:
            cuda_gpu_detected = False
    production_ready = (vllm_importable or sglang_importable) and cuda_gpu_detected
    blocker = None
    if not production_ready:
        blocker = "Production LoRA/vLLM/SGLang proof requires an importable serving runtime and CUDA GPU; current host only proves local Ollama/context-routed backend."
    readiness = RuntimeReadiness(
        ollama_available=ollama_available,
        vllm_importable=vllm_importable,
        sglang_importable=sglang_importable,
        nvidia_smi_available=nvidia_smi_available,
        cuda_gpu_detected=cuda_gpu_detected,
        production_adapter_runtime_ready=production_ready,
        blocker=blocker,
    )
    if output_path:
        write_runtime_readiness(readiness, output_path)
    return readiness


def runtime_readiness_to_dict(readiness: RuntimeReadiness) -> dict:
    return readiness.__dict__.copy()


def write_runtime_readiness(readiness: RuntimeReadiness, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime_readiness_to_dict(readiness), indent=2, sort_keys=True) + "\n", encoding="utf-8")
