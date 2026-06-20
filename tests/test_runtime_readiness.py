from routed_memory_experts.runtime_readiness import RuntimeReadiness, runtime_readiness_to_dict


def test_runtime_readiness_serializes_mlx_and_cuda_fields():
    readiness = RuntimeReadiness(
        ollama_available=True,
        vllm_importable=False,
        vllm_metal_importable=True,
        vllm_metal_venv_importable=False,
        sglang_importable=False,
        nvidia_smi_available=False,
        cuda_gpu_detected=False,
        apple_silicon_detected=True,
        mlx_runtime_candidate=True,
        production_adapter_runtime_ready=True,
        blocker=None,
    )
    data = runtime_readiness_to_dict(readiness)
    assert data["ollama_available"] is True
    assert data["vllm_metal_importable"] is True
    assert data["mlx_runtime_candidate"] is True
    assert data["production_adapter_runtime_ready"] is True
    assert data["blocker"] is None
