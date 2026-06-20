from routed_memory_experts.runtime_readiness import RuntimeReadiness, runtime_readiness_to_dict


def test_runtime_readiness_serializes_blocker():
    readiness = RuntimeReadiness(
        ollama_available=True,
        vllm_importable=False,
        sglang_importable=False,
        nvidia_smi_available=False,
        cuda_gpu_detected=False,
        production_adapter_runtime_ready=False,
        blocker="needs CUDA",
    )
    data = runtime_readiness_to_dict(readiness)
    assert data["ollama_available"] is True
    assert data["production_adapter_runtime_ready"] is False
    assert data["blocker"] == "needs CUDA"
