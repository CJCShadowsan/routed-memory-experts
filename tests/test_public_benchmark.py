import json

from routed_memory_experts.artifact_validation import validate_artifact
from routed_memory_experts.public_benchmark import build_public_benchmark_prompt
from routed_memory_experts.workload import load_workload


def test_public_benchmark_prompt_does_not_leak_expected_answer():
    prompt = build_public_benchmark_prompt(
        "Janet sells eggs. How much money does she make?",
        "math",
        "Math expert: solve the word problem step by step and end with the final numeric answer.",
    )
    assert "18" not in prompt
    assert "expected" not in prompt.lower()
    assert "Janet sells eggs" in prompt


def test_public_workload_preserves_metadata():
    item = load_workload("workloads/gsm8k_public_sample.jsonl")[0]
    assert item.source == "GSM8K (Grade School Math 8K)"
    assert item.license == "mit"
    assert item.split == "test"
    assert item.provenance_url == "https://huggingface.co/datasets/openai/gsm8k"


def test_public_openai_benchmark_artifact_validates(tmp_path):
    path = tmp_path / "gsm8k-public-openai-benchmark.json"
    path.write_text(
        json.dumps(
            {
                "artifact_family": "public-openai-benchmark",
                "benchmark": "workloads/gsm8k_public_sample.jsonl",
                "workload_count": 1,
                "base_model": "Qwen/Qwen3-0.6B",
                "expert_model": "math",
                "base_accuracy": 0.0,
                "expert_accuracy": 1.0,
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    result = validate_artifact(path)
    assert result.family == "public-openai-benchmark"
    assert result.valid is True
