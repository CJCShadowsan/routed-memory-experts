import json

from routed_memory_experts.artifact_validation import validate_artifact
from routed_memory_experts.proof_gap import summarize_proof_gaps


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_summarize_proof_gaps_marks_local_and_external_claims(tmp_path):
    runs = tmp_path / "runs"
    write_json(runs / "proof.json", {"routed_accuracy": 1.0})
    write_json(runs / "benchmark-proof.json", {"routed_accuracy": 1.0})
    write_json(runs / "fleet.json", {"hit_rate": 0.9})
    write_json(runs / "router-comparison.json", {"learned_accuracy": 1.0})
    write_json(runs / "ollama-proof.json", {"accuracy": 1.0})
    write_json(runs / "vllm-metal-multi-lora-models.json", {"data": []})
    write_json(runs / "vllm-metal-manifest-proof.json", {"accuracy": 1.0})
    write_json(runs / "cuda-vllm-models.json", {"data": []})
    write_json(runs / "cuda-vllm-tldr-proof.json", {"accuracy": 1.0})
    write_json(runs / "cuda-vllm-pts-proof.json", {"accuracy": 1.0})
    write_json(
        runs / "cuda-vllm-base-vs-tldr.json",
        {"workload_count": 6, "base_accuracy": 1.0, "expert_accuracy": 1.0, "expert_wins": 0, "expert_losses": 0},
    )
    write_json(runs / "cuda-vllm-concurrency.json", {"request_count": 24, "error_count": 0})

    out = runs / "proof-gap-ledger.json"
    data = summarize_proof_gaps(runs, out)

    statuses = {gap["claim"]: gap["status"] for gap in data["gaps"]}
    assert any(status == "proven_bounded" for status in statuses.values())
    assert any(status == "external_required" for status in statuses.values())
    assert any(status == "blocked_upstream" for status in statuses.values())
    assert data["external_required_count"] >= 2
    assert out.exists()


def test_summarize_proof_gaps_uses_bounded_cuda_gsm8k_evidence(tmp_path):
    runs = tmp_path / "runs"
    write_json(
        runs / "cuda-vllm-gsm8k-public-openai-benchmark.json",
        {
            "artifact_family": "public-openai-benchmark",
            "workload_count": 8,
            "base_accuracy": 0.25,
            "expert_accuracy": 0.375,
            "expert_wins": 1,
            "expert_losses": 0,
            "records": [],
        },
    )

    data = summarize_proof_gaps(runs)
    adapter_gap = next(
        gap for gap in data["gaps"] if gap["claim"] == "High-quality domain adapters beat the base model on a sufficiently large benchmark."
    )

    assert adapter_gap["status"] == "external_required"
    assert str(runs / "cuda-vllm-gsm8k-public-openai-benchmark.json") in adapter_gap["evidence"]
    assert "bounded CUDA GSM8K run shows expert > base" in adapter_gap["remaining"]
    assert ">=30 items" in adapter_gap["next_action"]
    assert "Run scripts/kaggle_cuda_gsm8k_vllm_public_benchmark.py on CUDA" not in adapter_gap["next_action"]


def test_proof_gap_ledger_validates(tmp_path):
    path = tmp_path / "proof-gap-ledger.json"
    path.write_text(
        json.dumps(
            {
                "artifact_family": "proof-gap-ledger",
                "completion_score": 0.5,
                "gap_count": 1,
                "open_gap_count": 1,
                "gaps": [],
            }
        ),
        encoding="utf-8",
    )
    result = validate_artifact(path)
    assert result.valid is True
    assert result.family == "proof-gap-ledger"
