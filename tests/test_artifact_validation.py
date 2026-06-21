import json

from routed_memory_experts.artifact_validation import validate_artifact, validate_artifacts, validation_results_to_dict


def test_validate_artifact_accepts_openai_proof(tmp_path):
    path = tmp_path / "vllm-metal-proof.json"
    path.write_text(json.dumps({"base_url": "x", "model": "m", "workload_count": 1, "correct_count": 1, "accuracy": 1.0, "records": []}))
    result = validate_artifact(path)
    assert result.valid is True
    assert result.family == "openai-proof"


def test_validate_artifacts_reports_missing_keys(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"workload_count": 1}))
    data = validation_results_to_dict(validate_artifacts(tmp_path))
    assert data["valid"] is False
    assert data["results"][0]["missing_keys"]
