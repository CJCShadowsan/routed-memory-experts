from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMAS: dict[str, set[str]] = {
    "proof": {"workload_count", "routed_correct", "baseline_correct", "routed_accuracy", "baseline_accuracy", "records"},
    "fleet": {"agent_count", "request_count", "hit_rate", "hot_hits", "cold_loads"},
    "router-comparison": {"train_count", "dev_count", "keyword_accuracy", "learned_accuracy", "learned_beats_keyword"},
    "runtime-readiness": {"apple_silicon_detected", "production_adapter_runtime_ready", "blocker"},
    "openai-proof": {"base_url", "model", "workload_count", "correct_count", "accuracy", "records"},
    "ollama-proof": {"model", "workload_count", "correct_count", "accuracy", "records"},
    "openai-comparison": {"base_url", "base_model", "expert_model", "base_accuracy", "expert_accuracy", "records"},
    "concurrency": {"base_url", "model", "request_count", "concurrency", "success_count", "error_count", "records"},
    "models": {"object", "data"},
}


@dataclass
class ArtifactValidationResult:
    path: str
    family: str
    valid: bool
    missing_keys: list[str]


def infer_family(path: Path, data: dict) -> str:
    name = path.name
    if "model-comparison" in name or {"base_model", "expert_model"} <= data.keys():
        return "openai-comparison"
    if "concurrency" in name or {"request_count", "concurrency", "success_count"} <= data.keys():
        return "concurrency"
    if "models" in name and "data" in data:
        return "models"
    if "runtime-readiness" in name:
        return "runtime-readiness"
    if "router-comparison" in name:
        return "router-comparison"
    if "fleet" in name:
        return "fleet"
    if "ollama" in name:
        return "ollama-proof"
    if "vllm-metal" in name or "openai" in name:
        return "openai-proof"
    if "routed_accuracy" in data:
        return "proof"
    return "openai-proof" if "accuracy" in data and "records" in data else "proof"


def validate_artifact(path: str | Path) -> ArtifactValidationResult:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ArtifactValidationResult(str(p), "unknown", False, ["<top-level-object>"])
    family = infer_family(p, data)
    required = SCHEMAS[family]
    missing = sorted(required - data.keys())
    return ArtifactValidationResult(str(p), family, not missing, missing)


def validate_artifacts(path: str | Path) -> list[ArtifactValidationResult]:
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    return [validate_artifact(file) for file in files]


def validation_results_to_dict(results: list[ArtifactValidationResult]) -> dict:
    return {
        "valid": all(result.valid for result in results),
        "artifact_count": len(results),
        "results": [result.__dict__ for result in results],
    }
