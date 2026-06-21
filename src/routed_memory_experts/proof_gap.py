from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProofGap:
    claim: str
    status: str
    evidence: list[str]
    remaining: str
    next_action: str


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _has_valid_artifact(runs_path: Path, name: str) -> bool:
    data = _load_json(runs_path / name)
    return bool(data)


def _comparison_shows_adapter_win(data: dict[str, Any], min_items: int) -> bool:
    workload_count = int(data.get("workload_count", 0))
    expert_wins = int(data.get("expert_wins", 0))
    expert_losses = int(data.get("expert_losses", 0))
    expert_accuracy = float(data.get("expert_accuracy", 0.0))
    base_accuracy = float(data.get("base_accuracy", 0.0))
    return workload_count >= min_items and expert_wins > expert_losses and expert_accuracy > base_accuracy


def _largest_concurrency_run(runs_path: Path) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for path in runs_path.glob("*concurrency*.json"):
        data = _load_json(path)
        if not data:
            continue
        if best is None or int(data.get("request_count", 0)) > int(best.get("request_count", 0)):
            best = data | {"path": str(path)}
    return best


def summarize_proof_gaps(runs_path: str | Path = "runs", output_path: str | Path | None = None) -> dict[str, Any]:
    runs = Path(runs_path)
    gaps: list[ProofGap] = []

    deterministic_ready = _has_valid_artifact(runs, "proof.json") and _has_valid_artifact(runs, "benchmark-proof.json")
    gaps.append(
        ProofGap(
            claim="Deterministic routing, cache tiers, and baseline comparison are executable.",
            status="proven" if deterministic_ready else "missing_local_artifact",
            evidence=["runs/proof.json", "runs/benchmark-proof.json"] if deterministic_ready else [],
            remaining="None for the CI-safe control-plane proof." if deterministic_ready else "Run local deterministic proof commands.",
            next_action="Keep as regression gate in scripts/run-local-proofs.sh.",
        )
    )

    gaps.append(
        ProofGap(
            claim="1,000-agent locality simulation supports the hot-set premise.",
            status="proven" if _has_valid_artifact(runs, "fleet.json") else "missing_local_artifact",
            evidence=["runs/fleet.json"] if _has_valid_artifact(runs, "fleet.json") else [],
            remaining="Needs real production traces before claiming traffic realism.",
            next_action="Use production traces or a public trace if one becomes available.",
        )
    )

    gaps.append(
        ProofGap(
            claim="Learned routing can beat keyword routing on synonym-heavy prompts.",
            status="proven" if _has_valid_artifact(runs, "router-comparison.json") else "missing_local_artifact",
            evidence=["runs/router-comparison.json"] if _has_valid_artifact(runs, "router-comparison.json") else [],
            remaining="Needs larger public benchmark routing labels before publication-grade claims.",
            next_action="Build or adapt a public labeled routing benchmark.",
        )
    )

    gaps.append(
        ProofGap(
            claim="A real local neural backend can consume routed specialist context.",
            status="proven" if _has_valid_artifact(runs, "ollama-proof.json") else "missing_local_artifact",
            evidence=["runs/ollama-proof.json"] if _has_valid_artifact(runs, "ollama-proof.json") else [],
            remaining="Context injection is not adapter-weight specialization.",
            next_action="Keep separate from LoRA claims in paper wording.",
        )
    )

    metal_multi = _has_valid_artifact(runs, "vllm-metal-multi-lora-models.json") and _has_valid_artifact(runs, "vllm-metal-manifest-proof.json")
    gaps.append(
        ProofGap(
            claim="Apple Silicon vLLM-Metal can serve multiple LoRA adapters.",
            status="proven" if metal_multi else "missing_live_artifact",
            evidence=["runs/vllm-metal-multi-lora-models.json", "runs/vllm-metal-manifest-proof.json"] if metal_multi else [],
            remaining="Metal still does not implement CPU LoRA cache tiering with max_cpu_loras > max_loras.",
            next_action="Track upstream vLLM-Metal support or design a Metal-specific tiering alternative.",
        )
    )

    cuda_cache = _has_valid_artifact(runs, "cuda-vllm-models.json") and _has_valid_artifact(runs, "cuda-vllm-tldr-proof.json") and _has_valid_artifact(runs, "cuda-vllm-pts-proof.json")
    gaps.append(
        ProofGap(
            claim="CUDA vLLM accepts max_cpu_loras > max_loras and serves multiple LoRA adapters.",
            status="proven_bounded" if cuda_cache else "external_required",
            evidence=["runs/cuda-vllm-models.json", "runs/cuda-vllm-tldr-proof.json", "runs/cuda-vllm-pts-proof.json"] if cuda_cache else [],
            remaining="Bounded Kaggle T4 run; not a production capacity result.",
            next_action="Run longer external CUDA/SGLang proof on controlled NVIDIA hardware.",
        )
    )

    comparisons = [data for path in runs.glob("*base-vs*.json") if (data := _load_json(path))]
    adapter_win = any(_comparison_shows_adapter_win(data, min_items=30) for data in comparisons)
    gaps.append(
        ProofGap(
            claim="High-quality domain adapters beat the base model on a sufficiently large benchmark.",
            status="proven" if adapter_win else "external_required",
            evidence=["runs/*base-vs*.json"] if adapter_win else ["runs/cuda-vllm-base-vs-tldr.json", "runs/vllm-metal-base-vs-lora.json"],
            remaining="Current public adapters prove serving mechanics; CUDA base-vs-TLDR tied on six items.",
            next_action="Find/train compatible domain adapters and run a >=30 item public benchmark comparison.",
        )
    )

    largest = _largest_concurrency_run(runs)
    production_scale = bool(largest and int(largest.get("request_count", 0)) >= 1000 and int(largest.get("error_count", 0)) == 0)
    gaps.append(
        ProofGap(
            claim="Production-scale concurrency and capacity are characterized.",
            status="proven" if production_scale else "external_required",
            evidence=[str(largest.get("path"))] if largest else [],
            remaining="Current concurrency runs are small hosted/local smoke tests.",
            next_action="Run >=1000 requests with saturation curves on controlled GPU hardware.",
        )
    )

    gaps.append(
        ProofGap(
            claim="vLLM-Metal supports upstream-style CPU LoRA cache tiering.",
            status="blocked_upstream",
            evidence=["docs/THESIS_PROGRESS.md", "runs/kaggle-vllm-startup-v0-xformers.log"] if cuda_cache else ["docs/THESIS_PROGRESS.md"],
            remaining="vLLM-Metal raises NotImplementedError for max_cpu_loras > max_loras.",
            next_action="Wait for upstream support, contribute implementation, or document a Metal-specific design alternative.",
        )
    )

    status_order = {"proven": 1.0, "proven_bounded": 0.75, "missing_local_artifact": 0.25, "missing_live_artifact": 0.25, "external_required": 0.0, "blocked_upstream": 0.0}
    score = sum(status_order.get(gap.status, 0.0) for gap in gaps) / len(gaps)
    data = {
        "artifact_family": "proof-gap-ledger",
        "runs_path": str(runs),
        "completion_score": round(score, 4),
        "gap_count": len(gaps),
        "open_gap_count": sum(1 for gap in gaps if gap.status not in {"proven", "proven_bounded"}),
        "external_required_count": sum(1 for gap in gaps if gap.status == "external_required"),
        "blocked_upstream_count": sum(1 for gap in gaps if gap.status == "blocked_upstream"),
        "gaps": [gap.__dict__ for gap in gaps],
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data
