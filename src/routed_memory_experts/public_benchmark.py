from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .expert_store import ExpertStore
from .openai_backend import OpenAICompatibleClient, percentile
from .router import KeywordRouter
from .workload import is_correct, load_workload


@dataclass
class PublicBenchmarkRecord:
    item_id: str
    source: str
    split: str
    oracle_domain: str
    routed_domain: str
    base_model: str
    expert_model: str
    base_correct: bool
    expert_correct: bool
    outcome: str
    expected_contains: list[str]
    base_latency_ms: float
    expert_latency_ms: float
    base_response: str
    expert_response: str


@dataclass
class PublicBenchmarkSummary:
    artifact_family: str
    base_url: str
    benchmark: str
    workload_count: int
    base_model: str
    expert_model: str
    base_correct_count: int
    expert_correct_count: int
    base_accuracy: float
    expert_accuracy: float
    expert_wins: int
    expert_losses: int
    ties: int
    base_p50_latency_ms: float
    base_p95_latency_ms: float
    expert_p50_latency_ms: float
    expert_p95_latency_ms: float
    records: list[PublicBenchmarkRecord]


def build_public_benchmark_prompt(user_prompt: str, routed_domain: str, specialist_note: str) -> str:
    """Build a benchmark prompt without leaking labels or expected answers."""
    return (
        "/no_think\n"
        "No thinking. Output only the final answer. For math, include the final numeric answer.\n"
        f"Routed domain: {routed_domain}.\n"
        "Use this domain guidance if helpful, but do not mention it unless needed.\n"
        f"Domain guidance: {specialist_note}\n"
        f"Question: {user_prompt}"
    )


def run_public_openai_benchmark(
    workload_path: str | Path,
    expert_dir: str | Path,
    base_url: str,
    base_model: str,
    expert_model: str,
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> PublicBenchmarkSummary:
    items = load_workload(workload_path)
    if limit is not None:
        items = items[:limit]

    router = KeywordRouter()
    store = ExpertStore(expert_dir)
    base_client = OpenAICompatibleClient(base_url, base_model)
    expert_client = OpenAICompatibleClient(base_url, expert_model)

    records: list[PublicBenchmarkRecord] = []
    base_latencies: list[float] = []
    expert_latencies: list[float] = []
    base_correct_count = 0
    expert_correct_count = 0
    wins = losses = ties = 0

    for item in items:
        routed_domain, confidence, _reason = router.route(item.prompt)
        if confidence < 0.4:
            routed_domain = "general"
        expert, _tier = store.get(routed_domain)
        prompt = build_public_benchmark_prompt(item.prompt, routed_domain, expert.default_response)

        start = time.perf_counter()
        base_response = base_client.chat(prompt)
        base_latency = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        expert_response = expert_client.chat(prompt)
        expert_latency = (time.perf_counter() - start) * 1000

        base_ok = is_correct(base_response, item.expected_contains)
        expert_ok = is_correct(expert_response, item.expected_contains)
        base_correct_count += int(base_ok)
        expert_correct_count += int(expert_ok)
        if expert_ok and not base_ok:
            outcome = "expert_win"
            wins += 1
        elif base_ok and not expert_ok:
            outcome = "expert_loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1

        base_latencies.append(base_latency)
        expert_latencies.append(expert_latency)
        records.append(
            PublicBenchmarkRecord(
                item_id=item.id,
                source=item.source,
                split=item.split,
                oracle_domain=item.domain,
                routed_domain=routed_domain,
                base_model=base_model,
                expert_model=expert_model,
                base_correct=base_ok,
                expert_correct=expert_ok,
                outcome=outcome,
                expected_contains=item.expected_contains,
                base_latency_ms=base_latency,
                expert_latency_ms=expert_latency,
                base_response=base_response,
                expert_response=expert_response,
            )
        )

    summary = PublicBenchmarkSummary(
        artifact_family="public-openai-benchmark",
        base_url=base_url,
        benchmark=str(workload_path),
        workload_count=len(items),
        base_model=base_model,
        expert_model=expert_model,
        base_correct_count=base_correct_count,
        expert_correct_count=expert_correct_count,
        base_accuracy=base_correct_count / len(items) if items else 0.0,
        expert_accuracy=expert_correct_count / len(items) if items else 0.0,
        expert_wins=wins,
        expert_losses=losses,
        ties=ties,
        base_p50_latency_ms=percentile(base_latencies, 0.50),
        base_p95_latency_ms=percentile(base_latencies, 0.95),
        expert_p50_latency_ms=percentile(expert_latencies, 0.50),
        expert_p95_latency_ms=percentile(expert_latencies, 0.95),
        records=records,
    )
    if output_path:
        write_public_benchmark_summary(summary, output_path)
    return summary


def public_benchmark_summary_to_dict(summary: PublicBenchmarkSummary) -> dict:
    return {
        "artifact_family": summary.artifact_family,
        "base_url": summary.base_url,
        "benchmark": summary.benchmark,
        "workload_count": summary.workload_count,
        "base_model": summary.base_model,
        "expert_model": summary.expert_model,
        "base_correct_count": summary.base_correct_count,
        "expert_correct_count": summary.expert_correct_count,
        "base_accuracy": summary.base_accuracy,
        "expert_accuracy": summary.expert_accuracy,
        "expert_wins": summary.expert_wins,
        "expert_losses": summary.expert_losses,
        "ties": summary.ties,
        "base_p50_latency_ms": summary.base_p50_latency_ms,
        "base_p95_latency_ms": summary.base_p95_latency_ms,
        "expert_p50_latency_ms": summary.expert_p50_latency_ms,
        "expert_p95_latency_ms": summary.expert_p95_latency_ms,
        "records": [record.__dict__ for record in summary.records],
    }


def write_public_benchmark_summary(summary: PublicBenchmarkSummary, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(public_benchmark_summary_to_dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
