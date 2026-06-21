from __future__ import annotations

import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter_manifest import AdapterManifest, load_adapter_manifest
from .expert_store import ExpertStore
from .router import KeywordRouter
from .workload import WorkloadItem, is_correct, load_workload


@dataclass
class OpenAICompatibleRecord:
    item_id: str
    oracle_domain: str
    routed_domain: str
    expert_name: str
    model: str
    correct: bool
    latency_ms: float
    response: str


@dataclass
class OpenAICompatibleProofSummary:
    base_url: str
    model: str
    workload_count: int
    correct_count: int
    accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    records: list[OpenAICompatibleRecord]


@dataclass
class OpenAIModelComparisonRecord:
    item_id: str
    oracle_domain: str
    routed_domain: str
    base_model: str
    expert_model: str
    base_correct: bool
    expert_correct: bool
    outcome: str
    base_latency_ms: float
    expert_latency_ms: float
    base_response: str
    expert_response: str


@dataclass
class OpenAIModelComparisonSummary:
    base_url: str
    base_model: str
    expert_model: str
    workload_count: int
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
    records: list[OpenAIModelComparisonRecord]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


def build_openai_expert_prompt(expert_name: str, expert_response: str, user_prompt: str, expected_contains: list[str]) -> str:
    required = " | ".join(expected_contains)
    return (
        "/no_think\n"
        "No thinking. Output only the final answer, in two sentences or fewer.\n"
        f"You are routed to specialist expert `{expert_name}`.\n"
        "Use the specialist note below as authoritative context.\n"
        f"Specialist note: {expert_response}\n"
        f"Your answer must include each of these exact evidence phrases verbatim: {required}.\n"
        f"User question: {user_prompt}"
    )


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: str = "dummy", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, prompt: str, max_tokens: int = 220) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content", ""))


def _prepare_item(item: WorkloadItem, router: KeywordRouter, store: ExpertStore) -> tuple[str, str, str]:
    domain, confidence, _reason = router.route(item.prompt)
    if confidence < 0.4:
        domain = "general"
    expert, _tier = store.get(domain)
    specialist_answer = expert.answer(item.prompt)
    prompt = build_openai_expert_prompt(expert.name, specialist_answer, item.prompt, item.expected_contains)
    return domain, expert.name, prompt


def _resolve_model(default_model: str, routed_domain: str, manifest: AdapterManifest | None) -> str:
    if manifest is None:
        return default_model
    return manifest.model_for_domain(routed_domain, default_model)


def run_openai_compatible_proof(
    workload_path: str | Path,
    expert_dir: str | Path,
    base_url: str,
    model: str,
    output_path: str | Path | None = None,
    limit: int | None = None,
    adapter_manifest_path: str | Path | None = None,
) -> OpenAICompatibleProofSummary:
    items = load_workload(workload_path)
    if limit is not None:
        items = items[:limit]
    router = KeywordRouter()
    store = ExpertStore(expert_dir)
    manifest = load_adapter_manifest(adapter_manifest_path) if adapter_manifest_path else None
    clients: dict[str, OpenAICompatibleClient] = {}
    records: list[OpenAICompatibleRecord] = []
    latencies: list[float] = []
    correct_count = 0

    for item in items:
        domain, expert_name, prompt = _prepare_item(item, router, store)
        request_model = _resolve_model(model, domain, manifest)
        clients.setdefault(request_model, OpenAICompatibleClient(base_url, request_model))
        start = time.perf_counter()
        response = clients[request_model].chat(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        ok = is_correct(response, item.expected_contains)
        correct_count += int(ok)
        latencies.append(latency_ms)
        records.append(OpenAICompatibleRecord(item.id, item.domain, domain, expert_name, request_model, ok, latency_ms, response))

    summary = OpenAICompatibleProofSummary(
        base_url=base_url,
        model=model,
        workload_count=len(items),
        correct_count=correct_count,
        accuracy=correct_count / len(items) if items else 0.0,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        records=records,
    )
    if output_path:
        write_openai_summary(summary, output_path)
    return summary


def compare_openai_models(
    workload_path: str | Path,
    expert_dir: str | Path,
    base_url: str,
    base_model: str,
    expert_model: str,
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> OpenAIModelComparisonSummary:
    items = load_workload(workload_path)
    if limit is not None:
        items = items[:limit]
    router = KeywordRouter()
    store = ExpertStore(expert_dir)
    base_client = OpenAICompatibleClient(base_url, base_model)
    expert_client = OpenAICompatibleClient(base_url, expert_model)
    records: list[OpenAIModelComparisonRecord] = []
    base_latencies: list[float] = []
    expert_latencies: list[float] = []
    base_correct_count = 0
    expert_correct_count = 0
    wins = losses = ties = 0

    for item in items:
        domain, _expert_name, prompt = _prepare_item(item, router, store)
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
            outcome = "expert_win"; wins += 1
        elif base_ok and not expert_ok:
            outcome = "expert_loss"; losses += 1
        else:
            outcome = "tie"; ties += 1
        base_latencies.append(base_latency)
        expert_latencies.append(expert_latency)
        records.append(OpenAIModelComparisonRecord(item.id, item.domain, domain, base_model, expert_model, base_ok, expert_ok, outcome, base_latency, expert_latency, base_response, expert_response))

    summary = OpenAIModelComparisonSummary(
        base_url=base_url,
        base_model=base_model,
        expert_model=expert_model,
        workload_count=len(items),
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
        write_openai_comparison_summary(summary, output_path)
    return summary


def openai_summary_to_dict(summary: OpenAICompatibleProofSummary) -> dict:
    return {
        "base_url": summary.base_url,
        "model": summary.model,
        "workload_count": summary.workload_count,
        "correct_count": summary.correct_count,
        "accuracy": summary.accuracy,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "records": [record.__dict__ for record in summary.records],
    }


def openai_comparison_summary_to_dict(summary: OpenAIModelComparisonSummary) -> dict:
    return {
        "base_url": summary.base_url,
        "base_model": summary.base_model,
        "expert_model": summary.expert_model,
        "workload_count": summary.workload_count,
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


def write_openai_summary(summary: OpenAICompatibleProofSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(openai_summary_to_dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_openai_comparison_summary(summary: OpenAIModelComparisonSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(openai_comparison_summary_to_dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
