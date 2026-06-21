from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .adapter_manifest import load_adapter_manifest
from .expert_store import ExpertStore
from .openai_backend import OpenAICompatibleClient, _prepare_item, _resolve_model, percentile
from .router import KeywordRouter
from .workload import is_correct, load_workload


@dataclass
class ConcurrencyRecord:
    item_id: str
    model: str
    success: bool
    correct: bool
    latency_ms: float
    error: str | None


@dataclass
class ConcurrencySummary:
    base_url: str
    model: str
    request_count: int
    concurrency: int
    success_count: int
    error_count: int
    correct_count: int
    accuracy: float
    throughput_requests_per_second: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    records: list[ConcurrencyRecord]


def benchmark_openai_concurrency(
    workload_path: str | Path,
    expert_dir: str | Path,
    base_url: str,
    model: str,
    request_count: int,
    concurrency: int,
    output_path: str | Path | None = None,
    adapter_manifest_path: str | Path | None = None,
) -> ConcurrencySummary:
    items = load_workload(workload_path)
    router = KeywordRouter()
    store = ExpertStore(expert_dir)
    manifest = load_adapter_manifest(adapter_manifest_path) if adapter_manifest_path else None
    prepared = []
    for index in range(request_count):
        item = items[index % len(items)]
        domain, _expert_name, prompt = _prepare_item(item, router, store)
        request_model = _resolve_model(model, domain, manifest)
        prepared.append((item, request_model, prompt))

    def one(item_model_prompt: tuple) -> ConcurrencyRecord:
        item, request_model, prompt = item_model_prompt
        client = OpenAICompatibleClient(base_url, request_model)
        start = time.perf_counter()
        try:
            response = client.chat(prompt)
            latency = (time.perf_counter() - start) * 1000
            return ConcurrencyRecord(item.id, request_model, True, is_correct(response, item.expected_contains), latency, None)
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            return ConcurrencyRecord(item.id, request_model, False, False, latency, f"{type(exc).__name__}: {exc}")

    started = time.perf_counter()
    records: list[ConcurrencyRecord] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in as_completed([pool.submit(one, item) for item in prepared]):
            records.append(future.result())
    elapsed = max(time.perf_counter() - started, 1e-9)
    successes = [record for record in records if record.success]
    latencies = [record.latency_ms for record in successes]
    correct = sum(1 for record in successes if record.correct)
    summary = ConcurrencySummary(
        base_url=base_url,
        model=model,
        request_count=request_count,
        concurrency=concurrency,
        success_count=len(successes),
        error_count=request_count - len(successes),
        correct_count=correct,
        accuracy=correct / len(successes) if successes else 0.0,
        throughput_requests_per_second=request_count / elapsed,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        p99_latency_ms=percentile(latencies, 0.99),
        records=records,
    )
    if output_path:
        write_concurrency_summary(summary, output_path)
    return summary


def concurrency_summary_to_dict(summary: ConcurrencySummary) -> dict:
    return {
        "base_url": summary.base_url,
        "model": summary.model,
        "request_count": summary.request_count,
        "concurrency": summary.concurrency,
        "success_count": summary.success_count,
        "error_count": summary.error_count,
        "correct_count": summary.correct_count,
        "accuracy": summary.accuracy,
        "throughput_requests_per_second": summary.throughput_requests_per_second,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "p99_latency_ms": summary.p99_latency_ms,
        "records": [record.__dict__ for record in summary.records],
    }


def write_concurrency_summary(summary: ConcurrencySummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(concurrency_summary_to_dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
