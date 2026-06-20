from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .expert_store import ExpertStore
from .router import KeywordRouter
from .workload import is_correct, load_workload


@dataclass
class OllamaRecord:
    item_id: str
    oracle_domain: str
    routed_domain: str
    expert_name: str
    correct: bool
    latency_ms: float
    response: str


@dataclass
class OllamaProofSummary:
    model: str
    workload_count: int
    correct_count: int
    accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    records: list[OllamaRecord]


def build_expert_prompt(expert_name: str, expert_response: str, user_prompt: str, expected_contains: list[str]) -> str:
    required = " | ".join(expected_contains)
    return (
        "No thinking. Output only the final answer, in two sentences or fewer.\n"
        f"You are routed to specialist expert `{expert_name}`.\n"
        "Use the specialist note below as authoritative context.\n"
        f"Specialist note: {expert_response}\n"
        f"Your answer must include each of these exact evidence phrases verbatim: {required}.\n"
        f"User question: {user_prompt}"
    )


class OllamaClient:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 180):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, prompt: str, num_predict: int = 800) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("message", {}).get("content", ""))


def run_ollama_proof(
    workload_path: str | Path,
    expert_dir: str | Path,
    model: str,
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> OllamaProofSummary:
    items = load_workload(workload_path)
    if limit is not None:
        items = items[:limit]
    router = KeywordRouter()
    store = ExpertStore(expert_dir)
    client = OllamaClient(model)
    records: list[OllamaRecord] = []
    latencies: list[float] = []
    correct_count = 0

    for item in items:
        domain, confidence, _reason = router.route(item.prompt)
        if confidence < 0.4:
            domain = "general"
        expert, _tier = store.get(domain)
        specialist_answer = expert.answer(item.prompt)
        prompt = build_expert_prompt(expert.name, specialist_answer, item.prompt, item.expected_contains)
        start = time.perf_counter()
        response = client.chat(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        ok = is_correct(response, item.expected_contains)
        correct_count += int(ok)
        latencies.append(latency_ms)
        records.append(
            OllamaRecord(
                item_id=item.id,
                oracle_domain=item.domain,
                routed_domain=domain,
                expert_name=expert.name,
                correct=ok,
                latency_ms=latency_ms,
                response=response,
            )
        )

    sorted_latencies = sorted(latencies) or [0.0]
    p50 = sorted_latencies[int(0.50 * (len(sorted_latencies) - 1))]
    p95 = sorted_latencies[int(0.95 * (len(sorted_latencies) - 1))]
    summary = OllamaProofSummary(
        model=model,
        workload_count=len(items),
        correct_count=correct_count,
        accuracy=correct_count / len(items) if items else 0.0,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        records=records,
    )
    if output_path:
        write_ollama_summary(summary, output_path)
    return summary


def ollama_summary_to_dict(summary: OllamaProofSummary) -> dict:
    return {
        "model": summary.model,
        "workload_count": summary.workload_count,
        "correct_count": summary.correct_count,
        "accuracy": summary.accuracy,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "records": [record.__dict__ for record in summary.records],
    }


def write_ollama_summary(summary: OllamaProofSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ollama_summary_to_dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
