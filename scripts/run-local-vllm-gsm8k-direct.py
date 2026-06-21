#!/usr/bin/env python3
"""Run GSM8K public benchmark locally through vLLM's Python API.

This avoids the OpenAI HTTP server path, which can be blocked by local FastAPI /
prometheus middleware version skew, while still exercising local vLLM model and
LoRA loading on the Mac.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from routed_memory_experts.expert_store import ExpertStore
from routed_memory_experts.public_benchmark import build_public_benchmark_prompt
from routed_memory_experts.router import KeywordRouter
from routed_memory_experts.workload import is_correct, load_workload


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def generate_one(llm: LLM, prompt: str, params: SamplingParams, lora_request: LoRARequest | None = None) -> tuple[str, float]:
    start = time.perf_counter()
    outputs = llm.generate([prompt], params, lora_request=lora_request)
    latency_ms = (time.perf_counter() - start) * 1000
    return outputs[0].outputs[0].text, latency_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local direct-vLLM GSM8K public benchmark")
    parser.add_argument("--workload", default="workloads/gsm8k_public_sample.jsonl")
    parser.add_argument("--experts", default="experts")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora")
    parser.add_argument("--expert-model", default="math")
    parser.add_argument("--output", default="runs/local-vllm-gsm8k-public-openai-benchmark.json")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    items = load_workload(args.workload)
    if args.limit is not None:
        items = items[: args.limit]

    router = KeywordRouter()
    store = ExpertStore(args.experts)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    llm = LLM(model=args.base_model, max_model_len=args.max_model_len, enable_lora=True, max_lora_rank=64, max_loras=1)
    lora_request = LoRARequest(args.expert_model, 1, args.adapter)

    records = []
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
        base_response, base_latency = generate_one(llm, prompt, params)
        expert_response, expert_latency = generate_one(llm, prompt, params, lora_request=lora_request)

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
            {
                "item_id": item.id,
                "source": item.source,
                "split": item.split,
                "oracle_domain": item.domain,
                "routed_domain": routed_domain,
                "base_model": args.base_model,
                "expert_model": args.expert_model,
                "base_correct": base_ok,
                "expert_correct": expert_ok,
                "outcome": outcome,
                "expected_contains": item.expected_contains,
                "base_latency_ms": base_latency,
                "expert_latency_ms": expert_latency,
                "base_response": base_response,
                "expert_response": expert_response,
            }
        )

    data = {
        "artifact_family": "public-openai-benchmark",
        "base_url": "local-vllm-python-api",
        "benchmark": args.workload,
        "workload_count": len(items),
        "base_model": args.base_model,
        "expert_model": args.expert_model,
        "adapter": args.adapter,
        "base_correct_count": base_correct_count,
        "expert_correct_count": expert_correct_count,
        "base_accuracy": base_correct_count / len(items) if items else 0.0,
        "expert_accuracy": expert_correct_count / len(items) if items else 0.0,
        "expert_wins": wins,
        "expert_losses": losses,
        "ties": ties,
        "base_p50_latency_ms": percentile(base_latencies, 0.5),
        "base_p95_latency_ms": percentile(base_latencies, 0.95),
        "expert_p50_latency_ms": percentile(expert_latencies, 0.5),
        "expert_p95_latency_ms": percentile(expert_latencies, 0.95),
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
