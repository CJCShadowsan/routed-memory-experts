from __future__ import annotations

import argparse
import json

from .adapter_manifest import load_adapter_manifest, manifest_to_dict
from .artifact_validation import validate_artifacts, validation_results_to_dict
from .concurrency import benchmark_openai_concurrency, concurrency_summary_to_dict
from .fleet import fleet_summary_to_dict, simulate_agent_fleet, write_fleet_summary
from .learned_router import compare_routers, router_comparison_to_dict
from .ollama_backend import ollama_summary_to_dict, run_ollama_proof
from .openai_backend import (
    compare_openai_models,
    openai_comparison_summary_to_dict,
    openai_summary_to_dict,
    run_openai_compatible_proof,
)
from .proof import run_proof, summary_to_dict
from .runtime_readiness import check_runtime_readiness, runtime_readiness_to_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rme", description="Routed Memory Experts proof harness")
    sub = parser.add_subparsers(dest="command", required=True)

    prove = sub.add_parser("prove", help="run end-to-end routed expert proof")
    prove.add_argument("--workload", required=True)
    prove.add_argument("--experts", required=True)
    prove.add_argument("--output", default="runs/proof.json")
    prove.add_argument("--min-accuracy", type=float, default=0.80)
    prove.add_argument("--max-regret", type=float, default=0.20)

    fleet = sub.add_parser("simulate-fleet", help="simulate routing across many agent-owned hot models")
    fleet.add_argument("--agents", type=int, default=1000)
    fleet.add_argument("--requests", type=int, default=5000)
    fleet.add_argument("--hot-capacity", type=int, default=128)
    fleet.add_argument("--locality-window", type=int, default=64)
    fleet.add_argument("--min-hit-rate", type=float, default=0.85)
    fleet.add_argument("--output", default="runs/fleet.json")

    ollama = sub.add_parser("prove-ollama", help="run routed expert proof through a real local Ollama model")
    ollama.add_argument("--model", required=True)
    ollama.add_argument("--workload", required=True)
    ollama.add_argument("--experts", required=True)
    ollama.add_argument("--output", default="runs/ollama-proof.json")
    ollama.add_argument("--limit", type=int, default=None)
    ollama.add_argument("--min-accuracy", type=float, default=0.75)

    openai = sub.add_parser("prove-openai", help="run routed expert proof through an OpenAI-compatible server")
    openai.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    openai.add_argument("--model", required=True)
    openai.add_argument("--workload", required=True)
    openai.add_argument("--experts", required=True)
    openai.add_argument("--output", default="runs/openai-proof.json")
    openai.add_argument("--limit", type=int, default=None)
    openai.add_argument("--min-accuracy", type=float, default=0.75)
    openai.add_argument("--adapter-manifest", default=None)

    compare_openai = sub.add_parser("compare-openai-models", help="compare base and expert models on the same routed prompts")
    compare_openai.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    compare_openai.add_argument("--base-model", required=True)
    compare_openai.add_argument("--expert-model", required=True)
    compare_openai.add_argument("--workload", required=True)
    compare_openai.add_argument("--experts", required=True)
    compare_openai.add_argument("--output", default="runs/openai-model-comparison.json")
    compare_openai.add_argument("--limit", type=int, default=None)

    concurrency = sub.add_parser("benchmark-openai-concurrency", help="measure concurrent OpenAI-compatible routed serving")
    concurrency.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    concurrency.add_argument("--model", required=True)
    concurrency.add_argument("--workload", required=True)
    concurrency.add_argument("--experts", required=True)
    concurrency.add_argument("--output", default="runs/openai-concurrency.json")
    concurrency.add_argument("--requests", type=int, default=12)
    concurrency.add_argument("--concurrency", type=int, default=3)
    concurrency.add_argument("--adapter-manifest", default=None)

    manifest = sub.add_parser("inspect-adapter-manifest", help="show adapter manifest route map")
    manifest.add_argument("--manifest", required=True)

    validate = sub.add_parser("validate-artifacts", help="validate proof artifact JSON files")
    validate.add_argument("--path", default="runs")

    routers = sub.add_parser("compare-routers", help="compare learned router against keyword baseline")
    routers.add_argument("--train", required=True)
    routers.add_argument("--dev", required=True)
    routers.add_argument("--output", default="runs/router-comparison.json")
    routers.add_argument("--min-learned-accuracy", type=float, default=0.80)

    runtime = sub.add_parser("check-runtimes", help="check local serving runtime readiness")
    runtime.add_argument("--output", default="runs/runtime-readiness.json")

    args = parser.parse_args(argv)

    if args.command == "prove":
        summary = run_proof(args.workload, args.experts, args.output)
        data = summary_to_dict(summary)
        print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2, sort_keys=True))
        if summary.routed_accuracy < args.min_accuracy:
            print(f"FAIL: routed accuracy {summary.routed_accuracy:.3f} < {args.min_accuracy:.3f}")
            return 2
        if summary.route_regret_rate > args.max_regret:
            print(f"FAIL: route regret {summary.route_regret_rate:.3f} > {args.max_regret:.3f}")
            return 3
        if summary.routed_accuracy <= summary.baseline_accuracy:
            print("FAIL: routed system did not outperform generalist baseline")
            return 4
        print("PASS: routed memory expert proof met thresholds")
        return 0

    if args.command == "simulate-fleet":
        summary = simulate_agent_fleet(args.agents, args.requests, args.hot_capacity, args.locality_window)
        write_fleet_summary(summary, args.output)
        data = fleet_summary_to_dict(summary)
        print(json.dumps(data, indent=2, sort_keys=True))
        if summary.hit_rate < args.min_hit_rate:
            print(f"FAIL: fleet hit rate {summary.hit_rate:.3f} < {args.min_hit_rate:.3f}")
            return 5
        print("PASS: fleet simulation met locality/cache threshold")
        return 0

    if args.command == "prove-ollama":
        summary = run_ollama_proof(args.workload, args.experts, args.model, args.output, args.limit)
        data = ollama_summary_to_dict(summary)
        print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2, sort_keys=True))
        if summary.accuracy < args.min_accuracy:
            print(f"FAIL: Ollama routed accuracy {summary.accuracy:.3f} < {args.min_accuracy:.3f}")
            return 6
        print("PASS: Ollama routed expert proof met threshold")
        return 0

    if args.command == "prove-openai":
        summary = run_openai_compatible_proof(args.workload, args.experts, args.base_url, args.model, args.output, args.limit, args.adapter_manifest)
        data = openai_summary_to_dict(summary)
        print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2, sort_keys=True))
        if summary.accuracy < args.min_accuracy:
            print(f"FAIL: OpenAI-compatible routed accuracy {summary.accuracy:.3f} < {args.min_accuracy:.3f}")
            return 10
        print("PASS: OpenAI-compatible routed expert proof met threshold")
        return 0

    if args.command == "compare-openai-models":
        summary = compare_openai_models(args.workload, args.experts, args.base_url, args.base_model, args.expert_model, args.output, args.limit)
        data = openai_comparison_summary_to_dict(summary)
        print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2, sort_keys=True))
        print("PASS: OpenAI-compatible model comparison completed")
        return 0

    if args.command == "benchmark-openai-concurrency":
        summary = benchmark_openai_concurrency(args.workload, args.experts, args.base_url, args.model, args.requests, args.concurrency, args.output, args.adapter_manifest)
        data = concurrency_summary_to_dict(summary)
        print(json.dumps({k: v for k, v in data.items() if k != "records"}, indent=2, sort_keys=True))
        if summary.error_count:
            print(f"FAIL: concurrent benchmark had {summary.error_count} errors")
            return 11
        print("PASS: OpenAI-compatible concurrency benchmark completed")
        return 0

    if args.command == "inspect-adapter-manifest":
        manifest = load_adapter_manifest(args.manifest)
        print(json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True))
        return 0

    if args.command == "validate-artifacts":
        results = validate_artifacts(args.path)
        data = validation_results_to_dict(results)
        print(json.dumps(data, indent=2, sort_keys=True))
        if not data["valid"]:
            return 12
        print("PASS: proof artifacts match required schemas")
        return 0

    if args.command == "compare-routers":
        summary = compare_routers(args.train, args.dev, args.output)
        data = router_comparison_to_dict(summary)
        print(json.dumps(data, indent=2, sort_keys=True))
        if summary.learned_accuracy < args.min_learned_accuracy:
            print(f"FAIL: learned router accuracy {summary.learned_accuracy:.3f} < {args.min_learned_accuracy:.3f}")
            return 7
        if not summary.learned_beats_keyword:
            print("FAIL: learned router did not beat keyword router")
            return 8
        print("PASS: learned router beat keyword baseline")
        return 0

    if args.command == "check-runtimes":
        readiness = check_runtime_readiness(args.output)
        data = runtime_readiness_to_dict(readiness)
        print(json.dumps(data, indent=2, sort_keys=True))
        if not readiness.production_adapter_runtime_ready:
            print("BLOCKED: production adapter runtime is not ready on this host")
            return 9
        print("PASS: production adapter runtime is ready")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
