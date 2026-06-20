from __future__ import annotations

import argparse
import json

from .fleet import fleet_summary_to_dict, simulate_agent_fleet, write_fleet_summary
from .proof import run_proof, summary_to_dict


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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
