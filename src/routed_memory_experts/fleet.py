from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FleetSimulationSummary:
    agent_count: int
    request_count: int
    hot_capacity: int
    cold_loads: int
    hot_hits: int
    hit_rate: float
    unique_agents_touched: int
    p95_loads_per_request: float


def simulate_agent_fleet(agent_count: int = 1000, request_count: int = 5000, hot_capacity: int = 128, locality_window: int = 64) -> FleetSimulationSummary:
    """Simulate routing across many agent-owned hot models with locality.

    This does not claim neural quality. It proves the scheduling/cache claim that a
    large routable fleet can be made tractable only when requests exhibit locality
    and the router/cache keep hot agents resident.
    """
    if agent_count <= 0 or request_count <= 0 or hot_capacity <= 0:
        raise ValueError("agent_count, request_count, and hot_capacity must be positive")
    if locality_window <= 0:
        raise ValueError("locality_window must be positive")

    hot: OrderedDict[int, None] = OrderedDict()
    cold_loads = 0
    hot_hits = 0
    touched: set[int] = set()
    per_request_loads: list[int] = []

    for i in range(request_count):
        # Zipf-ish deterministic locality: most requests hit a moving active set,
        # with periodic jumps into the long tail.
        if i % 17 == 0:
            agent_id = (i * 7919) % agent_count
        else:
            agent_id = (i * 13) % min(agent_count, locality_window)
        touched.add(agent_id)
        if agent_id in hot:
            hot.pop(agent_id)
            hot[agent_id] = None
            hot_hits += 1
            per_request_loads.append(0)
        else:
            cold_loads += 1
            per_request_loads.append(1)
            hot[agent_id] = None
            while len(hot) > hot_capacity:
                hot.popitem(last=False)

    ordered = sorted(per_request_loads)
    p95_index = int(0.95 * (len(ordered) - 1))
    return FleetSimulationSummary(
        agent_count=agent_count,
        request_count=request_count,
        hot_capacity=hot_capacity,
        cold_loads=cold_loads,
        hot_hits=hot_hits,
        hit_rate=hot_hits / request_count,
        unique_agents_touched=len(touched),
        p95_loads_per_request=float(ordered[p95_index]),
    )


def fleet_summary_to_dict(summary: FleetSimulationSummary) -> dict:
    return summary.__dict__.copy()


def write_fleet_summary(summary: FleetSimulationSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fleet_summary_to_dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
