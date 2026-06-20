from routed_memory_experts.fleet import simulate_agent_fleet


def test_thousand_agent_fleet_has_cache_locality():
    summary = simulate_agent_fleet(agent_count=1000, request_count=5000, hot_capacity=128, locality_window=64)
    assert summary.agent_count == 1000
    assert summary.unique_agents_touched > 300
    assert summary.hit_rate >= 0.85
    assert summary.p95_loads_per_request <= 1.0


def test_fleet_simulation_rejects_invalid_sizes():
    try:
        simulate_agent_fleet(agent_count=0)
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
