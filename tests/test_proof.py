from routed_memory_experts.proof import run_proof

def test_end_to_end_proof_outperforms_generalist():
    summary = run_proof("workloads/real_world_v1.jsonl", "experts")
    assert summary.workload_count >= 10
    assert summary.routed_accuracy >= 0.80
    assert summary.routed_accuracy > summary.baseline_accuracy
    assert summary.route_regret_rate <= 0.20
    assert summary.cold_loads >= 2

def test_end_to_end_proof_records_tiers():
    summary = run_proof("workloads/real_world_v1.jsonl", "experts")
    tiers = {r.tier for r in summary.records}
    assert "cold_nvme" in tiers
    assert "hot_hbm" in tiers or summary.hot_hits > 0
