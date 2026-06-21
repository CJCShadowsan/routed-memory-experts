from routed_memory_experts.openai_backend import OpenAIModelComparisonRecord, OpenAIModelComparisonSummary, openai_comparison_summary_to_dict, percentile


def test_percentile_uses_ordered_rank():
    assert percentile([30, 10, 20], 0.5) == 20
    assert percentile([], 0.95) == 0.0


def test_openai_comparison_summary_serializes_outcomes():
    record = OpenAIModelComparisonRecord(
        item_id="x",
        oracle_domain="python",
        routed_domain="python",
        base_model="base",
        expert_model="expert",
        base_correct=False,
        expert_correct=True,
        outcome="expert_win",
        base_latency_ms=10.0,
        expert_latency_ms=8.0,
        base_response="no",
        expert_response="yes",
    )
    summary = OpenAIModelComparisonSummary("http://local/v1", "base", "expert", 1, 0, 1, 0.0, 1.0, 1, 0, 0, 10.0, 10.0, 8.0, 8.0, [record])
    data = openai_comparison_summary_to_dict(summary)
    assert data["expert_wins"] == 1
    assert data["records"][0]["outcome"] == "expert_win"
