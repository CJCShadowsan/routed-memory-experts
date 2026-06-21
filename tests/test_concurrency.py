from routed_memory_experts.concurrency import ConcurrencyRecord, ConcurrencySummary, concurrency_summary_to_dict


def test_concurrency_summary_serializes_records():
    record = ConcurrencyRecord("x", "model", True, True, 12.0, None)
    summary = ConcurrencySummary("http://local/v1", "model", 1, 1, 1, 0, 1, 1.0, 2.0, 12.0, 12.0, 12.0, [record])
    data = concurrency_summary_to_dict(summary)
    assert data["success_count"] == 1
    assert data["records"][0]["correct"] is True
