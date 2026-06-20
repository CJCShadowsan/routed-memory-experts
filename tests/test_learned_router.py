from routed_memory_experts.learned_router import NaiveBayesRouter, compare_routers


def test_learned_router_classifies_training_synonyms():
    router = NaiveBayesRouter()
    router.fit([
        ("chart renders service templates", "kubernetes"),
        ("fixture driven unit checks", "python"),
    ])
    domain, confidence, reason = router.route("chart assertions for service templates")
    assert domain == "kubernetes"
    assert confidence > 0.5
    assert "naive-bayes" in reason


def test_learned_router_beats_keyword_on_synonym_devset():
    summary = compare_routers("workloads/router_train_v1.jsonl", "workloads/router_dev_v1.jsonl")
    assert summary.learned_beats_keyword
    assert summary.learned_accuracy >= 0.80
