from routed_memory_experts.router import KeywordRouter

def test_router_selects_kubernetes_for_helm():
    domain, confidence, reason = KeywordRouter().route("Kubernetes Helm deployment readiness")
    assert domain == "kubernetes"
    assert confidence > 0.5
    assert "helm" in reason.lower()

def test_router_falls_back_for_unknown_domain():
    domain, confidence, reason = KeywordRouter().route("Tell me something poetic about rain")
    assert domain == "general"
    assert confidence < 0.4
