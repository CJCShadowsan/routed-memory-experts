from routed_memory_experts.expert_store import ExpertStore
from routed_memory_experts.models import Tier

def test_expert_store_loads_cold_then_hot():
    store = ExpertStore("experts", hot_capacity=2, warm_capacity=2)
    expert, tier = store.get("kubernetes")
    assert expert.domain == "kubernetes"
    assert tier == Tier.COLD
    expert, tier = store.get("kubernetes")
    assert tier == Tier.HOT
    assert store.cold_loads == 1
    assert store.hot_hits == 1

def test_expert_store_demotes_to_warm():
    store = ExpertStore("experts", hot_capacity=1, warm_capacity=2)
    store.get("kubernetes")
    store.get("python")
    assert "kubernetes" in store.warm
    expert, tier = store.get("kubernetes")
    assert tier == Tier.WARM
    assert store.warm_hits == 1
