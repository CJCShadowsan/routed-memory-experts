from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

class Tier(str, Enum):
    HOT = "hot_hbm"
    WARM = "warm_dram"
    COLD = "cold_nvme"
    FALLBACK = "fallback_generalist"

@dataclass(frozen=True)
class WorkloadItem:
    id: str
    domain: str
    prompt: str
    expected_contains: List[str]
    risk: str = "normal"

@dataclass
class Expert:
    name: str
    domain: str
    facts: Dict[str, str]
    default_response: str
    tier_loaded_from: Tier = Tier.COLD
    def answer(self, prompt: str) -> str:
        lower = prompt.lower()
        for keyword, response in self.facts.items():
            if keyword.lower() in lower:
                return response
        return self.default_response

@dataclass
class ProofRecord:
    item_id: str
    oracle_domain: str
    routed_domain: str
    expert_name: str
    tier: str
    confidence: float
    correct: bool
    baseline_correct: bool
    route_regret: bool
    response: str
    baseline_response: str

@dataclass
class ProofSummary:
    workload_count: int
    routed_correct: int
    baseline_correct: int
    route_regret_count: int
    cold_loads: int
    warm_hits: int
    hot_hits: int
    fallbacks: int
    records: List[ProofRecord] = field(default_factory=list)
    @property
    def routed_accuracy(self) -> float:
        return self.routed_correct / self.workload_count if self.workload_count else 0.0
    @property
    def baseline_accuracy(self) -> float:
        return self.baseline_correct / self.workload_count if self.workload_count else 0.0
    @property
    def route_regret_rate(self) -> float:
        return self.route_regret_count / self.workload_count if self.workload_count else 0.0
