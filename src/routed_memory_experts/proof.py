from __future__ import annotations
import json
from pathlib import Path
from .expert_store import ExpertStore
from .models import ProofRecord, ProofSummary
from .router import KeywordRouter
from .workload import is_correct, load_workload

class RoutedExpertSystem:
    def __init__(self, expert_dir: str|Path, hot_capacity:int=2, warm_capacity:int=4):
        self.router=KeywordRouter(); self.store=ExpertStore(expert_dir, hot_capacity, warm_capacity); self.fallbacks=0
    def answer(self, prompt:str):
        domain,confidence,reason=self.router.route(prompt)
        if confidence<0.4: self.fallbacks+=1; domain="general"
        expert,tier=self.store.get(domain)
        return domain,confidence,reason,expert,tier,expert.answer(prompt)

def run_proof(workload_path: str|Path, expert_dir: str|Path, output_path: str|Path|None=None) -> ProofSummary:
    items=load_workload(workload_path); system=RoutedExpertSystem(expert_dir); baseline=ExpertStore(expert_dir)
    records=[]; routed_correct=baseline_correct=route_regret=0
    for item in items:
        routed_domain,confidence,_reason,expert,tier,response=system.answer(item.prompt)
        general,_=baseline.get("general"); baseline_response=general.answer(item.prompt)
        correct=is_correct(response,item.expected_contains); base_ok=is_correct(baseline_response,item.expected_contains)
        regret=routed_domain!=item.domain and item.domain!="general"
        routed_correct+=int(correct); baseline_correct+=int(base_ok); route_regret+=int(regret)
        records.append(ProofRecord(item.id,item.domain,routed_domain,expert.name,tier.value,confidence,correct,base_ok,regret,response,baseline_response))
    summary=ProofSummary(len(items),routed_correct,baseline_correct,route_regret,system.store.cold_loads,system.store.warm_hits,system.store.hot_hits,system.fallbacks,records)
    if output_path: write_summary(summary, output_path)
    return summary

def summary_to_dict(summary: ProofSummary)->dict:
    return {"workload_count":summary.workload_count,"routed_correct":summary.routed_correct,"baseline_correct":summary.baseline_correct,"routed_accuracy":summary.routed_accuracy,"baseline_accuracy":summary.baseline_accuracy,"route_regret_count":summary.route_regret_count,"route_regret_rate":summary.route_regret_rate,"cold_loads":summary.cold_loads,"warm_hits":summary.warm_hits,"hot_hits":summary.hot_hits,"fallbacks":summary.fallbacks,"records":[r.__dict__ for r in summary.records]}

def write_summary(summary:ProofSummary, output_path:str|Path)->None:
    path=Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(summary_to_dict(summary), indent=2, sort_keys=True)+"\n", encoding="utf-8")
