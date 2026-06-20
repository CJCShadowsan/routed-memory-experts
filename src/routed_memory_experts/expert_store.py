from __future__ import annotations
import json
from collections import OrderedDict
from pathlib import Path
from .models import Expert, Tier

class ExpertStore:
    """Three-tier expert cache: hot/HBM, warm/DRAM, cold/NVMe-like disk."""
    def __init__(self, expert_dir: str|Path, hot_capacity:int=2, warm_capacity:int=4):
        self.expert_dir=Path(expert_dir); self.hot_capacity=hot_capacity; self.warm_capacity=warm_capacity
        self.hot:OrderedDict[str,Expert]=OrderedDict(); self.warm:OrderedDict[str,Expert]=OrderedDict()
        self.cold_loads=0; self.warm_hits=0; self.hot_hits=0
    def get(self, domain: str) -> tuple[Expert,Tier]:
        if domain in self.hot:
            expert=self.hot.pop(domain); self.hot[domain]=expert; self.hot_hits+=1; expert.tier_loaded_from=Tier.HOT; return expert,Tier.HOT
        if domain in self.warm:
            expert=self.warm.pop(domain); self.warm_hits+=1; self._put_hot(domain, expert); expert.tier_loaded_from=Tier.WARM; return expert,Tier.WARM
        path=self.expert_dir/f"{domain}.json"
        if not path.exists(): path=self.expert_dir/"general.json"; domain="general"
        expert=self._load(path); self.cold_loads+=1; self._put_hot(domain, expert); expert.tier_loaded_from=Tier.COLD; return expert,Tier.COLD
    def _load(self, path:Path)->Expert:
        data=json.loads(path.read_text(encoding="utf-8"))
        return Expert(name=str(data["name"]), domain=str(data["domain"]), facts={str(k):str(v) for k,v in data.get("facts",{}).items()}, default_response=str(data.get("default_response","I do not know.")))
    def _put_hot(self, domain:str, expert:Expert)->None:
        self.hot[domain]=expert
        while len(self.hot)>self.hot_capacity:
            d,e=self.hot.popitem(last=False); self._put_warm(d,e)
    def _put_warm(self, domain:str, expert:Expert)->None:
        self.warm[domain]=expert
        while len(self.warm)>self.warm_capacity: self.warm.popitem(last=False)
