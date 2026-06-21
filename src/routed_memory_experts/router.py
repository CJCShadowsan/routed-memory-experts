from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

@dataclass(frozen=True)
class DomainRule:
    domain: str
    keywords: Tuple[str, ...]

DEFAULT_RULES=(
    DomainRule("kubernetes", ("kubernetes","k8s","pod","helm","deployment","nodeport","ingress")),
    DomainRule("python", ("python","pytest","decorator","generator","asyncio","dict","list")),
    DomainRule("finance", ("npv","dcf","discount rate","cash flow","ebitda","wacc","terminal value")),
    DomainRule("medical-literature", ("pubmed","randomized","cohort","hazard ratio","confidence interval","systematic review")),
    DomainRule("math", ("how many", "how much", "total", "sum", "difference", "product", "percent", "%", "ratio", "per", "each", "times", "mile", "hour", "minute", "pound", "feet", "discount", "cost", "paid", "$")),
    DomainRule("security", ("cve","xss","csrf","sql injection","threat model","rce")),
)

class KeywordRouter:
    """Deterministic router used as measurable stand-in for a learned router."""
    def __init__(self, rules: Iterable[DomainRule]=DEFAULT_RULES): self.rules=tuple(rules)
    def route(self, prompt: str) -> tuple[str,float,str]:
        lower=prompt.lower(); scores:Dict[str,int]={}; matched={}
        for rule in self.rules:
            hits=[kw for kw in rule.keywords if kw in lower]
            if hits: scores[rule.domain]=len(hits); matched[rule.domain]=hits
        if not scores: return "general",0.25,"no domain keywords matched"
        domain,score=max(scores.items(), key=lambda kv: kv[1])
        return domain, min(0.95,0.45+0.15*score), "matched "+", ".join(matched[domain])
