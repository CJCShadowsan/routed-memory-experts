from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .router import KeywordRouter
from .workload import load_workload

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*")


@dataclass
class RouterComparisonSummary:
    train_count: int
    dev_count: int
    keyword_correct: int
    learned_correct: int
    keyword_accuracy: float
    learned_accuracy: float
    learned_beats_keyword: bool


class NaiveBayesRouter:
    def __init__(self) -> None:
        self.domain_counts: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()

    def fit(self, examples: list[tuple[str, str]]) -> None:
        for text, domain in examples:
            self.domain_counts[domain] += 1
            for token in tokenize(text):
                self.token_counts[domain][token] += 1
                self.vocabulary.add(token)

    def route(self, prompt: str) -> tuple[str, float, str]:
        if not self.domain_counts:
            return "general", 0.0, "model is not trained"
        tokens = tokenize(prompt)
        total_examples = sum(self.domain_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        scores: dict[str, float] = {}
        for domain, count in self.domain_counts.items():
            domain_total = sum(self.token_counts[domain].values())
            score = math.log(count / total_examples)
            for token in tokens:
                score += math.log((self.token_counts[domain][token] + 1) / (domain_total + vocab_size))
            scores[domain] = score
        domain = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        confidence = min(0.99, 0.50 + margin / 10)
        return domain, confidence, "naive-bayes token posterior"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def compare_routers(train_path: str | Path, dev_path: str | Path, output_path: str | Path | None = None) -> RouterComparisonSummary:
    train_items = load_workload(train_path)
    dev_items = load_workload(dev_path)
    learned = NaiveBayesRouter()
    learned.fit([(item.prompt, item.domain) for item in train_items])
    keyword = KeywordRouter()
    keyword_correct = 0
    learned_correct = 0
    records = []
    for item in dev_items:
        keyword_domain, _, _ = keyword.route(item.prompt)
        learned_domain, confidence, _ = learned.route(item.prompt)
        keyword_ok = keyword_domain == item.domain
        learned_ok = learned_domain == item.domain
        keyword_correct += int(keyword_ok)
        learned_correct += int(learned_ok)
        records.append(
            {
                "id": item.id,
                "oracle_domain": item.domain,
                "keyword_domain": keyword_domain,
                "learned_domain": learned_domain,
                "learned_confidence": confidence,
                "keyword_correct": keyword_ok,
                "learned_correct": learned_ok,
            }
        )
    summary = RouterComparisonSummary(
        train_count=len(train_items),
        dev_count=len(dev_items),
        keyword_correct=keyword_correct,
        learned_correct=learned_correct,
        keyword_accuracy=keyword_correct / len(dev_items) if dev_items else 0.0,
        learned_accuracy=learned_correct / len(dev_items) if dev_items else 0.0,
        learned_beats_keyword=learned_correct > keyword_correct,
    )
    if output_path:
        write_router_comparison(summary, records, output_path)
    return summary


def router_comparison_to_dict(summary: RouterComparisonSummary) -> dict:
    return summary.__dict__.copy()


def write_router_comparison(summary: RouterComparisonSummary, records: list[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = router_comparison_to_dict(summary)
    payload["records"] = records
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
