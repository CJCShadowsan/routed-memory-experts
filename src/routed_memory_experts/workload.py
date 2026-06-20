from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List
from .models import WorkloadItem

def load_workload(path: str | Path) -> List[WorkloadItem]:
    items=[]
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no,line in enumerate(handle,1):
            line=line.strip()
            if not line: continue
            data=json.loads(line)
            try:
                items.append(WorkloadItem(id=str(data["id"]), domain=str(data["domain"]), prompt=str(data["prompt"]), expected_contains=[str(x).lower() for x in data["expected_contains"]], risk=str(data.get("risk","normal"))))
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no}: missing required field {exc}") from exc
    if not items: raise ValueError(f"workload {path} is empty")
    return items

def is_correct(response: str, expected_contains: Iterable[str]) -> bool:
    lower=response.lower()
    return all(token.lower() in lower for token in expected_contains)
