#!/usr/bin/env python3
"""Build a public benchmark workload JSONL from GSM8K.

GSM8K is a public grade-school math word-problem benchmark. This converter uses
Hugging Face's datasets-server rows API and writes records compatible with the
RME workload contract while preserving source/license/provenance metadata.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "openai/gsm8k"
CONFIG = "main"
DEFAULT_SPLIT = "test"
LICENSE = "mit"
SOURCE = "GSM8K (Grade School Math 8K)"
PROVENANCE_URL = "https://huggingface.co/datasets/openai/gsm8k"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"


def extract_final_answer(answer: str) -> str:
    marker = "####"
    if marker in answer:
        final = answer.rsplit(marker, 1)[1].strip()
    else:
        final = answer.strip().splitlines()[-1].strip()
    final = final.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", final)
    if not match:
        raise ValueError(f"could not extract numeric final answer from: {answer!r}")
    return match.group(0)


def fetch_rows(split: str, offset: int, length: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{DATASET_SERVER}?{params}", timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return list(data["rows"])


def convert_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    idx = int(row["row_idx"])
    payload = row["row"]
    answer = str(payload["answer"])
    final_answer = extract_final_answer(answer)
    return {
        "id": f"gsm8k-{split}-{idx}",
        "source": SOURCE,
        "license": LICENSE,
        "provenance_url": PROVENANCE_URL,
        "split": split,
        "domain": "math",
        "prompt": str(payload["question"]),
        "expected_contains": [final_answer],
        "answer": final_answer,
        "scorer": "expected_contains_numeric_final_answer",
    }


def build_workload(output: str | Path, split: str = DEFAULT_SPLIT, offset: int = 0, limit: int = 64) -> None:
    rows = fetch_rows(split, offset, limit)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(convert_row(row, split), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RME public benchmark JSONL from GSM8K")
    parser.add_argument("--output", default="workloads/gsm8k_public_sample.jsonl")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    build_workload(args.output, args.split, args.offset, args.limit)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
