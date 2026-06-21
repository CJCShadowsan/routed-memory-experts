import json
import importlib.util
from pathlib import Path

from routed_memory_experts.router import KeywordRouter
from routed_memory_experts.workload import load_workload

_SCRIPT = Path("scripts/build-gsm8k-public-workload.py")
_SPEC = importlib.util.spec_from_file_location("build_gsm8k_public_workload", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
convert_row = _MODULE.convert_row
extract_final_answer = _MODULE.extract_final_answer


def test_extract_final_answer_from_gsm8k_marker():
    answer = "She sells 9 eggs.\n#### 18"
    assert extract_final_answer(answer) == "18"


def test_convert_gsm8k_row_preserves_public_benchmark_metadata():
    row = {
        "row_idx": 7,
        "row": {
            "question": "There are 2 bags with 3 marbles each. How many marbles?",
            "answer": "2 * 3 = <<2*3=6>>6\n#### 6",
        },
    }
    item = convert_row(row, "test")
    assert item["id"] == "gsm8k-test-7"
    assert item["source"].startswith("GSM8K")
    assert item["license"] == "mit"
    assert item["provenance_url"].startswith("https://huggingface.co/datasets/openai/gsm8k")
    assert item["domain"] == "math"
    assert item["expected_contains"] == ["6"]


def test_gsm8k_public_sample_loads_and_routes_to_math():
    items = load_workload("workloads/gsm8k_public_sample.jsonl")
    assert len(items) >= 30
    assert {item.domain for item in items} == {"math"}
    first = json.loads(open("workloads/gsm8k_public_sample.jsonl", encoding="utf-8").readline())
    for key in ["source", "license", "provenance_url", "split", "scorer"]:
        assert key in first
    routes = [KeywordRouter().route(item.prompt)[0] for item in items]
    assert set(routes) == {"math"}
