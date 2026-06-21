#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
pytest -q
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
rme prove --workload workloads/benchmark_expanded_v1.jsonl --experts experts --output runs/benchmark-proof.json
rme simulate-fleet --agents 1000 --requests 5000 --hot-capacity 128 --locality-window 64 --output runs/fleet.json
rme compare-routers --train workloads/router_train_v1.jsonl --dev workloads/router_dev_v1.jsonl --output runs/router-comparison.json
rme check-runtimes --output runs/runtime-readiness.json || true
if python3 - <<'PY'
import urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)
except Exception:
    raise SystemExit(1)
PY
then
  rme prove-openai --base-url http://127.0.0.1:8000/v1 --model tldr --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-lora-proof.json --limit 6
  rme compare-openai-models --base-url http://127.0.0.1:8000/v1 --base-model Qwen/Qwen3-0.6B --expert-model tldr --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-base-vs-lora.json --limit 6
  rme benchmark-openai-concurrency --base-url http://127.0.0.1:8000/v1 --model tldr --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-concurrency.json --requests 12 --concurrency 3
else
  echo "Skipping live OpenAI-compatible proofs: http://127.0.0.1:8000/health is not reachable."
fi
rme validate-artifacts --path runs
