#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="http://127.0.0.1:8000/v1"
BASE_MODEL="Qwen/Qwen3-0.6B"
EXPERT_MODEL="tldr"
WORKLOAD=""
EXPERTS="experts"
REQUESTS="1000"
CONCURRENCY="8"
OUTPUT_PREFIX="runs/public-benchmark"
ADAPTER_MANIFEST=""

usage() {
  cat <<'USAGE'
Usage: scripts/run-openai-public-benchmark.sh --workload PATH [options]

Options:
  --base-url URL           OpenAI-compatible /v1 base URL (default: http://127.0.0.1:8000/v1)
  --base-model MODEL       Base served model id (default: Qwen/Qwen3-0.6B)
  --expert-model MODEL     Expert/adapter served model id (default: tldr)
  --workload PATH          Public/reviewed benchmark JSONL workload (required)
  --experts PATH           Expert directory for expected evidence scoring (default: experts)
  --requests N             Concurrency benchmark request count (default: 1000)
  --concurrency N          Concurrency level (default: 8)
  --output-prefix PREFIX   Output prefix under runs/ (default: runs/public-benchmark)
  --adapter-manifest PATH  Optional adapter manifest for routed proof/concurrency

This script assumes a live OpenAI-compatible server is already running. It does
not install or start vLLM/SGLang. Use it after benchmark licensing/provenance has
been reviewed and after the target base/adapters are served.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --expert-model) EXPERT_MODEL="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
    --adapter-manifest) ADAPTER_MANIFEST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$WORKLOAD" ]]; then
  echo "--workload is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$WORKLOAD" ]]; then
  echo "workload not found: $WORKLOAD" >&2
  exit 2
fi

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

BASE_URL="$BASE_URL" python3 - <<'PY'
import os
import urllib.request
base_url = os.environ['BASE_URL'].rstrip('/')
if base_url.endswith('/v1'):
    base_url = base_url[:-3]
url = base_url.rstrip('/') + '/health'
with urllib.request.urlopen(url, timeout=10) as response:
    if response.status != 200:
        raise SystemExit(f'health check failed: {response.status}')
print(f'health check passed: {url}')
PY

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

rme compare-openai-models \
  --base-url "$BASE_URL" \
  --base-model "$BASE_MODEL" \
  --expert-model "$EXPERT_MODEL" \
  --workload "$WORKLOAD" \
  --experts "$EXPERTS" \
  --output "${OUTPUT_PREFIX}-base-vs-adapter.json"

PROOF_ARGS=(
  --base-url "$BASE_URL"
  --model "$EXPERT_MODEL"
  --workload "$WORKLOAD"
  --experts "$EXPERTS"
  --output "${OUTPUT_PREFIX}-expert-proof.json"
)
CONCURRENCY_ARGS=(
  --base-url "$BASE_URL"
  --model "$EXPERT_MODEL"
  --workload "$WORKLOAD"
  --experts "$EXPERTS"
  --output "${OUTPUT_PREFIX}-concurrency.json"
  --requests "$REQUESTS"
  --concurrency "$CONCURRENCY"
)
if [[ -n "$ADAPTER_MANIFEST" ]]; then
  PROOF_ARGS+=(--adapter-manifest "$ADAPTER_MANIFEST")
  CONCURRENCY_ARGS+=(--adapter-manifest "$ADAPTER_MANIFEST")
fi

rme prove-openai "${PROOF_ARGS[@]}"
rme benchmark-openai-concurrency "${CONCURRENCY_ARGS[@]}"
rme summarize-proof-gaps --runs runs --output runs/proof-gap-ledger.json
rme validate-artifacts --path runs
