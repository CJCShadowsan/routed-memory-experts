# External Proof Runbook

This runbook covers the remaining thesis gaps that cannot be fully closed on the current local host.

## What is already locally proven

The repository already has machine-readable artifacts for:

- deterministic routed expert control plane;
- cache/load observations with hot/warm/cold stand-ins;
- 1,000-agent locality simulation;
- learned-router improvement on a synonym-heavy fixture;
- local neural context injection through Ollama;
- Apple Silicon vLLM-Metal base and LoRA serving;
- Apple Silicon multi-adapter manifest routing;
- bounded Kaggle CUDA vLLM proof with `max_cpu_loras > max_loras`.

Run locally:

```bash
. .venv/bin/activate
rme validate-artifacts --path runs
pytest -q
rme summarize-proof-gaps --runs runs --output runs/proof-gap-ledger.json
```

## Remaining external phases

### Phase A: Public benchmark selection

Before claiming public benchmark performance, pick a benchmark with:

- compatible license for redistribution or a documented download step;
- stable split/version;
- prompts that can be represented as JSONL;
- scoring that is either exact-match/contains based or implemented in a separate scorer;
- enough items to make base-vs-adapter comparison meaningful.

Do not use the current internal workload as a public benchmark. It is an engineering fixture.

### Phase B: High-quality adapter comparison

A serving-mechanics adapter is not necessarily a quality adapter. To claim adapter superiority, the run must produce a comparison artifact where:

- `workload_count >= 30`;
- `expert_accuracy > base_accuracy`;
- `expert_wins > expert_losses`;
- the model, adapter, workload, and scoring method are documented.

Current TLDR/PTS artifacts do not meet this quality-superiority bar; they prove loading/routing mechanics.

### Phase C: Production-scale concurrency

A production-scale serving claim requires more than the current smoke tests. Minimum external gate:

- at least 1000 requests;
- explicit concurrency level;
- zero request errors or a reported error budget;
- p50/p95/p99 latency;
- throughput;
- preserved server command and server log;
- GPU/runtime metadata.

### Phase D: Metal CPU LoRA cache-tier parity

vLLM-Metal currently rejects `max_cpu_loras > max_loras`. This is upstream-blocked unless one of these happens:

1. upstream vLLM-Metal implements CPU LoRA tiering;
2. this repo contributes and validates a local implementation;
3. the paper adopts a Metal-specific design that does not claim upstream parity.

## Kaggle CUDA smoke proof

For a free hosted NVIDIA smoke proof:

```bash
%cd /kaggle/working
!rm -rf routed-memory-experts
!git clone https://github.com/CJCShadowsan/routed-memory-experts.git
%cd routed-memory-experts
!python scripts/kaggle_cuda_vllm_proof.py
```

Download and copy back:

```text
runs/cuda-vllm-models.json
runs/cuda-vllm-tldr-proof.json
runs/cuda-vllm-pts-proof.json
runs/cuda-vllm-base-vs-tldr.json
runs/cuda-vllm-concurrency.json
runs/kaggle-vllm-startup-v0-xformers.log
```

## External public benchmark/capacity run

Once you have a reviewed benchmark workload JSONL and a live OpenAI-compatible server, use:

```bash
scripts/run-openai-public-benchmark.sh \
  --base-url http://127.0.0.1:8000/v1 \
  --base-model Qwen/Qwen3-0.6B \
  --expert-model tldr \
  --workload workloads/public_benchmark.jsonl \
  --requests 1000 \
  --concurrency 8
```

The script writes comparison, proof, concurrency, validation, and gap-ledger artifacts under `runs/`.

## Folding external artifacts back into the paper

After copying external artifacts locally:

```bash
. .venv/bin/activate
rme validate-artifacts --path runs
pytest -q
rme summarize-proof-gaps --runs runs --output runs/proof-gap-ledger.json
git diff --check
```

Only then update:

- `paper/routed-memory-experts.md`
- `docs/THESIS_PROGRESS.md`
- `docs/ADAPTER_CANDIDATES.md` if new adapters were evaluated

Use artifact values directly. Do not restate notebook output from memory.
