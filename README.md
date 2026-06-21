# Routed Memory Experts

Proof-oriented research and implementation of a routed memory-hierarchy AI thesis: resident base capability, thousands of routable focused experts/agents, and hot/warm/cold model tiers spanning accelerator memory, DRAM, and NVMe/SSD.

The project treats the thesis as falsifiable engineering work. Every supported claim should have a machine-readable artifact under `runs/`; every unsupported claim should be recorded as a blocker or limitation.

## Contents

- `paper/routed-memory-experts.md` — arXiv-style preprint draft with claims, methods, results, and limitations.
- `docs/COMPLETION_IMPLEMENTATION_PLAN.md` — detailed plan for completing all feasible local tasks.
- `docs/FULL_PROOF_PHASED_PLAN.md` — remaining proof plan with local and external phases.
- `docs/EXTERNAL_PROOF_RUNBOOK.md` — Kaggle/GPU/public-benchmark instructions for external gaps.
- `docs/PUBLIC_BENCHMARK_CONTRACT.md` — requirements before claiming public benchmark performance.
- `docs/ADAPTER_CANDIDATES.md` — adapter serving vs. quality-superiority status.
- `docs/IMPLEMENTATION_PLAN.md` — original phased implementation plan.
- `docs/THESIS_PROGRESS.md` — iteration-by-iteration evidence vs. thesis.
- `src/routed_memory_experts/` — executable proof harness.
- `workloads/real_world_v1.jsonl` — original domain-diverse workload fixture.
- `workloads/benchmark_expanded_v1.jsonl` — larger deterministic benchmark fixture.
- `adapters/vllm_metal_manifest.json` — current adapter route manifest.
- `scripts/` — reproducibility scripts.
- `runs/` — proof artifacts.

## Current proof scope

The repo now proves:

- deterministic routed expert control plane;
- hot/warm/cold cache observations with cold disk loads;
- 1,000-agent locality simulation;
- learned-router improvement over keyword routing on held-out synonym prompts;
- local neural routed context injection through Ollama;
- Apple Silicon vLLM-Metal/MLX OpenAI-compatible serving;
- real LoRA adapter loading under vLLM-Metal;
- CUDA vLLM LoRA serving on Kaggle T4 with `max_cpu_loras > max_loras`;
- artifact validation for proof JSONs;
- optional live base-vs-LoRA and concurrency benchmarks when a vLLM-Metal server is running.

It does **not** yet prove upstream-style vLLM-Metal CPU LoRA cache tiering with `max_cpu_loras > max_loras`; the current Metal runtime explicitly rejects that mode. The analogous upstream CUDA vLLM configuration is proven in the bounded Kaggle artifacts under `runs/cuda-vllm-*.json`.

## Quick start

```bash
uv venv --python 3.11
. .venv/bin/activate
uv pip install -e '.[dev]'
pytest -q
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
rme prove --workload workloads/benchmark_expanded_v1.jsonl --experts experts --output runs/benchmark-proof.json
rme summarize-proof-gaps --runs runs --output runs/proof-gap-ledger.json
rme validate-artifacts --path runs
```

## Apple Silicon vLLM-Metal proof

Install vLLM-Metal:

```bash
scripts/install-vllm-metal.sh
```

Start the known-good base + LoRA server:

```bash
scripts/start-vllm-metal-lora.sh
```

In another shell:

```bash
. .venv/bin/activate
rme prove-openai --base-url http://127.0.0.1:8000/v1 --model tldr \
  --workload workloads/real_world_v1.jsonl --experts experts \
  --output runs/vllm-metal-lora-proof.json --limit 6

rme compare-openai-models --base-url http://127.0.0.1:8000/v1 \
  --base-model Qwen/Qwen3-0.6B --expert-model tldr \
  --workload workloads/real_world_v1.jsonl --experts experts \
  --output runs/vllm-metal-base-vs-lora.json --limit 6

rme benchmark-openai-concurrency --base-url http://127.0.0.1:8000/v1 --model tldr \
  --workload workloads/real_world_v1.jsonl --experts experts \
  --output runs/vllm-metal-concurrency.json --requests 12 --concurrency 3
```

## One-command local proof suite

```bash
scripts/run-local-proofs.sh
```

This runs deterministic proofs and skips live OpenAI-compatible proofs if no server is reachable at `http://127.0.0.1:8000/health`.

## Hosted CUDA vLLM proof

With a Kaggle GPU notebook and internet enabled:

```bash
%cd /kaggle/working
!git clone https://github.com/CJCShadowsan/routed-memory-experts.git
%cd routed-memory-experts
!python scripts/kaggle_cuda_vllm_proof.py
```

The successful proof writes `runs/cuda-vllm-models.json`, `runs/cuda-vllm-tldr-proof.json`, `runs/cuda-vllm-pts-proof.json`, `runs/cuda-vllm-base-vs-tldr.json`, and `runs/cuda-vllm-concurrency.json`.
