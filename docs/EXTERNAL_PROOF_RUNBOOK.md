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

The selected initial public benchmark is GSM8K:

- Source: `openai/gsm8k`.
- Provenance: https://huggingface.co/datasets/openai/gsm8k
- License: MIT.
- Local converted sample: `workloads/gsm8k_public_sample.jsonl`.
- Builder: `scripts/build-gsm8k-public-workload.py`.

Before claiming broader public benchmark performance, any additional benchmark should have:

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

Once you have a live OpenAI-compatible server, run the selected GSM8K sample with:

```bash
scripts/run-openai-public-benchmark.sh \
  --base-url http://127.0.0.1:8000/v1 \
  --base-model Qwen/Qwen3-0.6B \
  --expert-model tldr \
  --workload workloads/gsm8k_public_sample.jsonl \
  --requests 1000 \
  --concurrency 8
```

The script writes comparison, proof, concurrency, validation, and gap-ledger artifacts under `runs/`. For adapter-quality claims, replace `--expert-model tldr` with a math-capable adapter served by the same base model; TLDR/PTS remain serving-mechanics adapters, not math-quality candidates.

For the currently selected math candidate on Kaggle/CUDA, use the dedicated script:

```bash
%cd /kaggle/working
!rm -rf routed-memory-experts
!git clone https://github.com/CJCShadowsan/routed-memory-experts.git
%cd routed-memory-experts
!python scripts/kaggle_cuda_gsm8k_vllm_public_benchmark.py
```

This serves `Qwen/Qwen2.5-0.5B-Instruct` with `tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora`, runs the non-leaky GSM8K public benchmark command, a routed mechanics proof, concurrency, proof-gap summarization, and artifact validation.

Local Mac fallback when the OpenAI HTTP server path is blocked:

```bash
python scripts/run-local-vllm-gsm8k-direct.py \
  --limit 32 \
  --output runs/local-vllm-gsm8k-public-openai-benchmark.json
```

The current local direct-vLLM result is intentionally not a quality win: base accuracy was 5/32 and adapter accuracy was 3/32.

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
