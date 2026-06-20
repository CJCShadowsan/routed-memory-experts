# Routed Memory Experts

Proof-oriented research and implementation of a routed memory-hierarchy AI thesis: resident base capability, thousands of routable focused experts/agents, and hot/warm/cold model tiers spanning HBM, DRAM, and NVMe/SSD.

The user inspiration is Dwarf Star-style tiny local agents/models, but this project expands the idea into a systems architecture where thousands of agents may each have their own hot model/adaptor state and can be routed, cached, evaluated, and escalated.

Contents:

- `paper/routed-memory-experts.md` — research paper and narrowed thesis.
- `docs/IMPLEMENTATION_PLAN.md` — full implementation plan.
- `docs/adr/0001-routed-memory-hierarchy.md` — architectural decision record.
- `docs/LOOP.md` — continual proof loop.
- `docs/THESIS_PROGRESS.md` — iteration-by-iteration evidence vs thesis.
- `src/routed_memory_experts/` — executable MVP.
- `workloads/real_world_v1.jsonl` — domain-diverse workload fixture.
- `tests/` — regression tests.

## Current proof scope

The repo now proves the control-plane mechanics, learned routing, a real local Ollama neural backend, and an Apple Silicon vLLM-Metal/MLX OpenAI-compatible serving path with a loaded LoRA adapter. It is still a bounded proof harness rather than a frontier-quality benchmark suite.

## Quick start

```bash
uv venv --python 3.11
. .venv/bin/activate
uv pip install -e '.[dev]'
pytest -q
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
rme compare-routers --train workloads/router_train_v1.jsonl --dev workloads/router_dev_v1.jsonl --output runs/router-comparison.json
```

Success criteria for Phase 1:

- routed accuracy >= 0.80
- route regret <= 0.20
- routed accuracy > generalist baseline
- proof JSON records cold loads and hot/warm cache behavior

Optional local neural proof, when Ollama is installed and a model is available:

```bash
rme prove-ollama --model gemma4:e4b --workload workloads/real_world_v1.jsonl --experts experts --output runs/ollama-proof.json --limit 6
```

This routes prompts through the same expert control plane, injects the selected expert context into a real local Ollama model, and records accuracy plus p50/p95 latency.

Apple Silicon vLLM-Metal proof, after installing vLLM-Metal into `~/.venv-vllm-metal` and starting the server:

```bash
source ~/.venv-vllm-metal/bin/activate
VLLM_METAL_MEMORY_FRACTION=0.5 VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 --max-model-len 1024 \
  --enable-lora --max-loras 2 --lora-modules tldr=phh/Qwen3-0.6B-TLDR-Lora

rme prove-openai --base-url http://127.0.0.1:8000/v1 --model tldr \
  --workload workloads/real_world_v1.jsonl --experts experts \
  --output runs/vllm-metal-lora-proof.json --limit 6
```

This proves the OpenAI-compatible vLLM-Metal/MLX serving path, with a Qwen3 base model and a loaded LoRA adapter exposed as model `tldr`.
