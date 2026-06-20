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

The current MVP proves the control-plane mechanics: routing, hot/warm/cold expert cache, cold on-disk expert loading, fallback, and metrics. It does **not yet** prove frontier neural quality or production vLLM/LoRA serving.

## Quick start

```bash
uv venv --python 3.11
. .venv/bin/activate
uv pip install -e '.[dev]'
pytest -q
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
```

Success criteria for Phase 1:

- routed accuracy >= 0.80
- route regret <= 0.20
- routed accuracy > generalist baseline
- proof JSON records cold loads and hot/warm cache behavior
