# Routed Memory Experts: A Feasibility Thesis for Memory-Hierarchical AI Systems

## Abstract

This paper argues for a narrowed version of the “micro-models streamed from SSD” thesis. The feasible future is not dense per-token streaming of arbitrary full neural models from SSD. The feasible future is a routed memory-hierarchy system: resident base models, sparse or adapter-based experts, learned routing, hot/warm/cold caches across HBM/DRAM/NVMe, retrieval/tool specialists, and escalation to stronger models when confidence is low. The inspiration includes Dwarf Star-style tiny local agents/models, expanded into a system where thousands of agents may each own hot model/adaptor state while remaining routable and measurable.

## Thesis

AI serving will become a routed memory-hierarchy problem. The winning architecture is unlikely to be thousands of full models streamed from SSD per token. It is more likely to be a resident general base model plus sparse or adapter-based experts, with learned routing, cache-aware scheduling, retrieval/tool specialists, and NVMe as a cold capacity tier. The technical hinge is whether routers can select and prefetch the right small expert at coarse enough granularity to preserve latency and batching while improving quality and cost.

## Claims

1. Dense full-weight SSD streaming per token is infeasible for interactive workloads because SSD bandwidth is orders of magnitude below HBM bandwidth.
2. Sparse MoE validates conditional compute, but production MoE generally keeps active experts resident in accelerator/distributed memory.
3. LoRA/adapters are the most practical “micro expert” unit because they are small enough to cache and move across tiers.
4. Retrieval and tools should absorb much of the long-tail knowledge/capability burden rather than encoding every specialty in weights.
5. Thousands of agent-owned hot models are plausible only with locality-aware routing, cache admission, fallback, and independent evaluation of route regret.

## Wargame

### Pro

Routing is already central to frontier and production AI: MoE, model cascades, retrieval, tools, and LoRA serving all reduce cost or increase specialization. Most user requests do not need all model capacity. A router can select an expert at request, session, document, or tool-call granularity, allowing NVMe to act as a large cold tier for long-tail experts.

### Con

SSD is far too slow to substitute for HBM. If every token requires loading arbitrary cold weights, latency collapses. Routers make mistakes, specialists can be brittle, and too many expert choices fragment batches. Operational complexity can overwhelm theoretical efficiency.

### Synthesis

The thesis is viable when the unit of specialization is small and the routing granularity is coarse. It is weak when it assumes token-level cold streaming of dense full models.

## Proposed Architecture

The architecture has five planes:

1. Router plane — classifies prompt domain, difficulty, risk, and tool needs; chooses top-k experts and fallback policy.
2. Expert plane — base models, LoRA/adapters, sparse experts, full specialist models, retrieval indexes, tools, and agent-owned hot models.
3. Memory plane — HBM hot tier, DRAM warm tier, NVMe cold tier, and object-store archive.
4. Serving plane — vLLM/SGLang/llama.cpp runtimes, batching, KV paging/offload, adapter loading.
5. Evaluation plane — workload suites, router regret, accuracy/cost/latency, cache hit rate, fallback rate, and safety failures.

## Proof Strategy

This repository proves the claims in escalating phases:

- Phase 1: Deterministic routed expert control plane with hot/warm/cold cache and real-world workload fixtures.
- Phase 2: Replace deterministic experts with local small models and/or LoRA adapters.
- Phase 3: Serve adapters through vLLM/SGLang and measure load/cache/latency behavior.
- Phase 4: Add real retrieval/tool specialists and route regret evaluation.
- Phase 5: Compare against always-generalist and always-frontier baselines on larger public benchmarks.

## Current Result

The current MVP demonstrates Phase 1 only. It proves the control-plane mechanics and measurement loop: a router selects focused experts, cold experts are loaded from disk, hot/warm cache behavior is observed, and the routed system outperforms a generalist baseline on the included real-world workload fixture. It does not yet prove neural model quality or production vLLM serving.
