# Thesis Progress

## Ultimate goal

Fully prove or falsify the paper’s narrowed thesis: routed memory-hierarchy expert systems can improve quality/cost tradeoffs by combining resident base capability, focused experts/adapters, learned routing, thousands of agent-owned hot models where useful, and hot/warm/cold storage tiers.

## Iteration 1 assessment

Status: Phase-1 control-plane scaffold implemented.

Closeness to ultimate goal: 20%.

Evidence added:

- Deterministic router.
- Hot/warm/cold expert cache.
- Cold expert loading from disk as NVMe stand-in.
- Domain-diverse real-world workload fixture.
- Generalist baseline comparison.
- Proof CLI that records route regret, cache behavior, fallback, and accuracy.
- Tests for routing, cache tiers, and end-to-end proof.

Claims supported:

- A routed expert control plane can be measured end-to-end.
- Cold specialists can be staged from a filesystem tier and promoted to hot/warm cache.
- Routed specialists can outperform a generalist baseline on a domain-labeled workload fixture.

Claims not yet proven:

- Real neural LoRA/adapters outperform a base model.
- Thousands of agent-owned hot models can be scheduled without unacceptable latency or batch fragmentation.
- vLLM/SGLang serving can load/cache adapters within acceptable p95 latency.
- Public benchmark workloads reproduce the result.
- Router learning beats keyword routing.
- NVMe/KV offload behavior improves real long-context serving.

Next highest-value iteration:

Add a real small-model or adapter backend while preserving deterministic CI fallback, then measure latency and quality deltas on the same workload.

## Iteration 2 assessment

Status: Added thousand-agent fleet simulation.

Closeness to ultimate goal: 25%.

Evidence added:

- Deterministic simulation of 1,000 routable agent-owned hot models.
- Locality-aware hot cache model with cold-load and hit-rate metrics.
- CLI command `rme simulate-fleet` that writes reproducible fleet proof JSON.
- Regression tests proving a 1,000-agent workload can reach high cache hit rate when request locality exists.

Claims additionally supported:

- The Dwarf Star-inspired expansion to thousands of routable agents is architecturally plausible only when locality-aware routing and cache admission keep active agents hot.

Claims still not proven:

- Real agent-owned neural models can maintain comparable locality under production traffic.
- Cache locality remains high under adversarial or uniformly random routing.
- Model memory pressure, batching fragmentation, and p95 latency are acceptable with real runtimes.
