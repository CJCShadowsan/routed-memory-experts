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

## Iteration 3 assessment

Status: Added real local Ollama model proof path.

Closeness to ultimate goal: 35%.

Evidence added:

- New `rme prove-ollama` command routes workload items through the expert control plane and then asks a real local Ollama model to answer using the selected specialist context.
- Verified on local model `gemma4:e4b` for the first six workload items.
- Proof artifact written to `runs/ollama-proof.json`.

Observed result:

- Workload count: 6.
- Correct count: 5.
- Accuracy: 0.8333.
- p50 latency: about 6095 ms.
- p95 latency: about 6988 ms.

Claims additionally supported:

- A real local neural model can consume routed specialist context and pass a bounded workload threshold.
- The deterministic control plane can front a real model backend without changing the workload/proof interface.

Claims still not proven:

- LoRA/adapters through vLLM/SGLang outperform a base model.
- The neural proof currently uses context injection, not adapter weights or separate hot models.
- Latency is high and not yet optimized for production serving.

## Iteration 4 assessment

Status: Added learned-router comparison.

Closeness to ultimate goal: 42%.

Evidence added:

- New Naive Bayes router trained from labeled workload examples.
- New synonym-heavy router train/dev workloads.
- New `rme compare-routers` command writes `runs/router-comparison.json`.
- Tests prove the learned router can beat the keyword router on held-out synonym prompts.

Observed result:

- Train count: 15.
- Dev count: 5.
- Keyword router accuracy: 0.0.
- Learned router accuracy: 1.0.
- Learned router beat keyword baseline: true.

Claims additionally supported:

- Router learning can improve route selection over fixed keyword rules when prompts use domain synonyms.

Claims still not proven:

- Learned routing has not yet been validated on a large public benchmark.
- Learned routing has not yet been connected to real adapter/model load balancing.
- vLLM/SGLang adapter proof remains the largest unproven claim.

## Iteration 5 assessment

Status: Added serving-runtime readiness gate and hit a real blocker.

Closeness to ultimate goal: 45%.

Evidence added:

- New `rme check-runtimes` command checks for Ollama, vLLM, SGLang, `nvidia-smi`, and CUDA GPU readiness.
- Proof artifact written to `runs/runtime-readiness.json`.
- Test coverage for serializing runtime readiness/blocker state.

Observed result on current host:

- Ollama available: true.
- vLLM importable: false.
- SGLang importable: false.
- `nvidia-smi` available: false.
- CUDA GPU detected: false.
- Production adapter runtime ready: false.

Blocker:

- Production LoRA/vLLM/SGLang proof requires an importable serving runtime and CUDA GPU. This macOS host can prove the local Ollama/context-routed backend, deterministic cache/routing control plane, learned-router improvement, and fleet-locality simulation, but it cannot honestly prove CUDA-backed vLLM/SGLang adapter serving.

Next required environment to prove remaining claims:

- Linux host with NVIDIA GPU and CUDA.
- vLLM or SGLang installed.
- At least one base model and multiple compatible LoRA adapters.
- Benchmark workload large enough to measure p50/p95 latency, adapter load time, cache hit rate, route regret, and quality vs base/fallback.
