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

Blocker, revised after checking current vLLM documentation:

- CUDA is not the only valid production-serving path. vLLM now documents Apple Silicon GPU acceleration through the community-maintained `vllm-metal` plugin, which uses MLX and Apple's Metal framework. The current blocker on this host is therefore not “no CUDA”; it is that `vllm_metal` is not installed/importable in the active environment yet.

Next required environment to prove remaining claims:

- Either Apple Silicon with vLLM-Metal/MLX installed, or Linux with NVIDIA GPU/CUDA plus vLLM/SGLang.
- At least one base model and multiple compatible LoRA/adapters, preferably from MLX-compatible model/adaptor artifacts on Apple Silicon.
- Benchmark workload large enough to measure p50/p95 latency, adapter load time, cache hit rate, route regret, and quality vs base/fallback.

## Iteration 6 assessment

Status: Corrected runtime readiness model for Apple Silicon.

Closeness to ultimate goal: 47%.

Evidence added:

- Checked upstream vLLM documentation and vLLM-Metal repository metadata.
- vLLM documents Apple Silicon GPU acceleration through `vllm-metal`, a community-maintained plugin using MLX and Metal.
- vLLM-Metal repository contains LoRA-related implementation paths under `vllm_metal/v1/lora/` and tests such as `tests/test_lora.py`.
- Runtime readiness now distinguishes CUDA vLLM/SGLang from Apple Silicon vLLM-Metal/MLX.

Claims additionally supported:

- The production adapter proof may be achievable on this Apple Silicon host through vLLM-Metal rather than requiring a CUDA machine.

Remaining blocker:

- `vllm_metal` is not installed/importable in the active environment. The next proof slice should install/vet vLLM-Metal, start a local vLLM-Metal server with an MLX-community model, and then extend `rme prove-ollama` into a generic OpenAI-compatible routed-serving proof.

## Iteration 7 assessment

Status: Installed vLLM-Metal and proved Apple Silicon vLLM/MLX serving with LoRA loaded.

Closeness to ultimate goal: 70%.

Evidence added:

- Installed vLLM-Metal into `~/.venv-vllm-metal` using upstream installer after inspecting docs and installer behavior.
- Verified `vllm`, `vllm_metal`, `mlx`, and `openai` are importable in the vLLM-Metal environment.
- Started a vLLM-Metal OpenAI-compatible server on Apple Silicon using `Qwen/Qwen3-0.6B`.
- Added generic `rme prove-openai` proof path for OpenAI-compatible servers.
- Started vLLM-Metal with `--enable-lora` and loaded `phh/Qwen3-0.6B-TLDR-Lora` as served model `tldr`.
- Recorded served model list in `runs/vllm-metal-models.json`, showing base model `Qwen/Qwen3-0.6B` and LoRA model `tldr` with parent `Qwen/Qwen3-0.6B`.
- Recorded routed LoRA proof in `runs/vllm-metal-lora-proof.json`.

Observed result:

- Runtime readiness: production adapter runtime ready via Apple Silicon vLLM-Metal/MLX.
- Base vLLM-Metal routed proof: accuracy 0.8333 on first six workload items, p50 about 238 ms, p95 about 319 ms.
- LoRA vLLM-Metal routed proof: accuracy 1.0 on first six workload items, p50 about 309 ms, p95 about 327 ms.

Important implementation finding:

- vLLM-Metal LoRA currently does not implement upstream `max_cpu_loras > max_loras` cache-tier behavior. Attempting `--max-cpu-loras 4 --max-loras 2` failed with `NotImplementedError`; setting/omitting `--max-cpu-loras` so it equals `max_loras` allowed the LoRA server to start. This means Apple Silicon proves LoRA serving, but not upstream-style CPU LoRA cache tiering yet.

Claims additionally supported:

- Apple Silicon can run the production-serving proof via vLLM-Metal/MLX without CUDA.
- The proof harness can exercise an OpenAI-compatible vLLM server.
- A real LoRA adapter can be loaded and routed through the vLLM-Metal serving path.

Claims still not fully proven:

- Large public benchmark validation.
- Many simultaneous LoRA/adaptor agents under real concurrency.
- Hot/warm/cold adapter tiering with `max_cpu_loras > max_loras` on Metal, because vLLM-Metal explicitly does not implement that upstream cache tier yet.
