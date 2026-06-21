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


## Iteration 8 assessment

Status: Completed all locally feasible roadmap tasks for the current Apple Silicon host.

Closeness to ultimate goal: 84%.

Evidence added:

- Wrote `docs/COMPLETION_IMPLEMENTATION_PLAN.md`, an explicit task-by-task completion plan with acceptance gates.
- Added `workloads/benchmark_expanded_v1.jsonl`, a 32-item expanded deterministic benchmark fixture spanning Kubernetes, Python, finance, medical literature, security, and general routing.
- Added artifact validation via `rme validate-artifacts --path runs`; current proof artifacts validate successfully.
- Added `rme compare-openai-models`, which compares a base OpenAI-compatible model and expert/adapter model on identical routed prompts and records wins/losses/ties plus latency.
- Added `rme benchmark-openai-concurrency`, a small concurrent OpenAI-compatible serving benchmark that records throughput, errors, accuracy, and p50/p95/p99 latency.
- Added adapter manifest support via `adapters/vllm_metal_manifest.json`, `rme inspect-adapter-manifest`, and `--adapter-manifest` routing for OpenAI-compatible proofs and concurrency benchmarks.
- Proved a two-adapter vLLM-Metal configuration with `tldr=phh/Qwen3-0.6B-TLDR-Lora` and `pts=codelion/Qwen3-0.6B-PTS-DPO-LoRA` loaded under `Qwen/Qwen3-0.6B`.
- Recorded multi-adapter served-model evidence in `runs/vllm-metal-multi-lora-models.json`.
- Recorded manifest-routed proof in `runs/vllm-metal-manifest-proof.json`.
- Recorded manifest-routed concurrency evidence in `runs/vllm-metal-manifest-concurrency.json`.
- Rewrote `paper/routed-memory-experts.md` into an arXiv-style preprint draft with abstract, claims table, methodology, results, limitations, reproducibility, related work, and artifact appendix.
- Added reproducibility scripts under `scripts/`.

Observed result:

- Expanded deterministic benchmark: 32 items, routed accuracy 0.9375, baseline accuracy 0.0625, route regret 0.0.
- Base-vs-LoRA comparison on six live vLLM-Metal items: base accuracy 0.8333, expert accuracy 1.0, expert wins 1, losses 0, ties 5.
- Single-adapter concurrency benchmark: 12 requests at concurrency 3, accuracy 1.0, error count 0, throughput about 3.81 requests/s, p95 about 1332 ms.
- Manifest-routed two-adapter proof: accuracy 1.0 on six live items.
- Manifest-routed two-adapter concurrency benchmark: 12 requests at concurrency 3, accuracy 1.0, error count 0, throughput about 3.58 requests/s, p95 about 1045 ms.

Important implementation findings:

- A second Qwen3-0.6B LoRA adapter initially failed under the default `--max-lora-rank 16` with `LoRA rank 64 is greater than max_lora_rank 16`; setting `--max-lora-rank 64` allowed two LoRAs to serve concurrently.
- vLLM-Metal still does not implement upstream-style CPU LoRA cache tiering with `max_cpu_loras > max_loras`; this remains impossible to prove on the current Metal runtime.

Claims additionally supported:

- Multi-adapter LoRA serving works on this Apple Silicon vLLM-Metal environment for at least two compatible Qwen3-0.6B adapters.
- The proof harness can route domains to different served adapter IDs through a manifest.
- The harness can compare base vs expert model quality and latency on identical prompts.
- The harness can measure small concurrent routed serving runs.

Claims still not fully proven, and why they are not fully completable right now:

- Large public benchmark validation: requires benchmark licensing/selection and substantially larger evaluation runs beyond the current local proof loop.
- Production-scale concurrency: requires longer runs, larger traffic mixes, and likely stronger hardware/runtime tuning.
- Upstream-style hot/warm/cold CPU LoRA cache tiering on Metal: blocked by vLLM-Metal `NotImplementedError` for `max_cpu_loras > max_loras`.
- Production-scale CUDA vLLM/SGLang adapter cache behavior: current local host has no NVIDIA/CUDA GPU; the Kaggle proof below establishes bounded CUDA vLLM cache-tier behavior but not production-scale serving.
- High-quality domain-specialized adapter superiority: current public adapters prove serving mechanics; proving domain superiority requires finding or training appropriate adapters and larger fair benchmarks.

## Iteration 9 assessment

Status: Completed hosted CUDA vLLM LoRA cache-tier proof on Kaggle T4.

Closeness to ultimate goal: 90%.

Evidence added:

- Ran `scripts/kaggle_cuda_vllm_proof.py` in a Kaggle GPU notebook with 2x Tesla T4 visible through `nvidia-smi`.
- Started CUDA vLLM with `Qwen/Qwen3-0.6B`, `--enable-lora`, `--max-loras 2`, `--max-cpu-loras 4`, and `--max-lora-rank 64`.
- Used vLLM 0.10.2 with the V0/XFormers fallback after latest vLLM V1/FlashInfer crashed on T4 LoRA prefill; pinned Transformers below 5 to keep the tokenizer API compatible.
- Loaded two LoRA adapters: `tldr=phh/Qwen3-0.6B-TLDR-Lora` and `pts=codelion/Qwen3-0.6B-PTS-DPO-LoRA`.
- Copied CUDA proof artifacts into `runs/`: `cuda-vllm-models.json`, `cuda-vllm-tldr-proof.json`, `cuda-vllm-pts-proof.json`, `cuda-vllm-base-vs-tldr.json`, `cuda-vllm-concurrency.json`, and `kaggle-vllm-startup-v0-xformers.log`.

Observed result:

- `runs/cuda-vllm-models.json` lists the base model plus both LoRA adapter IDs, with `tldr` rooted at `phh/Qwen3-0.6B-TLDR-Lora` and `pts` rooted at `codelion/Qwen3-0.6B-PTS-DPO-LoRA`, both parented to `Qwen/Qwen3-0.6B`.
- TLDR LoRA proof: 6 workload items, accuracy 1.0, p50 latency about 4317 ms, p95 latency about 4513 ms.
- PTS LoRA proof: 6 workload items, accuracy 1.0, p50 latency about 4478 ms, p95 latency about 5598 ms.
- Base-vs-TLDR comparison: base accuracy 1.0, expert accuracy 1.0, 6 ties, 0 expert wins, 0 expert losses.
- CUDA concurrency benchmark: 24 requests at concurrency 4, 24 successes, 0 errors, accuracy 0.9167, throughput about 0.653 requests/s, p50 about 4974 ms, p95 about 9702 ms, p99 about 10512 ms.
- `rme validate-artifacts --path runs` reported all proof artifacts valid in the Kaggle run.

Claims additionally supported:

- Upstream CUDA vLLM can accept and run the CPU LoRA cache-tier configuration that vLLM-Metal rejects: `max_cpu_loras > max_loras`.
- The same OpenAI-compatible proof harness can exercise CUDA vLLM and Apple Silicon vLLM-Metal.
- Multiple CUDA-served LoRA adapters can be named, listed, routed to, and benchmarked through the harness.
- Small concurrent routed CUDA serving works without request errors in the bounded Kaggle run.

Claims still not fully proven:

- Adapter quality superiority is not proven: the CUDA base-vs-TLDR comparison tied on the six-item fixture.
- Production-scale serving is not proven: Kaggle T4 is a hosted free notebook environment with small request counts and limited runtime control.
- Broad public benchmark validation remains future work.
- Metal CPU LoRA cache tiering remains unsupported by vLLM-Metal even though CUDA vLLM supports the analogous configuration.

## Iteration 10 assessment

Status: Identified and converted the first reviewed public benchmark workload.

Closeness to ultimate goal: 91%.

Evidence added:

- Selected GSM8K (`openai/gsm8k`) as the first reviewed public benchmark target because it has a public Hugging Face dataset card, MIT license metadata, stable train/test splits, and final numeric answers marked in the source answers.
- Added `scripts/build-gsm8k-public-workload.py`, which fetches rows from the Hugging Face datasets-server API and converts final answers after `####` into the existing `expected_contains` scorer format.
- Added `workloads/gsm8k_public_sample.jsonl`, a 64-item test-split sample with `source`, `license`, `provenance_url`, `split`, `domain`, `prompt`, `expected_contains`, `answer`, and `scorer` metadata.
- Added `experts/math.json` and math routing keywords so every committed GSM8K sample item routes to the `math` domain.
- Updated `docs/PUBLIC_BENCHMARK_CONTRACT.md`, `docs/EXTERNAL_PROOF_RUNBOOK.md`, `docs/ADAPTER_CANDIDATES.md`, and `README.md` with the selected benchmark and run commands.

Claims additionally supported:

- The project now has a concrete reviewed public benchmark workload ready for live OpenAI-compatible base-vs-adapter and concurrency runs.
- The public benchmark workload satisfies the repository metadata contract and can be regenerated from public source data.

Claims still not fully proven:

- No live GSM8K model/adaptor results are claimed yet; running the workload requires a live OpenAI-compatible server.
- TLDR and PTS adapters are not math-quality candidates. A math-capable adapter should be selected or trained before using GSM8K to claim adapter superiority.
- Production-scale concurrency still requires an external longer-running GPU/server run.
