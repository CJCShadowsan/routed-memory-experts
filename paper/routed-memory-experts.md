# Routed Memory-Hierarchy Expert Systems: A Proof-Oriented Study of Local Routing, Adapter Serving, and Capacity Tiers

**Author:** CJCShadowsan / Routed Memory Experts contributors
**Repository:** `CJCShadowsan/routed-memory-experts`
**Status:** arXiv-style preprint draft; not peer reviewed; empirical claims are limited to the included artifacts and current local hardware.

## Abstract

Interactive AI serving is often described as a contest between ever-larger monolithic models and smaller specialized systems. This paper studies a narrower and falsifiable thesis: useful specialization is more likely to emerge from routed memory-hierarchy expert systems than from per-token streaming of arbitrary dense model weights from SSD. In this architecture, a resident base model provides general competence, small experts such as LoRA adapters or prompt/tool/retrieval specialists provide local specialization, and a router selects the appropriate expert at request, session, document, or tool-call granularity. Hot, warm, and cold tiers correspond to accelerator memory, host memory, and local NVMe/object storage, but the routing granularity must be coarse enough that cold movement does not occur on every generated token. A further implication is infrastructural: if useful work can be decomposed across many small routed experts, then some deployments may scale across grids of smaller, lower-power accelerators instead of requiring a single large cluster of frontier-class GPUs.

We implement a proof-oriented repository that measures this architecture through deterministic control-plane tests, locality simulation for 1,000 agent-owned experts, a learned-router comparison, local neural context-injection through Ollama, Apple Silicon vLLM-Metal/MLX serving, and hosted CUDA vLLM serving on Kaggle T4. The strongest systems result is that CUDA vLLM served LoRA adapters while accepting the upstream-style CPU LoRA cache-tier configuration that vLLM-Metal currently rejects. The strongest public-benchmark result is a 32-item GSM8K CUDA run in which `tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora` narrowly beat `Qwen/Qwen2.5-0.5B-Instruct`, with expert accuracy 12/32 versus base accuracy 11/32, 4 expert wins, 3 expert losses, and 25 ties. This meets the repository's threshold for a bounded adapter-quality win, but the effect size is small and adapter latency is substantially higher. The principal remaining unproven claims are production-scale concurrency/capacity and full upstream-style CPU LoRA cache tiering on Metal.

## Keywords

local AI serving; LoRA; adapter routing; vLLM; MLX; Apple Silicon; memory hierarchy; model routing; expert systems; retrieval-augmented generation; proof-oriented research repositories

## 1. Introduction

The intuitive appeal of “micro-models streamed from SSD” is clear: local storage is large and cheap, while accelerator memory is scarce. If an AI system could keep thousands of specialized neural components on local storage and activate only the right one for each request, then it might achieve better quality/cost tradeoffs than a single always-on frontier-scale model. The naive version of this idea, however, is technically implausible for interactive generation: dense model weights cannot be fetched from SSD at token granularity without running into bandwidth and latency gaps relative to HBM or unified accelerator memory.

This paper refines the idea into a routed memory-hierarchy architecture. The unit of movement is not an arbitrary dense model per token. The practical units are smaller and coarser: LoRA/adapters, retrieval indexes, tools, prompt specialists, small local models, and session-level agent state. The router selects one or more experts before or during a request; the serving layer keeps frequently used experts hot; and the evaluation layer records route regret, accuracy, fallback, cache behavior, and latency.

The contribution is not a claim that this repository has solved production multi-adapter serving or proved a complete cost model for distributed inference. Instead, it provides a falsifiable implementation path and an evidence ledger. Each claim is backed by an executable artifact where possible, and unproven claims are explicitly listed as blockers.

## 2. Background and Motivation

### 2.1 Memory hierarchy

Modern AI serving is constrained by a hierarchy of storage and memory. Accelerator memory or unified GPU memory offers the highest bandwidth and lowest latency but is capacity-limited. Host DRAM is larger but slower and may require transfer or page-management overhead. NVMe storage is much larger and cheaper, but it is not a substitute for accelerator memory during per-token dense inference. Therefore, any architecture that relies on cold storage must move data at coarse granularity and exploit locality.

### 2.2 Sparse experts and MoE

Mixture-of-experts architectures show that conditional compute can be effective. However, production MoE systems generally keep active experts resident across accelerator or distributed memory rather than loading arbitrary experts from SSD for each token. This supports the general principle of routing, but it does not validate naive SSD-per-token streaming.

### 2.3 LoRA and adapters

LoRA adapters are attractive as micro-expert units because they are small relative to base models and can be attached to a shared base model. A serving system can expose adapters as named models or request-time modules. In principle, this makes adapters a useful unit for hot/warm/cold tiering: a small active set is hot, a larger candidate set is warm, and long-tail adapters are cold.

### 2.4 Retrieval and tools

Not every specialization should be encoded in weights. Retrieval indexes and tools can handle dynamic knowledge, exact records, APIs, and procedural actions. A mature routed system should route among models, adapters, retrieval, and tools rather than forcing every specialist into a neural adapter.

### 2.5 vLLM-Metal and Apple Silicon

The repository initially treated CUDA vLLM/SGLang as the only production-grade adapter serving path. That was too narrow. vLLM now has a community-maintained Apple Silicon path through vLLM-Metal, which uses MLX and Metal. This enables local OpenAI-compatible serving on the hardware used for the present study. The important caveat is that feature parity with upstream CUDA vLLM is not complete; in particular, the current vLLM-Metal LoRA manager does not implement the upstream `max_cpu_loras > max_loras` cache tier.

### 2.6 Infrastructure distribution and smaller accelerators

The memory-hierarchy framing also changes the infrastructure question. A monolithic frontier model tends to imply centralized clusters, high-end accelerators, dense networking, and large power/cooling envelopes. A routed expert system suggests a different scaling surface: many smaller models, adapters, retrieval shards, or tool specialists can be placed near the requests or tenants that use them most. In a grid-like environment of many smaller GPUs or neural accelerators, the scheduler can keep local hot experts resident on inexpensive, lower-power cards while routing only misses, escalations, or high-complexity requests to larger shared systems.

This is not automatically cheaper. Routing errors, underutilized shards, cold-load latency, inter-node transfer, operational complexity, and weaker batching can erase the benefit. However, the architecture makes a cost/power hypothesis testable: compare a centralized large-GPU serving stack against a distributed routed stack on the same workload, measuring accuracy, p50/p95 latency, throughput per watt, accelerator utilization, cold-load rate, network transfer, and capital cost per successful request. The current repository contains only early evidence for the locality precondition (`runs/fleet.json`) and bounded small-GPU serving on Kaggle T4; it does not yet prove an end-to-end infrastructure cost advantage.

## 3. Thesis and Falsifiable Claims

The thesis is that AI serving can be framed as a routed memory-hierarchy problem. A resident base model and a set of smaller specialists can improve quality/cost tradeoffs when the router selects the right specialist at coarse granularity and the serving runtime manages hot/warm/cold tiers without destroying latency or batching efficiency. The infrastructure corollary is that some workloads may be served more economically by distributing many small hot experts across power-efficient accelerators than by centralizing all inference behind a small number of very large accelerators. This corollary is conditional on workload locality, routing accuracy, acceptable tail latency, and high enough utilization of the distributed devices.

| Claim | Current status | Evidence artifact | Remaining threat |
| --- | --- | --- | --- |
| Dense full-weight SSD streaming per token is a poor fit for interactive serving | Analytical claim, accepted as design constraint | Paper discussion | Needs quantitative hardware table for publication-grade rigor |
| Routing plus small experts can be measured end-to-end | Proven in deterministic harness | `runs/proof.json`, `runs/benchmark-proof.json` | Deterministic experts overstate controllability |
| Hot/warm/cold expert-cache behavior is observable | Proven in deterministic harness | `runs/proof.json` | Expert JSON files are stand-ins for real model/adaptor artifacts |
| 1,000 agent-owned experts are plausible when request locality exists | Simulated | `runs/fleet.json` | Simulation assumptions may not match real traffic |
| Learned routing can beat keyword routing | Proven on small synonym fixture | `runs/router-comparison.json` | Needs larger public benchmark |
| A real local neural backend can consume routed context | Proven via Ollama | `runs/ollama-proof.json` | Context injection is not adapter-weight specialization |
| Apple Silicon can serve vLLM-compatible local base model | Proven via vLLM-Metal | `runs/vllm-metal-proof.json` | Model is small and workload is bounded |
| Apple Silicon can load and serve real LoRA adapters through vLLM-Metal | Proven for two adapters | `runs/vllm-metal-lora-proof.json`, `runs/vllm-metal-multi-lora-models.json`, `runs/vllm-metal-manifest-proof.json` | Does not prove Metal CPU LoRA cache tiering |
| CUDA vLLM accepts upstream-style CPU LoRA cache tiering with more CPU-resident adapters than active adapters | Proven on Kaggle T4 | `runs/cuda-vllm-models.json`, `runs/kaggle-vllm-startup-v0-xformers.log` | Small model, hosted-notebook runtime, and only two configured adapters |
| CUDA vLLM can serve multiple LoRA adapters through the same routed proof harness | Proven on bounded workload | `runs/cuda-vllm-tldr-proof.json`, `runs/cuda-vllm-pts-proof.json`, `runs/cuda-vllm-concurrency.json` | Public adapters prove serving mechanics, not broad domain superiority |
| A math LoRA can beat its base model on a reviewed public benchmark sample | Proven on bounded 32-item GSM8K CUDA run | `runs/cuda-vllm-gsm8k-public-openai-benchmark.json` | Narrow +1-item win and higher adapter latency; needs full benchmark/repeated runs for stronger claims |
| Distributed small-accelerator grids can reduce cost or power versus centralized large-GPU serving | Hypothesis; not yet proven | `runs/fleet.json`, bounded Kaggle T4 artifacts | Requires a hardware/cost/power benchmark with utilization, watts, network, and tail-latency measurements |
| Full upstream-style adapter cache tiering works on Metal | Not proven; currently falsified for `max_cpu_loras > max_loras` | vLLM-Metal error log summarized in `docs/THESIS_PROGRESS.md` | Requires future vLLM-Metal support or a different Metal serving design |

## 4. Architecture

The proposed architecture has five planes.

### 4.1 Router plane

The router maps a request to a domain, confidence, and routing explanation. The current implementation includes a deterministic keyword router and a simple learned Naive Bayes router. Future implementations can use classifier models, RouteLLM-style policies, DSPy-optimized routers, or cost-aware cascades.

### 4.2 Expert plane

Experts can be JSON-backed deterministic specialists, prompt/context specialists, local models, LoRA adapters, retrieval indexes, or tools. The repository currently proves deterministic specialists, Ollama context specialists, vLLM-Metal LoRA serving, and CUDA vLLM LoRA serving.

### 4.3 Memory plane

The proof harness models hot, warm, and cold tiers. Hot approximates accelerator-resident state; warm approximates host memory; cold approximates local disk/NVMe. The deterministic cache records cold loads, warm hits, and hot hits. The real vLLM-Metal path currently proves loaded LoRA serving, not full CPU-to-accelerator adapter cache tiering. The CUDA vLLM path proves that upstream CUDA vLLM accepts a configuration with `max_cpu_loras` larger than `max_loras`, meaning the runtime can expose a larger CPU-resident adapter tier than the active accelerator-resident adapter set.

### 4.4 Serving plane

The serving plane exposes OpenAI-compatible local APIs. The repository's `rme prove-openai` command sends routed prompts to any compatible server. This makes the harness usable with vLLM-Metal, CUDA vLLM, SGLang, llama.cpp servers, or other compatible local runtimes.

### 4.5 Infrastructure placement plane

The same router that chooses an expert can also inform placement. In a centralized deployment, hot experts may share a large GPU pool. In a distributed deployment, experts can be pinned to smaller accelerators near a tenant, region, edge site, or agent cohort. The scheduler's objective is then not only correctness and latency, but also locality, power, utilization, and escalation cost. A useful implementation would record where each request ran, whether it was served by a local small accelerator or escalated to a larger pool, and how much data moved across the network. This paper treats that placement layer as a design implication and future benchmark target, not as a completed result.

### 4.6 Evaluation plane

Every proof writes JSON artifacts. Metrics include accuracy, baseline accuracy, route regret, fallback count, cache hits, cold loads, p50/p95 latency, model comparison wins/losses/ties, concurrency success/error count, and throughput.

## 5. Methodology

The repository follows a staged proof design.

1. **Control-plane proof:** Use deterministic experts to prove that routing, cache tiers, fallback, and metric recording work without neural nondeterminism.
2. **Fleet locality simulation:** Simulate 1,000 agents and measure whether a locality-aware request stream can keep most requests hot.
3. **Learned-router proof:** Train a small classifier on labeled examples and evaluate it against a keyword router on synonym-heavy prompts.
4. **Local neural proof:** Route to expert context, then ask a local Ollama model to answer using that context.
5. **vLLM-Metal base proof:** Serve `Qwen/Qwen3-0.6B` locally on Apple Silicon and exercise it through the same OpenAI-compatible proof harness.
6. **vLLM-Metal LoRA proof:** Serve `phh/Qwen3-0.6B-TLDR-Lora` as model `tldr` and prove that the harness can route through a real loaded adapter.
7. **Base-vs-LoRA comparison and concurrency:** Compare the base model and served adapter on identical routed prompts and measure concurrent serving latency where the live server is available.
8. **Multi-adapter manifest proof:** Serve two LoRA adapters simultaneously, route domains to served adapter IDs through a manifest, and record manifest-routed proof/concurrency artifacts.
9. **CUDA vLLM cache-tier proof:** Run the same OpenAI-compatible harness on Kaggle T4 with CUDA vLLM, `--max-loras 2`, `--max-cpu-loras 4`, and two Qwen3-0.6B LoRA adapters to test the upstream CPU LoRA tier that Metal rejects.

The benchmark fixtures are intentionally transparent JSONL rather than hidden or proprietary. This makes the results reproducible but also limits external validity.

## 6. Current Results

The latest committed artifacts show the following.

### 6.1 Deterministic original workload

`runs/proof.json` records a 12-item original workload. The routed system substantially outperforms the generalist baseline and records cold/warm/hot cache behavior.

### 6.2 Expanded deterministic benchmark

`runs/benchmark-proof.json` records a 32-item expanded benchmark fixture. In the current run, deterministic routed accuracy is 0.9375, baseline accuracy is 0.0625, and route regret is 0.0. This is not a public benchmark result; it is a larger internal proof fixture designed to exercise the domains covered by the included specialists.

### 6.3 Fleet simulation

`runs/fleet.json` simulates 1,000 agents over 5,000 requests. The recorded hit rate is above the configured acceptance threshold, supporting the claim that locality is mandatory for agent-owned hot models.

### 6.4 Learned router

`runs/router-comparison.json` shows the learned router beating the keyword router on a small synonym-heavy development set. This proves the code path and the concept, not production router robustness.

### 6.5 Ollama local neural proof

`runs/ollama-proof.json` shows a local model consuming routed context. This is a real neural proof, but it is not a LoRA weight proof.

### 6.6 vLLM-Metal base, LoRA, and multi-adapter proof

`runs/vllm-metal-models.json` shows served models from the local vLLM-Metal server. The first live proof used base `Qwen/Qwen3-0.6B` and adapter-served model `tldr`, whose root is `phh/Qwen3-0.6B-TLDR-Lora` and parent is `Qwen/Qwen3-0.6B`. `runs/vllm-metal-lora-proof.json` records routed proof results through the adapter model.

The subsequent multi-adapter proof loaded both `tldr=phh/Qwen3-0.6B-TLDR-Lora` and `pts=codelion/Qwen3-0.6B-PTS-DPO-LoRA` under the same base model. The `pts` adapter required `--max-lora-rank 64`; the default rank 16 rejected it. `runs/vllm-metal-multi-lora-models.json` records the served base plus two LoRA models, and `runs/vllm-metal-manifest-proof.json` records a domain-routed manifest proof that maps finance/general/medical-literature to `tldr` and python/kubernetes/security to `pts`.

### 6.7 Negative result: Metal LoRA CPU cache tier

When started with `--max-loras 2 --max-cpu-loras 4`, vLLM-Metal failed with a `NotImplementedError` stating that Metal LoRA does not implement the upstream `max_cpu_loras > max_loras` cache tier and that every added adapter is activated immediately. This is a useful negative result: Apple Silicon currently proves LoRA serving, but not full upstream-style LoRA cache tiering.

### 6.8 CUDA vLLM LoRA cache-tier proof

The Kaggle CUDA run used `vllm==0.10.2`, `transformers==4.57.6`, `Qwen/Qwen3-0.6B`, and two served adapters: `tldr=phh/Qwen3-0.6B-TLDR-Lora` and `pts=codelion/Qwen3-0.6B-PTS-DPO-LoRA`. The server was started with `--enable-lora --max-loras 2 --max-cpu-loras 4 --max-lora-rank 64` on a Tesla T4. `runs/cuda-vllm-models.json` records three served models: the base model plus both LoRA adapter model IDs, with each adapter parented to `Qwen/Qwen3-0.6B`.

The bounded six-item routed proofs both passed: `runs/cuda-vllm-tldr-proof.json` records accuracy 1.0 with p50 latency 4316.7 ms and p95 latency 4513.4 ms; `runs/cuda-vllm-pts-proof.json` records accuracy 1.0 with p50 latency 4477.6 ms and p95 latency 5597.6 ms. `runs/cuda-vllm-base-vs-tldr.json` records base accuracy 1.0 and expert accuracy 1.0 on the same six items, with six ties and no wins/losses. This means the adapters did not demonstrate quality superiority on this small fixture; they demonstrated that multiple LoRA adapters can be loaded, named, routed, and exercised through the same proof interface.

The concurrency proof `runs/cuda-vllm-concurrency.json` ran 24 requests at concurrency 4. It recorded 24 successes, 0 errors, accuracy 0.9167, throughput 0.653 requests/s, p50 latency 4974.3 ms, p95 latency 9701.6 ms, and p99 latency 10512.5 ms. This supports the serving-mechanics and cache-tier claims but remains a small hosted-notebook benchmark rather than a production capacity result.

### 6.9 CUDA GSM8K public benchmark proof

The dedicated GSM8K CUDA run used `vllm==0.10.2`, `Qwen/Qwen2.5-0.5B-Instruct`, and `math=tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora` on Kaggle T4. The runner used the non-leaky `benchmark-public-openai` path, which does not include ground-truth `expected_contains` labels in prompts. `runs/cuda-vllm-gsm8k-public-openai-benchmark.json` records 32 public GSM8K items: base accuracy 0.34375 (11/32), expert accuracy 0.375 (12/32), 4 expert wins, 3 expert losses, and 25 ties. This satisfies the repository's current `>=30` item threshold for a bounded adapter-quality win, but it is a narrow +1-item result rather than a large margin.

The same run wrote `runs/cuda-vllm-gsm8k-math-proof.json`, with 13/32 correct routed math responses through the `math` adapter, and `runs/cuda-vllm-gsm8k-concurrency.json`, with 32/32 successful concurrent requests, 0 errors, throughput 0.286 requests/s, p50 latency 11407 ms, p95 latency 27216 ms, and p99 latency 27291 ms at concurrency 4. These artifacts move the adapter-quality gap to proven in the ledger, but production-scale concurrency remains open because the run is still a small hosted-notebook benchmark.

## 7. Reproducibility

Install the Python package for deterministic proofs:

```bash
uv venv --python 3.11
. .venv/bin/activate
uv pip install -e '.[dev]'
pytest -q
```

Run deterministic proofs:

```bash
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
rme prove --workload workloads/benchmark_expanded_v1.jsonl --experts experts --output runs/benchmark-proof.json
rme simulate-fleet --agents 1000 --requests 5000 --hot-capacity 128 --locality-window 64 --output runs/fleet.json
rme compare-routers --train workloads/router_train_v1.jsonl --dev workloads/router_dev_v1.jsonl --output runs/router-comparison.json
rme validate-artifacts --path runs
```

Install and run vLLM-Metal on Apple Silicon:

```bash
scripts/install-vllm-metal.sh
scripts/start-vllm-metal-lora.sh
```

In another shell:

```bash
rme prove-openai --base-url http://127.0.0.1:8000/v1 --model tldr --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-lora-proof.json --limit 6
rme compare-openai-models --base-url http://127.0.0.1:8000/v1 --base-model Qwen/Qwen3-0.6B --expert-model tldr --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-base-vs-lora.json --limit 6
rme benchmark-openai-concurrency --base-url http://127.0.0.1:8000/v1 --model tldr --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-concurrency.json --requests 12 --concurrency 3
rme prove-openai --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-0.6B --adapter-manifest adapters/vllm_metal_manifest.json --workload workloads/real_world_v1.jsonl --experts experts --output runs/vllm-metal-manifest-proof.json --limit 6
```

Run the hosted CUDA proof on Kaggle with GPU enabled and internet enabled:

```bash
%cd /kaggle/working
!git clone https://github.com/CJCShadowsan/routed-memory-experts.git
%cd routed-memory-experts
!python scripts/kaggle_cuda_vllm_proof.py
```

The CUDA runner writes `runs/cuda-vllm-models.json`, `runs/cuda-vllm-tldr-proof.json`, `runs/cuda-vllm-pts-proof.json`, `runs/cuda-vllm-base-vs-tldr.json`, and `runs/cuda-vllm-concurrency.json`.

## 8. Limitations and Threats to Validity

The largest limitation is benchmark scale. The workload fixtures are transparent and useful for engineering, and the GSM8K run is a reviewed public-benchmark sample, but the current strongest adapter-quality result is still only 32 items and a narrow one-item win. The second limitation is adapter quality breadth: the public TLDR and PTS adapters prove serving mechanics and multi-adapter routing, while only the Qwen2.5 GSM8K math adapter currently has a bounded public-benchmark win. The third limitation is concurrency scale. Current local and Kaggle concurrency runs are small and constrained by Apple Silicon memory, T4 memory, hosted-notebook time, and runtime tuning. The fourth limitation is platform parity: CUDA vLLM proves the upstream CPU LoRA cache-tier configuration, while vLLM-Metal still does not implement that mode. The fifth limitation is infrastructure economics: the paper argues that routed experts make distributed small-accelerator deployments plausible, but the repository does not yet measure watts, capital cost, network overhead, operational complexity, or utilization across a real multi-node grid.

These limitations are intentionally surfaced as falsification boundaries. The repository should not claim that all thesis claims are solved until large benchmark, multi-adapter, and cache-tier experiments are complete.

## 9. Related Work

This work is adjacent to mixture-of-experts models, model cascades, retrieval-augmented generation, tool-use agents, LoRA serving systems, vLLM paged attention, SGLang serving, llama.cpp local inference, MLX on Apple Silicon, and systems work on memory hierarchy. The unifying perspective here is not that any one technique dominates, but that local AI systems should be designed as routed memory hierarchies with explicit evaluation of route regret and cache behavior.

## 10. Conclusion

The current evidence supports a refined thesis: routed memory-hierarchy expert systems are plausible when specialization units are small, routing is coarse-grained, and evaluation records quality, regret, latency, and cache behavior. The repository now proves deterministic routing, simulated fleet locality, learned-router improvement, local neural context routing, Apple Silicon vLLM-Metal LoRA serving, multi-adapter serving, CUDA vLLM CPU LoRA cache-tier configuration with `max_cpu_loras > max_loras`, and a bounded 32-item public GSM8K adapter-quality win. It also motivates an infrastructure corollary: routed systems may let some workloads scale through many smaller, power-efficient accelerators distributed across tenants, edge sites, or grid-like clusters, reducing the need for a single massive buildout when locality is strong. That corollary remains a hypothesis until measured against centralized large-GPU baselines. The repository does not yet prove production-scale concurrency, end-to-end infrastructure cost superiority, or full Metal hot/warm/cold adapter tiering. Those remaining gaps are concrete and measurable, which is the central purpose of this proof-oriented research artifact.

## Appendix A: Artifact Map

- `runs/proof.json`: deterministic original workload proof.
- `runs/benchmark-proof.json`: deterministic expanded benchmark proof.
- `runs/fleet.json`: 1,000-agent locality simulation.
- `runs/router-comparison.json`: learned vs keyword router comparison.
- `runs/ollama-proof.json`: local neural context-injection proof.
- `runs/runtime-readiness.json`: runtime readiness and blocker state.
- `runs/vllm-metal-models.json`: OpenAI-compatible served model list with LoRA parent metadata.
- `runs/vllm-metal-proof.json`: base vLLM-Metal routed proof.
- `runs/vllm-metal-lora-proof.json`: LoRA vLLM-Metal routed proof.
- `runs/vllm-metal-base-vs-lora.json`: live base-vs-LoRA comparison when generated.
- `runs/vllm-metal-concurrency.json`: live concurrency benchmark when generated.
- `runs/vllm-metal-multi-lora-models.json`: live served-model list with two LoRA adapters.
- `runs/vllm-metal-manifest-proof.json`: domain-to-adapter manifest proof.
- `runs/vllm-metal-manifest-concurrency.json`: manifest-routed concurrent serving benchmark.
- `runs/cuda-vllm-models.json`: Kaggle CUDA vLLM served-model list with base model plus two LoRA adapters under `max_cpu_loras > max_loras`.
- `runs/cuda-vllm-tldr-proof.json`: Kaggle CUDA routed proof through the TLDR LoRA adapter.
- `runs/cuda-vllm-pts-proof.json`: Kaggle CUDA routed proof through the PTS LoRA adapter.
- `runs/cuda-vllm-base-vs-tldr.json`: Kaggle CUDA base-vs-adapter comparison.
- `runs/cuda-vllm-concurrency.json`: Kaggle CUDA concurrent routed serving benchmark.
- `runs/kaggle-vllm-startup-v0-xformers.log`: startup/runtime log for the successful CUDA vLLM proof.
- `runs/cuda-vllm-gsm8k-models.json`: Kaggle CUDA vLLM served-model list for Qwen2.5 base plus GSM8K math LoRA.
- `runs/cuda-vllm-gsm8k-public-openai-benchmark.json`: non-leaky 32-item public GSM8K base-vs-math-adapter comparison.
- `runs/cuda-vllm-gsm8k-math-proof.json`: routed GSM8K proof through the math adapter.
- `runs/cuda-vllm-gsm8k-concurrency.json`: bounded concurrent routed GSM8K serving benchmark.
- `runs/kaggle-vllm-gsm8k-startup-v0-xformers.log`: startup/runtime log for the successful CUDA GSM8K vLLM proof.

## Appendix B: Current Impossible or External Tasks

The following tasks are not fully completable in the current local-only setting unless additional assets or runtime support become available:

1. Proving upstream-style `max_cpu_loras > max_loras` tiering on vLLM-Metal, because the runtime currently raises `NotImplementedError` for that mode.
2. Proving production-scale CUDA vLLM/SGLang adapter cache behavior, because the current local host has no NVIDIA/CUDA GPU and the Kaggle proof is a bounded hosted-notebook run.
3. Strengthening broad public benchmark performance beyond the current 32-item GSM8K sample, because publication-grade claims require larger run budgets, repeated seeds or larger samples, and possibly stronger models/adapters.
4. Proving many high-quality domain-specific adapters, because compatible adapters must be found or trained and then evaluated fairly.
5. Proving distributed small-accelerator infrastructure economics, because that requires multi-node hardware, power telemetry, utilization traces, network measurements, and a centralized large-GPU baseline for comparison.
