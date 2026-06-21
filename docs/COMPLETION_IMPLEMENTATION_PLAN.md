# Routed Memory Experts Completion Implementation Plan

> **For Hermes:** Execute this plan task-by-task in verified loops. Each task must produce executable evidence or an explicit blocker artifact. Do not mark a claim complete from prose alone.

**Goal:** Finish every locally feasible task needed to turn the repository into a serious proof-oriented research artifact for routed memory-hierarchy expert AI systems, and identify only those remaining tasks that require unavailable hardware, unavailable compatible adapters, paid/external services, or time-consuming training outside the current session.

**Architecture:** The repository will keep a deterministic CI-safe core and layer optional live serving proofs on top. All live proofs write machine-readable JSON artifacts. The paper will cite the current artifact set and distinguish deterministic simulation, local neural context injection, Apple Silicon vLLM-Metal base serving, and Apple Silicon vLLM-Metal LoRA serving.

**Tech Stack:** Python standard library, pytest, JSON/JSONL fixtures, vLLM-Metal/MLX optional runtime in `~/.venv-vllm-metal`, OpenAI-compatible HTTP API, shell scripts for reproducibility.

---

## Acceptance state

The project is complete for the current hardware/session only when:

1. `pytest -q` passes.
2. Deterministic proof passes on `workloads/real_world_v1.jsonl` and expanded benchmark slices.
3. Runtime readiness reports Apple Silicon vLLM-Metal/MLX readiness when `~/.venv-vllm-metal` exists.
4. OpenAI-compatible proof works against a live vLLM-Metal server when that server is available.
5. Base-vs-LoRA comparison writes an artifact showing per-item win/loss/tie and latency delta.
6. Concurrent OpenAI-compatible benchmark writes throughput and p50/p95/p99 latency artifact.
7. Artifact schema validation catches missing required fields for all current `runs/*.json` artifacts.
8. Multi-adapter routing is implemented at the manifest/harness level and either proven with available adapters or blocked by a concrete runtime/hub compatibility reason.
9. The paper is arXiv-style, detailed, and honest about evidence tiers and remaining limitations.
10. Reproducibility scripts exist for install, start, proof, and validation flows.

---

## Task 1: Add expanded benchmark workload

**Objective:** Replace single tiny workload as the only evidence with a larger license-safe local fixture.

**Files:**
- Create: `workloads/benchmark_expanded_v1.jsonl`
- Test indirectly via existing loaders and proof commands.

**Steps:**
1. Add 30+ domain-diverse JSONL items across Kubernetes/devops, Python/code, security, finance, medical-literature, summarization/general.
2. Keep `expected_contains` factual and compact so deterministic and neural proofs can be scored by substring.
3. Run `rme prove --workload workloads/benchmark_expanded_v1.jsonl --experts experts --output runs/benchmark-proof.json`.
4. If deterministic experts do not cover new expected tokens, either improve expert fixtures minimally or mark as neural-only benchmark. Prefer deterministic pass for CI.

---

## Task 2: Implement OpenAI-compatible base-vs-expert comparison

**Objective:** Prove whether a served expert model/LoRA improves over the base model on the exact same routed prompts.

**Files:**
- Modify: `src/routed_memory_experts/openai_backend.py`
- Modify: `src/routed_memory_experts/cli.py`
- Create: `tests/test_openai_comparison.py`

**Implementation:**
1. Add dataclasses for `OpenAIModelComparisonRecord` and `OpenAIModelComparisonSummary`.
2. For each workload item, build the same routed expert prompt, call base model and expert model, score both, and capture latency.
3. Compute base accuracy, expert accuracy, expert win/loss/tie counts, p50/p95 latency for both.
4. Add CLI `rme compare-openai-models --base-model ... --expert-model ...`.
5. JSON artifact path defaults to `runs/openai-model-comparison.json`.

**Verification:**
- Unit-test summary serialization and win/loss/tie accounting with fake records.
- Live-run against vLLM-Metal when available.

---

## Task 3: Implement OpenAI-compatible concurrency benchmark

**Objective:** Measure p50/p95/p99 latency, throughput, and error rate for routed requests under concurrent clients.

**Files:**
- Create: `src/routed_memory_experts/concurrency.py`
- Modify: `src/routed_memory_experts/cli.py`
- Create: `tests/test_concurrency.py`

**Implementation:**
1. Use `concurrent.futures.ThreadPoolExecutor` with standard-library HTTP client.
2. Build prompts with the same routed context builder.
3. Parameters: `--requests`, `--concurrency`, `--model`, `--base-url`, `--workload`, `--experts`, `--output`.
4. Record request count, success count, error count, throughput, p50/p95/p99, accuracy on successful requests.

**Verification:**
- Unit-test percentile calculation and summary serialization.
- Live-run small benchmark against vLLM-Metal when available.

---

## Task 4: Implement artifact schema validation

**Objective:** Make proof artifacts reviewable and fail closed when required fields disappear.

**Files:**
- Create: `src/routed_memory_experts/artifact_validation.py`
- Modify: `src/routed_memory_experts/cli.py`
- Create: `tests/test_artifact_validation.py`

**Implementation:**
1. Map artifact families to required top-level keys.
2. Infer family from filename and/or present keys.
3. Validate every JSON file in a directory.
4. CLI `rme validate-artifacts --path runs` returns non-zero on validation failure.

**Verification:**
- Unit-test valid and invalid sample artifacts.
- Run against `runs/`.

---

## Task 5: Add adapter manifest and multi-adapter routed model support

**Objective:** Support many served adapter model IDs even if current hardware can only prove one adapter today.

**Files:**
- Create: `adapters/vllm_metal_manifest.json`
- Create: `src/routed_memory_experts/adapter_manifest.py`
- Modify: `src/routed_memory_experts/openai_backend.py`
- Modify: `src/routed_memory_experts/cli.py`
- Create: `tests/test_adapter_manifest.py`

**Implementation:**
1. Manifest fields: `base_model`, `adapters[]`, each adapter with `name`, `served_model`, `source`, `domains`, `status`, `notes`.
2. Add loader that maps workload domain -> served model, falling back to base model or default expert model.
3. Add CLI option `--adapter-manifest` to `prove-openai` and `benchmark-openai-concurrency` so the model can vary by route.
4. Add `rme inspect-adapter-manifest` to print route map and readiness.

**Verification:**
- Unit-test domain-to-model resolution.
- Live-run with current single `tldr` adapter manifest.
- Attempt second adapter only if compatible public adapter is available; otherwise write blocker to progress log.

---

## Task 6: Add reproducibility scripts

**Objective:** Give reviewers exact commands for environment setup and proof reproduction.

**Files:**
- Create: `scripts/install-vllm-metal.sh`
- Create: `scripts/start-vllm-metal-lora.sh`
- Create: `scripts/run-local-proofs.sh`
- Create: `scripts/validate-artifacts.sh`

**Implementation:**
1. Scripts must be safe, explicit, and not hide destructive cleanup.
2. `install-vllm-metal.sh` prints upstream URL and installs to `~/.venv-vllm-metal`.
3. `start-vllm-metal-lora.sh` starts the proven Qwen3 + TLDR LoRA server.
4. `run-local-proofs.sh` runs deterministic proofs and skips live OpenAI proof if server health check fails.
5. `validate-artifacts.sh` calls `rme validate-artifacts --path runs`.

---

## Task 7: Rewrite the paper to arXiv-ready detail

**Objective:** Replace the short thesis note with a paper-like manuscript.

**Files:**
- Modify: `paper/routed-memory-experts.md`

**Required sections:**
1. Title, abstract, keywords.
2. Introduction and problem statement.
3. Background: HBM/DRAM/NVMe hierarchy, MoE, LoRA/adapters, RAG/tools, vLLM/SGLang/MLX.
4. Claims and falsifiability table.
5. Architecture and algorithms.
6. Experimental methodology.
7. Results from current artifacts.
8. Limitations and threats to validity.
9. Reproducibility instructions.
10. Related work discussion.
11. Conclusion.
12. Appendix with artifact map and command transcripts.

**Honesty rules:**
- Do not claim peer-reviewed/arXiv submission.
- Do not overstate curated benchmark results as public benchmark wins.
- Explicitly state that Metal does not yet prove upstream `max_cpu_loras > max_loras` tiering.

---

## Task 8: Run live vLLM-Metal loop if possible

**Objective:** Complete every live proof possible on current Apple Silicon host.

**Steps:**
1. Start LoRA server with `scripts/start-vllm-metal-lora.sh`.
2. Health-check `/health` and `/v1/models`.
3. Run:
   - `rme prove-openai --model tldr ...`
   - `rme compare-openai-models --base-model Qwen/Qwen3-0.6B --expert-model tldr ...`
   - `rme benchmark-openai-concurrency --model tldr --requests 12 --concurrency 3 ...`
4. Stop server.
5. If startup fails, record exact blocker and still commit all deterministic work.

---

## Task 9: Final verification and push

**Objective:** Leave repository review-ready.

**Commands:**
```bash
pytest -q
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
rme prove --workload workloads/benchmark_expanded_v1.jsonl --experts experts --output runs/benchmark-proof.json
rme simulate-fleet --agents 1000 --requests 5000 --hot-capacity 128 --locality-window 64 --output runs/fleet.json
rme compare-routers --train workloads/router_train_v1.jsonl --dev workloads/router_dev_v1.jsonl --output runs/router-comparison.json
rme check-runtimes --output runs/runtime-readiness.json
rme validate-artifacts --path runs
```

Then commit and push. Final report must list completed tasks and remaining impossible tasks only.
