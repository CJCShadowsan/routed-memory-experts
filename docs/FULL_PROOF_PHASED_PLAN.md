# Full Thesis Proof Phased Implementation Plan

> **For Hermes:** Execute this plan in bounded verified loops. Implement every locally feasible task first. When a task requires external hardware, credentials, paid services, long-running benchmark time, or upstream runtime support, produce an executable script/runbook and a machine-readable blocker or gap artifact instead of claiming completion.

**Goal:** Convert the routed memory-hierarchy expert systems repository from a bounded proof artifact into the strongest feasible evidence package for publication: public benchmark evidence, adapter-quality evidence, capacity evidence, and explicit external blockers.

**Architecture:** Keep the deterministic CI-safe proof harness as the control plane. Layer optional live model-serving proofs on top through OpenAI-compatible APIs, vLLM-Metal, CUDA vLLM, and future SGLang runners. Every phase writes machine-readable artifacts under `runs/`, updates the paper/progress ledger from those artifacts, and keeps unsupported claims explicitly marked as external or upstream-blocked.

**Tech Stack:** Python 3.11+, pytest, argparse CLI (`rme`), JSON/JSONL proof artifacts, local vLLM-Metal/MLX, hosted or external CUDA vLLM/SGLang, Kaggle notebooks for free NVIDIA GPU proofs.

---

## Phase 0: Baseline hygiene and current-state ledger

### Task 0.1: Confirm branch state and existing artifacts

**Objective:** Establish what is already proven before adding new work.

**Files:**
- Read: `docs/THESIS_PROGRESS.md`
- Read: `paper/routed-memory-experts.md`
- Read: `runs/*.json`

**Steps:**
1. Run `git status --short --branch`.
2. Run `rme validate-artifacts --path runs`.
3. Run `pytest -q`.
4. Record current proof status in `runs/proof-gap-ledger.json`.

**Acceptance:** Tests pass, artifacts validate, and the gap ledger separates `proven`, `proven_bounded`, `external_required`, and `blocked_upstream` claims.

### Task 0.2: Add a proof-gap ledger command

**Objective:** Make remaining work executable and machine-readable instead of prose-only.

**Files:**
- Create: `src/routed_memory_experts/proof_gap.py`
- Modify: `src/routed_memory_experts/cli.py`
- Modify: `src/routed_memory_experts/artifact_validation.py`
- Create: `tests/test_proof_gap.py`

**Steps:**
1. Add `summarize_proof_gaps()` that reads `runs/` and reports every thesis claim, status, evidence, remaining gap, and next action.
2. Add `rme summarize-proof-gaps --runs runs --output runs/proof-gap-ledger.json`.
3. Add an artifact schema for `proof-gap-ledger`.
4. Add tests for status classification and validation.
5. Run `pytest tests/test_proof_gap.py -q`.

**Acceptance:** The command writes a valid ledger artifact and test coverage proves at least one bounded proof, one external-required gap, and one upstream-blocked gap.

---

## Phase 1: Public benchmark readiness

### Task 1.1: Define benchmark workload contract

**Objective:** Make public benchmark inputs reproducible and auditable.

**Files:**
- Create: `docs/PUBLIC_BENCHMARK_CONTRACT.md`
- Optional create: `benchmarks/README.md`

**Contract:** Each benchmark JSONL item must include:
- `id`
- `source`
- `license`
- `domain`
- `prompt`
- `expected_contains` or externally scored answer fields
- `split`
- `provenance_url`

**Acceptance:** Docs explain what can and cannot be claimed from benchmark artifacts.

### Task 1.2: Add public benchmark smoke runner

**Objective:** Reuse the existing proof harness on a benchmark-shaped workload without weakening the normal deterministic tests.

**Files:**
- Modify or create: benchmark runner module if needed.
- Add tests for source/provenance enforcement.

**Acceptance:** A benchmark workload without provenance fails validation; a valid workload runs and emits JSON.

### Task 1.3: External full benchmark execution

**Objective:** Run a larger public benchmark on a live OpenAI-compatible server.

**External requirement:** Live model server with base model and compatible adapter(s), plus benchmark selection and license review.

**Acceptance:** Produce artifacts such as:
- `runs/public-benchmark-base-vs-adapter.json`
- `runs/public-benchmark-routed-proof.json`
- `runs/public-benchmark-summary.json`

**Stop condition:** If no live server or reviewed benchmark is available locally, provide `scripts/run-openai-public-benchmark.sh` and instructions instead of fabricating results.

---

## Phase 2: Adapter quality superiority

### Task 2.1: Adapter candidate table

**Objective:** Track which adapters are serving-mechanics candidates vs. quality candidates.

**Files:**
- Create: `docs/ADAPTER_CANDIDATES.md`

**Acceptance:** Each candidate has base model, adapter path, rank, license, intended domain, serving status, and quality status.

### Task 2.2: Base-vs-adapter benchmark gate

**Objective:** Require adapter-quality claims to clear a larger fair comparison.

**Gate:** A quality win requires at least 30 items, expert wins > losses, and expert accuracy > base accuracy.

**Acceptance:** `runs/proof-gap-ledger.json` continues to mark this external-required until such an artifact exists.

---

## Phase 3: Production-scale concurrency and capacity

### Task 3.1: Capacity benchmark script

**Objective:** Provide an external runner for long concurrency tests.

**Files:**
- Create: `scripts/run-openai-public-benchmark.sh`

**Acceptance:** The script accepts base URL, base model, expert model, workload, request count, and concurrency, then emits comparison, proof, concurrency, validation, and gap-ledger artifacts.

### Task 3.2: External capacity run

**External requirement:** Controlled NVIDIA GPU or long-running hosted GPU environment.

**Acceptance:** A production-scale claim requires >=1000 requests, 0 request errors, p50/p95/p99 latency, throughput, GPU/runtime metadata, and a preserved server log.

---

## Phase 4: Runtime parity and upstream blockers

### Task 4.1: Metal CPU LoRA cache-tier blocker tracking

**Objective:** Keep the Metal limitation explicit and non-regressing.

**Acceptance:** The ledger marks vLLM-Metal `max_cpu_loras > max_loras` as `blocked_upstream` until the runtime supports it or a local implementation lands.

### Task 4.2: CUDA/SGLang parity runbook

**Objective:** Make future external proof runs repeatable.

**Files:**
- Update: `docs/EXTERNAL_PROOF_RUNBOOK.md`

**Acceptance:** Instructions distinguish Kaggle smoke proof, controlled CUDA proof, and SGLang parity proof.

---

## Phase 5: Publication hardening

### Task 5.1: Paper evidence table refresh

**Objective:** Ensure every paper claim has an artifact or a blocker.

**Files:**
- Modify: `paper/routed-memory-experts.md`
- Modify: `docs/THESIS_PROGRESS.md`

**Acceptance:** No paper claim says “proven” unless the ledger has an artifact-backed status.

### Task 5.2: arXiv conversion

**Objective:** Convert the markdown preprint into LaTeX only after benchmark/capacity claims are settled.

**External requirement:** Citation selection and final target venue style.

**Acceptance:** LaTeX source builds locally and cites all external datasets, runtimes, models, and adapters.

---

## Current local stop condition

Local execution should continue until these conditions are true:

1. `rme summarize-proof-gaps` exists and writes `runs/proof-gap-ledger.json`.
2. The proof-gap ledger validates through `rme validate-artifacts --path runs`.
3. `pytest -q` passes.
4. `scripts/run-openai-public-benchmark.sh` exists for the next external benchmark/capacity run.
5. `docs/EXTERNAL_PROOF_RUNBOOK.md` explains how to run the remaining online/GPU phases.
6. The paper/progress docs do not overclaim adapter quality, public benchmark performance, production-scale concurrency, or Metal CPU LoRA tiering.

After that, the remaining work is genuinely external: benchmark selection/licensing, compatible high-quality adapters, controlled GPU capacity runs, or upstream vLLM-Metal implementation work.
