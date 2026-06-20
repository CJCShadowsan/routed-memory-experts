# Routed Memory Experts Implementation Plan

> **For Hermes:** Execute this plan in bounded iterations. After every iteration, run the proof harness and update `docs/THESIS_PROGRESS.md` with evidence vs. the paper’s claims.

**Goal:** Build a public proof repository that demonstrates routed memory-hierarchy expert AI systems end-to-end on real-world workloads.

**Architecture:** Start with a deterministic control-plane MVP, then replace stand-ins with real local models/adapters and serving runtimes. Each phase must preserve measurable proof artifacts: route decisions, cache tier behavior, quality, latency, and regret.

**Tech Stack:** Python, pytest, JSONL workloads, vLLM/SGLang/llama.cpp in later phases, LoRA/adapters, NVMe-backed artifact store.

---

## Phase 1: Control-plane MVP

Acceptance:

```bash
pytest -q
rme prove --workload workloads/real_world_v1.jsonl --experts experts --output runs/proof.json
```

Expected: routed accuracy >= 0.80, route regret <= 0.20, routed accuracy > generalist baseline, proof JSON written, cold and hot tier behavior observed.

## Phase 2: Real local model baseline

Add optional llama.cpp or transformers backend. Keep deterministic mode as CI fallback. Measure latency and answer correctness on the same workload.

## Phase 3: LoRA/adapters as micro-experts

Add adapter manifest, compatibility checks, vLLM/LoRAX serving path, adapter load-time metrics, and GPU/CPU/NVMe cache policy.

## Phase 4: Retrieval/tool specialists

Add vector/retrieval and tool-specialist interfaces. Route prompts to model, adapter, retrieval, or tool. Include prompt-injection and tool-safety tests.

## Phase 5: Router learning and regret evaluation

Replace keyword router with trainable classifier or RouteLLM-style policy. Track regret vs oracle expert and fallback model.

## Phase 6: Real-world benchmark expansion

Add public benchmark slices where license permits. Include code, Kubernetes/devops, finance, security, medical-literature, and general prompts.

## Phase 7: Production-serving experiment

Deploy vLLM/SGLang locally or in container, enable LoRA serving, and exercise route -> adapter load -> generation -> metric recording.

## Iteration Loop

Each iteration: observe current proof results; pick the highest-impact failing or missing claim; implement one bounded improvement; run tests and proof harness; update `docs/THESIS_PROGRESS.md`; commit. Stop only when all paper claims have direct proof artifacts or when a claim is falsified and the paper is revised accordingly.
