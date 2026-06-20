# ADR 0001: Use a Routed Memory-Hierarchy Architecture

## Status

Accepted

## Context

The project thesis is that future AI systems will combine routing with memory hierarchy. The naive interpretation — streaming arbitrary full model weights from SSD every token — is not feasible for low-latency dense inference because SSD bandwidth is far below GPU HBM bandwidth. Evidence from MoE, LoRA, PagedAttention, and offload systems supports a narrower architecture.

## Decision

The repository will implement and evaluate a routed memory-hierarchy system with hot/HBM, warm/DRAM, and cold/NVMe tiers; a router; expert/agent registry; generalist baseline; and proof harness. The first implementation uses deterministic experts to prove the control plane. Later phases add real local models and LoRA/adapters.

## Consequences

Positive: immediate proof harness, lightweight CI, pluggable neural backends, measurable claims. Negative: Phase 1 does not prove neural quality and deterministic experts can overstate clean-domain behavior.

## Non-goals

Token-level SSD streaming of dense full-model weights; claiming production LLM quality from deterministic experts; hiding fallback/escalation cost.
