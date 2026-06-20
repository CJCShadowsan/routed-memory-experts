# Continual Proof Loop

Prompt:

> Re-read `paper/routed-memory-experts.md`, `docs/IMPLEMENTATION_PLAN.md`, and latest `docs/THESIS_PROGRESS.md`. Run tests and proof harness. Pick the highest-impact unproven claim, implement one bounded improvement with tests, rerun proof, update thesis progress with evidence and closeness score, then commit. Stop only if all claims are proven/falsified or a blocker requires user input.

Guardrails: do not claim neural-serving proof from deterministic experts; do not claim SSD per-token feasibility unless measured; preserve deterministic CI path; record failed hypotheses honestly; use real workloads or clearly labeled fixtures with provenance.
