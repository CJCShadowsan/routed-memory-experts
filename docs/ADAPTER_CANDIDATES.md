# Adapter Candidates

This table distinguishes adapters that prove serving mechanics from adapters that can support quality-superiority claims.

| Served id | Adapter | Base model | Rank / runtime notes | License status | Intended domain | Serving status | Quality status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tldr` | `phh/Qwen3-0.6B-TLDR-Lora` | `Qwen/Qwen3-0.6B` | Works with vLLM-Metal and CUDA vLLM; served in current artifacts | External Hub license must be reviewed before redistribution claims | summarization/general | Proven serving on Metal and CUDA | Not quality-proven; CUDA base-vs-adapter tied on six items |
| `pts` | `codelion/Qwen3-0.6B-PTS-DPO-LoRA` | `Qwen/Qwen3-0.6B` | Requires `--max-lora-rank 64`; works in current multi-LoRA artifacts | External Hub license must be reviewed before redistribution claims | preference/DPO-style behavior | Proven serving on Metal and CUDA | Not quality-proven; no large public benchmark comparison yet |
| TBD math adapter | TBD compatible Qwen3-0.6B math/GSM8K adapter | `Qwen/Qwen3-0.6B` preferred for parity | Must fit vLLM LoRA rank limits or declare required `--max-lora-rank` | Must be reviewed before use | grade-school math/GSM8K | Not selected | Required before claiming adapter quality superiority on `workloads/gsm8k_public_sample.jsonl` |

## Quality-superiority gate

Do not mark an adapter as quality-proven unless an artifact shows:

- at least 30 benchmark items;
- same prompts for base and adapter;
- `expert_accuracy > base_accuracy`;
- `expert_wins > expert_losses`;
- benchmark source/license/provenance documented according to `docs/PUBLIC_BENCHMARK_CONTRACT.md`.

## Next candidate work

1. Use `workloads/gsm8k_public_sample.jsonl` as the initial reviewed public benchmark sample.
2. Find or train a math-capable adapter with compatible base model, rank, and license.
3. Serve base + adapter through vLLM-Metal or CUDA vLLM.
4. Run `scripts/run-openai-public-benchmark.sh --workload workloads/gsm8k_public_sample.jsonl`.
5. Update this file from artifact values, not from qualitative impressions.
