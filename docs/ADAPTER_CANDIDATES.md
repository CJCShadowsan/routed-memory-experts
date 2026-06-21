# Adapter Candidates

This table distinguishes adapters that prove serving mechanics from adapters that can support quality-superiority claims.

| Served id | Adapter | Base model | Rank / runtime notes | License status | Intended domain | Serving status | Quality status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tldr` | `phh/Qwen3-0.6B-TLDR-Lora` | `Qwen/Qwen3-0.6B` | Works with vLLM-Metal and CUDA vLLM; served in current artifacts | External Hub license must be reviewed before redistribution claims | summarization/general | Proven serving on Metal and CUDA | Not quality-proven; CUDA base-vs-adapter tied on six items |
| `pts` | `codelion/Qwen3-0.6B-PTS-DPO-LoRA` | `Qwen/Qwen3-0.6B` | Requires `--max-lora-rank 64`; works in current multi-LoRA artifacts | External Hub license must be reviewed before redistribution claims | preference/DPO-style behavior | Proven serving on Metal and CUDA | Not quality-proven; no large public benchmark comparison yet |
| `math` | `tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora` | `Qwen/Qwen2.5-0.5B-Instruct` | PEFT LoRA, `--max-lora-rank 64`; loaded through local direct vLLM Python API on Mac | Apache-2.0 declared in model card metadata; still cite source before publication | grade-school math/GSM8K | Proven local direct-vLLM load; Kaggle CUDA script added | Not quality-proven; local 32-item direct-vLLM run had base 5/32 and adapter 3/32, so adapter did not beat base |
| rejected Qwen3 math candidate | `aokesem/qwen3-0.6B_gsm8k_lora` | `Qwen/Qwen3-0.6B` | vLLM-Mac failed to add adapter with `StopIteration`; checkpoint subpath was not loadable as an adapter | No license declared in model card metadata at inspection time | grade-school math/GSM8K | Rejected for now | Do not use until adapter packaging/license are fixed |

## Quality-superiority gate

Do not mark an adapter as quality-proven unless an artifact shows:

- at least 30 benchmark items;
- same prompts for base and adapter;
- `expert_accuracy > base_accuracy`;
- `expert_wins > expert_losses`;
- benchmark source/license/provenance documented according to `docs/PUBLIC_BENCHMARK_CONTRACT.md`.

## Next candidate work

1. Use `workloads/gsm8k_public_sample.jsonl` as the initial reviewed public benchmark sample.
2. Prefer `tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora` for reproducible mechanics, but do not claim it improves quality unless a larger run reverses the current local 32-item result.
3. Serve base + adapter through CUDA vLLM using `scripts/kaggle_cuda_gsm8k_vllm_public_benchmark.py`.
4. If quality remains worse than base, train/select a stronger math adapter and rerun the same benchmark gate.
5. Update this file from artifact values, not from qualitative impressions.
