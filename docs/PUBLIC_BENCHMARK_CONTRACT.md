# Public Benchmark Contract

The repository's internal workloads are engineering fixtures. A result is only a public benchmark result when the input workload and scorer satisfy this contract.

## Required JSONL fields

Each line must be a JSON object with:

- `id`: stable benchmark item id.
- `source`: benchmark or dataset name, including version where available.
- `license`: dataset license or redistribution note.
- `provenance_url`: URL for the dataset, paper, card, or exact source record.
- `split`: benchmark split, such as `validation`, `test`, or `dev`.
- `domain`: route domain used by the harness.
- `prompt`: model prompt.
- `expected_contains`: exact evidence strings for the current harness scorer, or a documented alternate scorer.

Optional fields:

- `answer`: canonical answer when exact-string evidence is not enough.
- `choices`: multiple-choice options.
- `scorer`: scorer id if a future scorer is not `expected_contains`.
- `notes`: caveats about conversion from the source dataset.

## Claim levels

- **Smoke benchmark:** fewer than 30 items, useful for integration only.
- **Adapter-quality benchmark:** at least 30 items, same prompts for base and adapter, expert accuracy > base accuracy, expert wins > expert losses.
- **Publication-grade benchmark:** source/license reviewed, stable split, raw outputs preserved, scoring method documented, and enough items for credible confidence intervals.

## Non-goals

- Do not call an internal fixture a public benchmark.
- Do not claim adapter superiority from a serving-only proof.
- Do not hide ties: if base and adapter tie, report a tie.
- Do not redistribute benchmark items unless the license permits it.

## Selected initial public benchmark: GSM8K

The first reviewed public benchmark for the next external run is GSM8K (Grade School Math 8K):

- Dataset: `openai/gsm8k` on Hugging Face.
- Provenance: https://huggingface.co/datasets/openai/gsm8k
- License: MIT, as declared by the Hugging Face dataset card.
- Split: `test`.
- Task shape: grade-school math word problems with final numeric answers marked after `####`.
- RME conversion: `scripts/build-gsm8k-public-workload.py` extracts the final numeric answer and writes `expected_contains` for the existing exact-evidence scorer.
- Current sample: `workloads/gsm8k_public_sample.jsonl`, 64 records from test offset 0.

This benchmark satisfies the workload metadata contract and is appropriate for public benchmark smoke/comparison runs. It does **not** by itself prove TLDR/PTS adapter superiority; those are serving-mechanics adapters, not math-quality candidates. The committed CUDA GSM8K artifact with `tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora` provides a bounded 32-item adapter-quality win over its Qwen2.5 base, but the win is narrow and should be expanded before broad publication claims.

## Current status

The current repo now includes both the reviewed public benchmark sample workload and live CUDA public benchmark artifacts under `runs/cuda-vllm-gsm8k-*.json`. Future external runs should expand sample size, repeatability, and capacity measurements rather than treating the 32-item sample as a final benchmark result.
