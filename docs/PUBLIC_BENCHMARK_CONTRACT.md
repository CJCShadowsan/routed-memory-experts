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

## Current status

The current repo has public-runtime proof artifacts, but not public benchmark performance artifacts. The next external run should use `scripts/run-openai-public-benchmark.sh` with a reviewed workload JSONL and a live server.
