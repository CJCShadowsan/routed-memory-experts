# Kaggle CUDA vLLM proof for routed-memory-experts

Use this in a Kaggle Notebook with:

- Accelerator: GPU, preferably T4/P100 or better
- Internet: On

## Cell 1: Clone repo

```bash
!git clone https://github.com/CJCShadowsan/routed-memory-experts.git
%cd routed-memory-experts
```

## Cell 2: Run CUDA proof

```bash
!python scripts/kaggle_cuda_vllm_proof.py
```

The runner creates `.kaggle-venv/` and installs vLLM there before re-running
itself. This avoids modifying Kaggle/Colab's preloaded global Python image,
which can otherwise emit many unrelated dependency-conflict warnings for
packages such as TensorFlow, cuDF, BigFrames, Gradio, and Google ADK.

If you already ran an older version of this script and saw resolver warnings,
restart the Kaggle session, pull the latest repo, and rerun:

```bash
%cd /kaggle/working/routed-memory-experts
!git pull
!rm -rf .kaggle-venv
!python scripts/kaggle_cuda_vllm_proof.py
```

## Cell 3: Inspect artifacts

```bash
!ls -lh runs/cuda-*.json
!python - <<'PY'
import json, glob
for path in sorted(glob.glob('runs/cuda-*.json')):
    print('\n###', path)
    data=json.load(open(path))
    for key in ['workload_count','accuracy','base_accuracy','expert_accuracy','error_count','throughput_requests_per_second','p95_latency_ms']:
        if key in data:
            print(key, data[key])
PY
```

## Expected evidence

The key missing project proof is whether CUDA vLLM accepts:

```text
--max-cpu-loras 4 --max-loras 2
```

If the server starts and `/v1/models` lists base + adapters, the CUDA runtime supports the cache-tier configuration that vLLM-Metal currently rejects.

Download these artifacts and commit them back to the repo if successful:

- `runs/cuda-vllm-models.json`
- `runs/cuda-vllm-tldr-proof.json`
- `runs/cuda-vllm-pts-proof.json`
- `runs/cuda-vllm-base-vs-tldr.json`
- `runs/cuda-vllm-concurrency.json`
