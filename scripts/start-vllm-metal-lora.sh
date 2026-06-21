#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.venv-vllm-metal/bin/activate"
export VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-0.5}"
export VLLM_METAL_USE_PAGED_ATTENTION="${VLLM_METAL_USE_PAGED_ATTENTION:-1}"
exec vllm serve Qwen/Qwen3-0.6B \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8000}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-1024}" \
  --enable-lora \
  --max-loras "${VLLM_MAX_LORAS:-2}" \
  --max-lora-rank "${VLLM_MAX_LORA_RANK:-64}" \
  --lora-modules \
    tldr=phh/Qwen3-0.6B-TLDR-Lora \
    pts=codelion/Qwen3-0.6B-PTS-DPO-LoRA
