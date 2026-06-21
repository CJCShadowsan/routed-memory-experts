#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.venv-vllm-metal/bin/activate"
export VLLM_METAL_MEMORY_FRACTION="${VLLM_METAL_MEMORY_FRACTION:-0.5}"
export VLLM_METAL_USE_PAGED_ATTENTION="${VLLM_METAL_USE_PAGED_ATTENTION:-1}"
exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8000}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-1024}" \
  --enable-lora \
  --max-loras "${VLLM_MAX_LORAS:-1}" \
  --max-lora-rank "${VLLM_MAX_LORA_RANK:-64}" \
  --lora-modules \
    math=tayyib-sayyid/qwen2.5-0.5b-gsm8k-lora
