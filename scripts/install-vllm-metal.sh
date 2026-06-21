#!/usr/bin/env bash
set -euo pipefail

URL="https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh"
echo "Installing vLLM-Metal from: ${URL}"
echo "Target virtualenv: ${HOME}/.venv-vllm-metal"
echo "This script follows upstream vLLM-Metal installation guidance for Apple Silicon."
curl -fsSL "${URL}" | bash
source "${HOME}/.venv-vllm-metal/bin/activate"
python - <<'PY'
import importlib.util, platform, sys
print('python', sys.version)
print('platform', sys.platform, platform.machine())
for module in ['vllm', 'vllm_metal', 'mlx', 'openai']:
    print(module, importlib.util.find_spec(module) is not None)
PY
