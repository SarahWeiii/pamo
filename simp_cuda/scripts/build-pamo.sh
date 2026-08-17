#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${PAMO_DIST_DIR:-${ROOT}/dist}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build PaMO" >&2
  exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" && -z "${PAMO_BUILD_PYTHON:-}" ]]; then
  echo "Activate a virtualenv first, or set PAMO_BUILD_PYTHON to that interpreter" >&2
  exit 1
fi

PYTHON="${PAMO_BUILD_PYTHON:-${VIRTUAL_ENV}/bin/python}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0+PTX}"

"${PYTHON}" - <<'PY'
from packaging.version import Version
import os
import torch

cuda = torch.version.cuda
if not cuda:
    raise SystemExit("build requires a CUDA-enabled PyTorch (CUDA 12)")
if int(str(cuda).split(".", 1)[0]) != 12:
    raise SystemExit(f"build requires CUDA 12 PyTorch, got CUDA {cuda}")
torch_version = Version(torch.__version__.split("+", 1)[0])
if torch_version < Version("2.2"):
    raise SystemExit(f"build requires PyTorch >= 2.2, got {torch.__version__}")
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0+PTX")
print(
    f"building pamo-0.1.0+torch{torch_version.major}.{torch_version.minor}."
    f"cu{''.join(str(cuda).split('.'))} "
    f"against torch {torch.__version__} cuda {cuda} archs {arch_list}"
)
PY

mkdir -p "${DIST}"
cd "${ROOT}"
uv build \
  --wheel \
  --no-build-isolation \
  --python "${PYTHON}" \
  --out-dir "${DIST}" \
  "${ROOT}"
