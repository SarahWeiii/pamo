# PaMO (internal research build)

This directory builds one Linux x86-64 wheel containing PaMO and its three
pipeline stages. Source builds target:

- Python 3.10, 3.11, and 3.12
- PyTorch 2.2 or newer with CUDA 12
- warp-lang 1.16 or newer
- NVIDIA GPUs with compute capability 8.0 or newer (native Ampere through
  Hopper cubins; Blackwell and newer via PTX JIT)

The wheel preserves the upstream import:

```python
from pamo import PaMO
```

CUDA 11 and CUDA 12 cannot share one native wheel: the extensions link to
the PyTorch CUDA runtime they were compiled against. GPU architectures can
share one wheel. The default fat binary keeps four native cubins for the
mainstream CUDA 12 fleet, then uses PTX instead of extra Blackwell copies:

- `sm_80`: A100
- `sm_86`: A10, A40, RTX 30
- `sm_89`: L40, L20, L4, RTX 40
- `sm_90`: H100, H200
- `compute_90` PTX: Blackwell (`sm_100` / `sm_120`) and newer, JIT'd by the driver

Native `sm_100` and `sm_120` cubins are omitted on purpose: datacenter and
consumer Blackwell are different SMs, so both would be extra full copies of
every kernel. The published matrix is therefore Python ABI x PyTorch minor x
CUDA Runtime, for example:

```text
pamo-0.1.0+torch2.7.cu126-cp311-cp311-linux_x86_64.whl
```

PEP 440 puts that matrix in the local version (`+torch2.7.cu126`), not in
extra dash-separated filename fields. That wheel requires `torch>=2.7,<2.8`
built with the `cu126` runtime, and GPUs of compute capability 8.0 or newer.
Rebuild once per PyTorch x CUDA Runtime pair you need to ship.

To produce a narrower Ada-only artifact, override the arch list; the extra
SM tag keeps it from colliding with the default fat wheel:

```bash
TORCH_CUDA_ARCH_LIST="8.9+PTX" scripts/build-pamo.sh
```

which yields `pamo-0.1.0+torch2.7.cu126.sm89.ptx-...`.

Build from this directory inside an activated virtualenv that already has a
CUDA 12 PyTorch 2.2+ install. Isolated builds may otherwise pull a CPU wheel.
The frontend is uv; setuptools is only the CUDA extension backend:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install ninja setuptools packaging
scripts/build-pamo.sh
```

Select a different interpreter by creating that virtualenv first:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install ninja setuptools packaging
scripts/build-pamo.sh
```

The same script is included in the wheel's corresponding-source archive.
After extracting it, activate a compatible virtualenv and rebuild:

```bash
source /path/to/.venv/bin/activate
scripts/build-pamo.sh
```

Stage 3 buffer allocation is delayed until Stage 2 has produced its mesh and
is sized from the actual input and output. This avoids the upstream worst-case
preallocation on 16 GiB cards. Pass an explicit `stage3_config` only when a
workload needs larger collision buffers.

This build is for internal, non-commercial research only. Do not publish it to
an external package index. See `NOTICE` and the files under `licenses/` before
redistributing the wheel internally. The wheel also carries a version-matched
corresponding-source archive under `share/pamo/corresponding-source/` so its
three native extensions can be rebuilt without relying on an uncommitted tree.
