"""Build the self-contained PaMO research wheel."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import torch
from packaging.version import Version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
BASE_VERSION = "0.1.0"
MIN_TORCH = Version("2.2")
# Native cubins for the mainstream CUDA 12 fleet: A100 (8.0), A10/A40/RTX 30
# (8.6), L40/L20/RTX 40 (8.9), H100/H200 (9.0). Blackwell is covered by
# compute_90 PTX JIT instead of extra sm_100 and sm_120 cubins, which are
# different chips and would roughly add two more full copies of the kernels.
# Override TORCH_CUDA_ARCH_LIST to produce a slimmer or broader wheel.
DEFAULT_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0+PTX"
CUDA_ABI_MODULE = ROOT / "pamo" / "_cuda_abi.py"

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_CUDA_ARCH_LIST)

COMMON_CXX_FLAGS = ["-O3", "-std=c++17"]
COMMON_NVCC_FLAGS = ["-O3", "-std=c++17"]
RUNTIME_DEPENDENCIES = [
    "numpy>=1.26",
    "scipy>=1.12",
    "trimesh>=4.4",
    "libigl>=2.5.1,<3",
    "warp-lang>=1.16",
]


def _path(*parts: str) -> str:
    return Path(*parts).as_posix()


def parse_cuda_arch_list(arch_list: str) -> tuple[tuple[int, int], bool]:
    """Return the lowest compiled SM and whether forward-compatible PTX is present."""
    min_capability = None
    has_ptx = False
    for raw in arch_list.replace(",", " ").replace(";", " ").split():
        item = raw.strip()
        if not item:
            continue
        if item.upper().endswith("+PTX"):
            has_ptx = True
            item = item[: -len("+PTX")]
        elif item.upper() == "PTX":
            has_ptx = True
            continue
        major_text, minor_text = item.split(".", 1)
        capability = (int(major_text), int(minor_text))
        if min_capability is None or capability < min_capability:
            min_capability = capability
    if min_capability is None:
        raise RuntimeError(f"TORCH_CUDA_ARCH_LIST has no SMs: {arch_list!r}")
    return min_capability, has_ptx


def _sm_tag(capability: tuple[int, int]) -> str:
    return f"sm{capability[0]}{capability[1]}"


def _cuda_runtime_tag(cuda: str) -> str:
    """Map torch.version.cuda '12.6' to the PyTorch wheel tag 'cu126'."""
    return "cu" + "".join(str(cuda).split("."))


def resolve_build_identity() -> tuple[str, str, tuple[int, int], bool, str]:
    """Tag the wheel with the PyTorch x CUDA runtime matrix it was compiled against."""
    cuda = torch.version.cuda
    if not cuda:
        raise RuntimeError("PaMO wheels must be built against a CUDA-enabled PyTorch")
    cuda_major = int(str(cuda).split(".", 1)[0])
    if cuda_major != 12:
        raise RuntimeError(f"PaMO wheels require CUDA 12 PyTorch, got CUDA {cuda}")

    torch_version = Version(torch.__version__.split("+", 1)[0])
    if torch_version < MIN_TORCH:
        raise RuntimeError(
            f"PaMO requires PyTorch >= {MIN_TORCH}, got {torch.__version__}"
        )

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", DEFAULT_CUDA_ARCH_LIST)
    min_capability, has_ptx = parse_cuda_arch_list(arch_list)
    local = (
        f"torch{torch_version.major}.{torch_version.minor}."
        f"{_cuda_runtime_tag(cuda)}"
    )
    if arch_list != DEFAULT_CUDA_ARCH_LIST:
        local += f".{_sm_tag(min_capability)}"
        if has_ptx:
            local += ".ptx"
    torch_requirement = (
        f"torch>={torch_version.major}.{torch_version.minor},"
        f"<{torch_version.major}.{torch_version.minor + 1}"
    )
    return (
        f"{BASE_VERSION}+{local}",
        torch_requirement,
        min_capability,
        has_ptx,
        arch_list,
    )


def write_cuda_abi_module(
    min_capability: tuple[int, int], has_ptx: bool, arch_list: str
) -> None:
    CUDA_ABI_MODULE.write_text(
        "\"\"\"Build-time CUDA ABI recorded for runtime checks.\"\"\"\n\n"
        f"MIN_CUDA_CAPABILITY = {min_capability!r}\n"
        f"HAS_PTX = {has_ptx!r}\n"
        f"CUDA_ARCH_LIST = {arch_list!r}\n"
    )


def get_extensions() -> list[CUDAExtension]:
    pamo_sources = [
        _path("src", "pybind.cpp"),
        _path("src", "cusimp.cu"),
        _path("src", "cusimp_free.cu"),
    ]
    pdmc_sources = [
        _path("third_party", "pdmc", "src", "pybind.cpp"),
        _path("third_party", "pdmc", "src", "cudualmc.cu"),
    ]
    cumesh_root = Path("third_party") / "cumesh2sdf"
    cumesh_sources = [
        str(cumesh_root / "binding.cpp"),
        str(cumesh_root / "torchcumesh2sdf.cu"),
    ]

    return [
        CUDAExtension(
            "pamo._C",
            pamo_sources,
            include_dirs=[_path("src")],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": COMMON_CXX_FLAGS,
                "nvcc": [*COMMON_NVCC_FLAGS, "--extended-lambda", "--fmad=false"],
            },
        ),
        CUDAExtension(
            "pdmc._C",
            pdmc_sources,
            include_dirs=[_path("third_party", "pdmc", "src")],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={"cxx": COMMON_CXX_FLAGS, "nvcc": COMMON_NVCC_FLAGS},
        ),
        CUDAExtension(
            "torchcumesh2sdf",
            cumesh_sources,
            include_dirs=[str(cumesh_root)],
            extra_compile_args={
                "cxx": COMMON_CXX_FLAGS,
                "nvcc": [*COMMON_NVCC_FLAGS, "--use_fast_math"],
            },
        ),
    ]


def build_corresponding_source_archive(version: str) -> Path:
    """Bundle the exact source needed to rebuild this AGPL binary wheel."""
    archive_path = ROOT / "build" / f"pamo-{version}-corresponding-source.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    patterns = [
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "licenses/*",
        "pamo/*.py",
        "pdmc/*.py",
        "safe_project/src/**/*.py",
        "src/**/*",
        "tests/*.py",
        "third_party/cumesh2sdf/**/*",
        "third_party/pdmc/src/**/*",
    ]
    source_files = sorted(
        {
            path
            for pattern in patterns
            for path in ROOT.glob(pattern)
            if path.is_file()
            and path.suffix != ".pyc"
            and path.name != "_cuda_abi.py"
        }
    )
    build_script = ROOT / "scripts" / "build-pamo.sh"
    if not build_script.is_file():
        raise FileNotFoundError("scripts/build-pamo.sh is required for source archival")

    source_root = f"pamo-{version}"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in source_files:
            archive.add(path, arcname=f"{source_root}/{path.relative_to(ROOT)}")
        archive.add(build_script, arcname=f"{source_root}/scripts/build-pamo.sh")
    return archive_path


VERSION, TORCH_REQUIREMENT, MIN_CUDA_CAPABILITY, HAS_PTX, CUDA_ARCH_LIST = (
    resolve_build_identity()
)
write_cuda_abi_module(MIN_CUDA_CAPABILITY, HAS_PTX, CUDA_ARCH_LIST)
safe_project_packages = find_packages(where="safe_project/src")
source_archive = build_corresponding_source_archive(VERSION)

setup(
    version=VERSION,
    install_requires=[TORCH_REQUIREMENT, *RUNTIME_DEPENDENCIES],
    packages=["pamo", "pdmc", *safe_project_packages],
    package_dir={"pamo_safe_project": "safe_project/src/pamo_safe_project"},
    data_files=[
        (
            "share/pamo/corresponding-source",
            [source_archive.relative_to(ROOT).as_posix()],
        )
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=False)},
    zip_safe=False,
)
