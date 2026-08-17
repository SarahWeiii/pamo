import inspect

import pamo
import pytest
import trimesh
from pamo import MIN_CUDA_CAPABILITY, PaMO


def test_pamo_constructor_exposes_cuda_device_selection():
    """The bundled PaMO API must not hard-code cuda:0."""
    parameters = inspect.signature(PaMO).parameters

    assert "device" in parameters
    assert parameters["device"].default == "cuda"


def test_stage3_default_capacity_fits_16_gib_class_gpu():
    config = PaMO._stage3_config_for_mesh(
        gt_vertices=52_433,
        gt_faces=97_112,
        stage2_vertices=468,
        stage2_faces=932,
    )

    assert config.max_particles == 1_024
    assert config.max_gt_particles == 65_536
    assert config.max_blocks == 1 << 20


def test_pamo_accepts_compiled_cuda_capability(monkeypatch):
    monkeypatch.setattr(pamo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pamo.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        pamo.torch.cuda, "get_device_capability", lambda _device: MIN_CUDA_CAPABILITY
    )

    optimizer = PaMO(
        trimesh.creation.box(),
        use_stage1=False,
        use_stage3=False,
        device="cuda:0",
    )

    assert optimizer.device == pamo.torch.device("cuda:0")


def test_pamo_accepts_newer_cuda_capability_for_ptx_jit(monkeypatch):
    monkeypatch.setattr(pamo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pamo.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(pamo.torch.cuda, "get_device_capability", lambda _device: (9, 0))

    optimizer = PaMO(
        trimesh.creation.box(),
        use_stage1=False,
        use_stage3=False,
        device="cuda:0",
    )

    assert optimizer.device == pamo.torch.device("cuda:0")


def test_pamo_rejects_cuda_capability_older_than_compiled_target(monkeypatch):
    older = (
        (MIN_CUDA_CAPABILITY[0], MIN_CUDA_CAPABILITY[1] - 1)
        if MIN_CUDA_CAPABILITY[1]
        else (MIN_CUDA_CAPABILITY[0] - 1, 9)
    )
    monkeypatch.setattr(pamo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pamo.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        pamo.torch.cuda, "get_device_capability", lambda _device: older
    )

    with pytest.raises(RuntimeError, match="compute capability"):
        PaMO(
            trimesh.creation.box(),
            use_stage1=False,
            use_stage3=False,
            device="cuda:0",
        )
