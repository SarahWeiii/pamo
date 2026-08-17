import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from pamo import PaMO

MESH_PATH = Path(__file__).parents[2] / "mesh" / "Dino_B015CZP872_SmallOrangeTRex.obj"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
def test_real_dino_mesh_runs_all_three_stages(tmp_path):
    mesh = trimesh.load_mesh(MESH_PATH, process=False)
    assert isinstance(mesh, trimesh.Trimesh)

    optimizer = PaMO(
        mesh,
        use_stage1=True,
        use_stage3=True,
        device="cuda:0",
    )
    vertices, faces = optimizer.run(
        torch.from_numpy(mesh.vertices),
        torch.from_numpy(mesh.faces),
        ratio=0.01,
        iter=10_000,
        stage3_iters=1,
    )

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert optimizer.target_faces is not None
    assert 0 < faces.shape[0] <= optimizer.target_faces
    assert optimizer.config.max_blocks <= 1 << 22
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0
    assert faces.max() < vertices.shape[0]

    output_path = Path(os.environ.get("PAMO_TEST_OUTPUT", tmp_path / "dino-pamo.obj"))
    output_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    output_mesh.export(output_path)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "input_vertices": int(mesh.vertices.shape[0]),
                "input_faces": int(mesh.faces.shape[0]),
                "output_vertices": int(vertices.shape[0]),
                "output_faces": int(faces.shape[0]),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
