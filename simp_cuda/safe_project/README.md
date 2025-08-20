# Stage 3 

## Dependencies

- libigl
- cgal
- trimesh
- numpy
- warp


## Installation

Clone this repository with the `--recurse-submodules` flag to include the submodules. 

```bash
git clone --recurse-submodules git@github.com:Rabbit-Hu/stage3.git
```

Build Warp and pip install it.

```bash
cd stage3/warp_
python build_lib.py --cuda_path /usr/local/cuda  # Replace with your CUDA path
pip install .
```

Finally install the stage3 package.

```bash
cd ..  # back to stage3
pip install -e .
```


## Running

```python
config = stage3.config.Stage3Config()  # default config
# Optionally modify the config here before creating the system
system = stage3.system.Stage3System(config)  # create a system (with all the cuda arrays)

stage3_V, stage3_F = stage3.process(
    gt_mesh.vertices,
    gt_mesh.faces,
    stage2_mesh.vertices,
    stage2_mesh.faces,
    5,
    system=system,  # if provided, reuse the same system to avoid memory allocation
    config=config,  # if system is not provided, use this config to create a new system
)
```


## Scripts

Try processing a single mesh:

```bash
python scripts/try_process.py --id 1087134
python scripts/try_process.py --stage2_dir data/examples/stage2 --gt_dir data/examples/gt --id cubehole
```

Evaluate stage2 outputs (mainly for sanity check):

```bash
python scripts/test_stage2.py
```

Run stage 3 on all meshes:

```bash
python scripts/run_stage3.py --save_mesh
python scripts/run_stage3.py --stage2_dir /home/xiaodi/Desktop/Repos/stage3/data/wostage3_0510_0.2 --output_dir output/stage3_0509_0510_0.2 --save_mesh
```

Verify that all meshes are intersection-free using CGAL:
    
```bash
./scripts/cgal_test_all.sh output/stage3_0506_matrix_free/meshes output/log.txt
```


## Notes

- Elasticity
    - [x] Stretching (edge spring, or linear elasticity)
        - [x] Implemented StVK elasticity (needs testing)
    - [x] Bending (vertex curvature by Laplacian)
        - Doesn't work at all
    - [x] Bending (hinge angle)
        - Seems to work
- Distance to target
    - [x] output vertex to GT triangle
    - [x] GT sample to output triangle
- Collision barrier
    - [x] IPC collision barrier
    - [x] CCD
    - [ ] Buffer-free collision barrier energy
        - Needs to re-detect collision pairs at every CCD, compute_energy, compute_diff, compute_hess_dx
        - No need to clamp updates

### Arrays and memory

Assume 1M GT vertices, 50% decimation (MP = MP_GT / 2)

Recall that: F = V * 2, E = F * 1.5 = V * 3

- System: MP_GT * 20, NS_GT * 4
    - MP * 19 = MP_GT * 9.5
    - MP_GT * 3
    - MF_GT * 3 = MP_GT * 6
    - NS_GT * 4
    - MF * 3 = MP_GT * 3
    - ME * 2 = MP * 3 = MP_GT * 1.5
- CG Solver: MP_GT * 6
    - MP * 12 = MP_GT * 6
- Energy: MP_GT * 2, NS_GT * 15, MB * 19
    - Mesh2GTDistance: 
        - MP * 4 = MP_GT * 2
    - GT2MeshDistance:
        - NS_GT * 15
    - Collision:
        - MB * 19

In total: MP_GT * 28, NS_GT * 19, MB * 19
If we can remove MB, then MP_GT * 28 * 4B = 112M


## Progress Log

### 05/02/2024

- [x] Learn how to evaluate stage2 outputs
    - [x] Set a seed? Increase the number of samples? Need to discuss with Seonghunn

### 05/03/2024

- New framework
- Test CG solver with a dummy energy

### 05/04/2024

- [x] Ask Seonghun about the "shift" in the data
- Implement and test PPDistance energy
- Implement Collision energy
- [x] Implement CCD
- Important: I removed all non outer terms from hessian matrix. Seems it still works, even more numerically stable. If so we can make barrier functions matrix-free. 

### 05/05/2024

- [x] Add 2-way distance energy (like 2-way CD)
    - New results are in `/home/xiaodi/Repos/stage3/output/stage3_0505`

### 05/06/2024

- Increase the threshold for edge length in ee classification
    - if either edge length < 1e-5, no EE
- [x] Make collision barrier matrix-free
    - New results are in `/home/xiaodi/Repos/stage3/output/stage3_0506_matrix_free`

### 05/12/2024

- TODO: add a CCD to prevent triangles with too small areas

### 05/13/2024

- TODO: convergence curve for Newton's method

