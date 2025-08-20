# Parallel Mesh Optimization (PaMO)
PaMO: Parallel Mesh Optimization for Intersection-Free Low-Poly Modeling on the GPU [PG2025]

*Seonghun Oh\*, Xiaodi Yuan\*, Xinyue Wei\*, Ruoxi Shi, Fanbo Xiang, Minghua Liu, Hao Su*

## Intro
We present a novel GPU-based mesh optimization pipeline with three core components:
1. Parallel remeshing: Converts arbitrary meshes into watertight, manifold, and intersection-free meshes while improving triangle quality.
2. Robust parallel simplification: Reduces mesh complexity with guaranteed intersection-free results.
3. Optimization-based safe projection: Realigns the simplified mesh to the original input, eliminating surface shifts from remeshing and restoring sharp features.

Our approach is highly efficient, simplifying a 2-million-face mesh to 20k triangles in just 3 seconds on an RTX 4090.

## Installation
### Option 1: Docker environment
```
docker run --name pamo -i -t --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility sarahwei0210/pamo:0.0.1 /bin/bash
bash install_pamo.sh
```
### Option 2: Install by anaconda (may take ~5min)
```
git clone --recurse-submodules https://github.com/SarahWeiii/pamo.git
conda env create -f env.yaml
conda activate pamo
bash setup.sh
```

## Demo

```
bash demo.sh
```
We offer three meshes stored under `./mesh` folder for the demo. The results will be saved under `./examples` folder.

## Example
```
python example.py --input INPUT_DIR --output OUTPUT_DIR --ratio 0.001
```

- **`--input`**: Specify the path to the input mesh file. If not provided, it defaults to `./mesh/crab.obj`.
- **`--ratio`**: Set the simplification ratio to control the target reduction in the number of triangles. For example, `--ratio 0.001` (default) means reducing the number of triangles to 0.1% of the original.
- **`--min-vertex`**: Add this flag to constrain the minimum number of vertices after simplification, default=0.
- **`--disable_stage1`**: Add this flag to skip the remeshing process (stage 1), default=false.
- **`--disable_stage3`**: Add this flag to skip the safe projection process (stage 3), default=false.

## Usage
### Import
```
from pamo import PaMO
```
### Constructor
Creates an instance of the `PAMO` class using the input mesh data.
```
pamo = PaMO(input_mesh, use_stage1=True, use_stage3=True)
```
### Run
Performs mesh optimization to reduce the complexity of the mesh while preserving essential details according to specified parameters.
```
pamo.run(points, triangles, ratio, tolerance=4, threshold=1e-3, iter=100000)
```

#### Parameters
**points** (`float Tensor`): Vertices of the mesh. A tensor of floating-point numbers representing the 3D coordinates of each vertex.

**triangles** (`int Tensor`): Faces of the mesh. A tensor of integers where each row represents a triangle in the mesh defined by indices into the points array.

**ratio** (`float`): Decimation ratio specifying the target reduction in the number of triangles.

**use_stage1** (`bool`, *default = True*): Whether to use a remeshing (stage 1) before simplification.

**use_stage3** (`bool`, *default = True*): Whether to use a safe projection (stage 3) after simplification.

**tolerance** (`int`, *default = 4*): Defines the number of iterations to run without edge collapses before stopping, accumulating invalid edges that do not qualify for collapsing. Lower values quicken termination, while higher values allow more iterations for potential optimization

## Cite
```
Coming soon...
```