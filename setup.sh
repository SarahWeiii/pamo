# Stage 1: Remeshing
echo "Installing stage 1: remeshing..."
pip install git+https://github.com/eliphatfs/cumesh2sdf.git
pip install pdmc

# Stage 2: Simplification
echo "Installing stage 2: simplification..."
cd simp_cuda
pip install .

# Stage 3: Safe Projection
echo "Installing stage 3: safe projection..."
cd safe_project/warp_
chmod +x ./tools/packman/packman
python build_lib.py --cuda_path /usr/local/cuda  # CUDA dir
pip install .
cd ..
pip install .
cd ../../

echo "Installation complete"