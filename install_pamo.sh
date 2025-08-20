cd simp_cuda
pip install -e .
cd safe_project/warp_
chmod +x ./tools/packman/packman
python build_lib.py --cuda_path /usr/local/cuda  # Replace with your CUDA path
pip install . 
cd .. # move to simp_cuda/safe_project
pip install -e .