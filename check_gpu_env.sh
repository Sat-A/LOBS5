#!/bin/bash

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Load CUDA module
module load cuda/12.6

echo "=== Checking CONDA_PREFIX ==="
echo $CONDA_PREFIX

echo -e "\n=== Finding NVIDIA libraries ==="
find $CONDA_PREFIX/lib/python3.11/site-packages/nvidia -name lib -type d 2>/dev/null

echo -e "\n=== Setting LD_LIBRARY_PATH ==="
export LD_LIBRARY_PATH=$(find $CONDA_PREFIX/lib/python3.11/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH

echo -e "\n=== Current LD_LIBRARY_PATH ==="
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia

echo -e "\n=== Checking JAX GPU detection ==="
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
