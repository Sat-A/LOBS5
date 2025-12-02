#!/bin/bash
#SBATCH --job-name=count_params
#SBATCH --output=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/count_params_%j.out
#SBATCH --error=/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/count_params_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

source /lus/lfs1aip2/home/s5e/kangli.s5e/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Load CUDA module
module load cuda/12.6

# Set LD_LIBRARY_PATH for JAX CUDA libraries (complete list from working environment)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH


cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5
python /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/count_params.py
