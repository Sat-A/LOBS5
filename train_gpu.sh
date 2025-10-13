#!/bin/bash

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5

# Load CUDA module
module load cuda/12.6

# Set library paths for NVIDIA libraries (explicit paths for reliability)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH

# Verify GPU is available
echo "Checking GPU..."
python -c "import jax; print(f'JAX Devices: {jax.devices()}')"

# Run training
python run_train.py \
  --C_init=trunc_standard_normal \
  --batchnorm=True \
  --bidirectional=False \
  --blocks=8 \
  --bsz=16 \
  --d_model=32 \
  --dataset=lobster-prediction \
  --dir_name=./data/train \
  --dt_global=False \
  --epochs=1 \
  --jax_seed=1919 \
  --lr_factor=1 \
  --n_layers=6 \
  --opt_config=standard \
  --p_dropout=0.0 \
  --ssm_lr_base=0.001 \
  --ssm_size_base=32 \
  --warmup_end=1 \
  --weight_decay=0.05 \
  --use_book_data=False \
  --masking=causal
