#!/bin/bash
#
# TEST SCRIPT: Small model for testing intra-epoch checkpoint resume
# Model: 1024x12 (d_model=1024, n_layers=12)
# Each segment: 10 batches only
#
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

# Debug output
echo "========================================"
echo "[TEST] Running resume test on node: $(hostname)"
echo "[TEST] SLURM_NODEID: ${SLURM_NODEID:-N/A}"
echo "[TEST] SLURM_PROCID: ${SLURM_PROCID:-N/A}"
echo "[TEST] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "========================================"

# Source conda
source ~/miniforge3/etc/profile.d/conda.sh

# Activate environment
conda activate lobs5

# Load CUDA module
module load cuda/12.6

# Set LD_LIBRARY_PATH for JAX CUDA libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# GPU memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# JAX distributed coordination service timeout
export JAX_COORDINATOR_TIMEOUT_MS=600000
export TF_CPP_MIN_LOG_LEVEL=1

# Multi-node communication configuration
if [ -n "$JAX_COORDINATOR_ADDRESS" ]; then
    echo "[TEST] Multi-node coordinator: $JAX_COORDINATOR_ADDRESS"
    export JAX_PROCESS_COUNT=${SLURM_NNODES:-1}
    export JAX_PROCESS_INDEX=${SLURM_PROCID:-0}
    export JAX_LOCAL_PROCESS_COUNT=1
    export JAX_LOCAL_PROCESS_INDEX=0
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        export JAX_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    fi
fi

# CUDA cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "$SLURM_TMPDIR" ]; then
    export TMPDIR="$SLURM_TMPDIR"
    export CUDA_CACHE_PATH="$SLURM_TMPDIR/.nv/ComputeCache"
else
    export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"
fi
mkdir -p "$CUDA_CACHE_PATH" || true

# Check available GPUs
echo "[TEST] Available GPUs:"
nvidia-smi --list-gpus | head -4

# Run small model for testing resume functionality
python -u -B run_train.py \
        --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
        --blocks=16 --per_gpu_bsz=2 --d_model=1024 --dataset=lobster-prediction --merging=padded \
        --dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded24' \
        --test_dir_name='/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2_encoded24' \
        --data_mode='encoded' \
        --clip_eigs=True --activation_fn=half_glu1 \
        --dt_global=False --epochs=5 --jax_seed=42 --lr_factor=1 --n_layers=12 \
        --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0001 --ssm_size_base=1024 \
        --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
        --use_book_data=True --use_simple_book=False --book_transform=True  \
        --masking=none \
        --num_devices=4 --n_data_workers=4 \
        --debug_loading=False \
        --enable_profiler=False \
        --random_offsets_train=True \
        --shuffle_train=True \
        --debug_overfit=False \
        --lr_patience=3 \
        --USE_WANDB=False \
        --wandb_project=lobs5-test-resume \
        --wandb_entity=kang-oxford
