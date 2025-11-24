#!/bin/bash
#SBATCH --job-name=encode24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=logs_encode24/encode_%j_%x.log
#SBATCH --error=logs_encode24/encode_%j_%x.log

# =============================================================================
# Pre-encode LOBSTER data to 24-token format (base-100 size encoding)
# =============================================================================
# This script encodes raw message data (14 columns) to encoded format (24 tokens)
# and transforms orderbook data to volume image representation.
#
# Usage:
#   sbatch batch_encode24.sh                    # Encode both train and test
#   sbatch batch_encode24.sh train              # Encode train only
#   sbatch batch_encode24.sh test               # Encode test only
# =============================================================================

# set -euo pipefail

# Create log directory
mkdir -p logs_encode24

echo "=============================================="
echo "Pre-encoding LOBSTER data to 24-token format"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="

# Environment setup
cd /lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lobs5
module load cuda/12.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_cupti/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvshmem/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH


# JAX settings for encoding - use CPU for multiprocessing safety
export JAX_PLATFORMS="cpu"
# export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# export XLA_PYTHON_CLIENT_MEM_FRACTION="0.5"
# export CUDA_VISIBLE_DEVICES=0

# Data directories
TRAIN_INPUT="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021"
TRAIN_OUTPUT="/lus/lfs1aip2/home/s5e/kangli.s5e/GOOG2016TO2021_encoded24"

TEST_INPUT="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2"
TEST_OUTPUT="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/tokenized_lobs5_v2_encoded24"

# Number of workers (use multiple CPUs)
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

# Determine what to encode
ENCODE_MODE="${1:-all}"

encode_dataset() {
    local input_dir="$1"
    local output_dir="$2"
    local name="$3"

    echo ""
    echo "=============================================="
    echo "Encoding: $name"
    echo "  Input:  $input_dir"
    echo "  Output: $output_dir"
    echo "=============================================="

    # Check if input exists
    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: Input directory not found: $input_dir"
        return 1
    fi

    # Run encoding
    python -u pre_encode_data.py \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --num_workers "$NUM_WORKERS"

    # Verify output
    local n_msg=$(find "$output_dir" -name "*message*.npy" 2>/dev/null | wc -l)
    local n_book=$(find "$output_dir" -name "*orderbook*.npy" -o -name "*book*.npy" 2>/dev/null | wc -l)

    echo ""
    echo "Verification for $name:"
    echo "  Message files: $n_msg"
    echo "  Book files: $n_book"

    # Quick shape check on first file
    local first_msg=$(find "$output_dir" -name "*message*.npy" | head -1)
    if [[ -n "$first_msg" ]]; then
        python -c "import numpy as np; d=np.load('$first_msg'); print(f'  Sample message shape: {d.shape}')"
    fi

    local first_book=$(find "$output_dir" -name "*orderbook*.npy" -o -name "*book*.npy" | head -1)
    if [[ -n "$first_book" ]]; then
        python -c "import numpy as np; d=np.load('$first_book'); print(f'  Sample book shape: {d.shape}')"
    fi
}

# Main execution
case "$ENCODE_MODE" in
    train)
        encode_dataset "$TRAIN_INPUT" "$TRAIN_OUTPUT" "TRAINING DATA (GOOG 2016-2021)"
        ;;
    test)
        encode_dataset "$TEST_INPUT" "$TEST_OUTPUT" "TEST DATA (JAN 2023)"
        ;;
    all|*)
        encode_dataset "$TRAIN_INPUT" "$TRAIN_OUTPUT" "TRAINING DATA (GOOG 2016-2021)"
        encode_dataset "$TEST_INPUT" "$TEST_OUTPUT" "TEST DATA (JAN 2023)"
        ;;
esac

echo ""
echo "=============================================="
echo "Pre-encoding completed!"
echo "=============================================="
echo "Output directories:"
echo "  Train: $TRAIN_OUTPUT"
echo "  Test:  $TEST_OUTPUT"
echo ""
echo "To use these in training, update run_lobster_padded_large.sh:"
echo "  --dir_name='$TRAIN_OUTPUT'"
echo "  --test_dir_name='$TEST_OUTPUT'"
echo "  --data_mode='encoded'"
echo "=============================================="
