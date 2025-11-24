#!/bin/bash
#
# JAX and XLA Optimization Configuration
# Source this before training for optimal performance
#

echo "=========================================="
echo "Configuring JAX/XLA Optimizations"
echo "=========================================="

# ============================================
# JAX Configuration
# ============================================

# Compilation cache (speeds up restarts)
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache_${SLURM_JOB_ID:-default}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
echo "[*] JAX compilation cache: $JAX_COMPILATION_CACHE_DIR"

# Prevent recompilation of C extensions
export JAX_DISABLE_JIT_COMPILE_C_EXTENSIONS=1
echo "[*] JAX C extension compilation: disabled (faster startup)"

# Platform
export JAX_PLATFORMS="cuda"
echo "[*] JAX platform: $JAX_PLATFORMS"

# Precision
export JAX_ENABLE_X64=false
echo "[*] JAX 64-bit: disabled (faster computation)"

# ============================================
# XLA Optimization Flags
# ============================================

XLA_OPTIMIZATION_FLAGS=(
    # GEMM optimizations
    "--xla_gpu_triton_gemm_any=true"

    # Reduce compilation parallelism (more stable)
    "--xla_gpu_force_compilation_parallelism=1"

    # Memory optimizations
    "--xla_gpu_enable_while_loop_double_buffering=true"

    # CuDNN optimizations (if available)
    "--xla_gpu_enable_cudnn_fmha=true"

    # Reduce memory fragmentation
    "--xla_gpu_strict_conv_algorithm_picker=false"
)

# Combine flags
XLA_FLAGS_STR=$(IFS=' '; echo "${XLA_OPTIMIZATION_FLAGS[*]}")

# Append to existing XLA_FLAGS if any
if [ -n "$XLA_FLAGS" ]; then
    export XLA_FLAGS="$XLA_FLAGS $XLA_FLAGS_STR"
else
    export XLA_FLAGS="$XLA_FLAGS_STR"
fi

echo "[*] XLA optimization flags configured:"
for flag in "${XLA_OPTIMIZATION_FLAGS[@]}"; do
    echo "    $flag"
done

# ============================================
# Additional Performance Tuning
# ============================================

# NCCL (for multi-GPU/multi-node)
if [ -n "$SLURM_NTASKS" ] && [ "$SLURM_NTASKS" -gt 1 ]; then
    export NCCL_DEBUG=INFO  # Set to WARN in production
    export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
    export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
    echo "[*] NCCL optimizations enabled (multi-node training)"
fi

# CUDA configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Serialize kernels for better performance
echo "[*] CUDA max connections: 1 (serialized kernels)"

echo "=========================================="
echo "Optimization configuration complete!"
echo "=========================================="
echo ""
