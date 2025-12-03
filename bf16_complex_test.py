#!/usr/bin/env python3
"""
BF16 complex matrix multiplication microbenchmark for JAX.

Usage:
  python bf16_complex_matmul_benchmark.py \
    --batch 4 --m 1024 --k 1024 --n 1024 --runs 5 --seed 0

Notes:
- Your arrays are represented as (..., M, K, 2) where axis -1 = (real, imag).
- The BF16 path performs two BF16 real matmuls:
    real = a_real @ b_real - a_imag @ b_imag
    imag = a_real @ b_imag + a_imag @ b_real
- The reference uses complex64 arithmetic for correctness checking.
"""
from __future__ import annotations
import argparse
import time
import sys

import jax
import jax.numpy as jnp
from jax import lax
import os

def print_device_info():
    devices = jax.devices()
    print("JAX devices:")
    for d in devices:
        # device has attributes like id, platform, device_kind
        kind = getattr(d, "device_kind", None)
        print(f" - id={d.id} platform={d.platform} kind={kind} "
              f"name={d}")


def make_random_pair(key, shape, dtype=jnp.float32):
    """Return pair-of-real representation with trailing axis of size 2."""
    x = jax.random.normal(key, shape + (2,), dtype=dtype)
    return x


def to_bf16_parts(x: jnp.ndarray):
    """Return (real, imag) as bfloat16 arrays from (..., ..., 2) float32 input."""
    return x[..., 0].astype(jnp.bfloat16), x[..., 1].astype(jnp.bfloat16)


@jax.jit
def complex_matmul_bf16(a: jnp.ndarray, b: jnp.ndarray, use_dot_general: bool = False) -> jnp.ndarray:
    """
    Batched complex matmul using BF16 real matmuls.

    Inputs:
      a: (..., M, K, 2) float32
      b: (..., K, N, 2) float32
    Returns:
      (..., M, N, 2) float32
    """
    a_real_bf, a_imag_bf = to_bf16_parts(a)
    b_real_bf, b_imag_bf = to_bf16_parts(b)

    if use_dot_general:
        # Generic dimension_numbers for standard batched matmul:
        # contract last axis of a with second-last axis of b
        # batch dims are all leading dims up to the matrix dims.
        a_nd = a_real_bf.ndim
        b_nd = b_real_bf.ndim
        # contraction dims: a's -2 with b's -3? We assume shapes (..., M, K) @ (..., K, N)
        # For simplicity, rely on jnp.matmul style via lax.dot_general with
        # the common mapping below for batch-matmul of rank >= 2.
        dimension_numbers = (((a_nd - 1,), (b_nd - 2,)), ((a_nd - 2,), (b_nd - 1,)))
        P = lax.Precision.HIGHEST
        rr = lax.dot_general(a_real_bf, b_real_bf, dimension_numbers, precision=P)
        ii = lax.dot_general(a_imag_bf, b_imag_bf, dimension_numbers, precision=P)
        ri = lax.dot_general(a_real_bf, b_imag_bf, dimension_numbers, precision=P)
        ir = lax.dot_general(a_imag_bf, b_real_bf, dimension_numbers, precision=P)
    else:
        # Use jnp.matmul which maps conveniently to device GEMM kernels
        rr = jnp.matmul(a_real_bf, b_real_bf)
        ii = jnp.matmul(a_imag_bf, b_imag_bf)
        ri = jnp.matmul(a_real_bf, b_imag_bf)
        ir = jnp.matmul(a_imag_bf, b_real_bf)

    real_bf = rr - ii
    imag_bf = ri + ir

    # Cast back to float32 for downstream uses
    real_f32 = real_bf.astype(jnp.float32)
    imag_f32 = imag_bf.astype(jnp.float32)
    return jnp.stack([real_f32, imag_f32], axis=-1)


def to_complex_pair(pair: jnp.ndarray) -> jnp.ndarray:
    """Convert pair-of-reals (..., 2) to complex dtype (...,)."""
    return pair[..., 0] + 1j * pair[..., 1]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7
    p = argparse.ArgumentParser(description="BF16 complex matmul benchmark (JAX)")
    p.add_argument("--batch", type=int, default=4, help="Batch size (B)")
    p.add_argument("--m", type=int, default=1024, help="M (rows of A output)")
    p.add_argument("--k", type=int, default=1024, help="K (inner dimension)")
    p.add_argument("--n", type=int, default=1024, help="N (cols of B output)")
    p.add_argument("--runs", type=int, default=5, help="Number of timed runs (after warmup)")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed")
    p.add_argument("--use-dot-general", action="store_true", help="Use lax.dot_general with highest precision")
    args = p.parse_args()

    print_device_info()
    print()
    print(f"Benchmark configuration: batch={args.batch}, M={args.m}, K={args.k}, N={args.n}, runs={args.runs}")
    print(f"use_dot_general={args.use_dot_general}")
    print()

    key = jax.random.PRNGKey(args.seed)
    # split keys for a and b
    key_a, key_b = jax.random.split(key, 2)

    # Create random float32 pair-of-real arrays
    a = make_random_pair(key_a, (args.batch, args.m, args.k), dtype=jnp.float32)
    b = make_random_pair(key_b, (args.batch, args.k, args.n), dtype=jnp.float32)

    # Warmup / compile
    print("JIT-compiling and warming up...")
    fn = lambda x, y: complex_matmul_bf16(x, y, use_dot_general=args.use_dot_general)
    compiled = jax.jit(fn)
    out = compiled(a, b)
    out.block_until_ready()
    print("Warmup complete. Starting timed runs.")
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = compiled(a, b)
        out.block_until_ready()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f" Run {i + 1:2d}: {elapsed:.6f} s")
    best = min(times)
    mean = sum(times) / len(times)
    print()
    print(f"Best time: {best:.6f} s   Mean time: {mean:.6f} s")

    # Compute GFLOPS estimate
    # Common complex-GEMM flop accounting approximates 8 * M*K*N real FLOPs per matrix multiply.
    # For batched shapes: multiply by batch size.
    ops_per_mat = 8 * args.m * args.k * args.n
    total_ops = ops_per_mat * args.batch
    gflops = total_ops / best / 1e9
    print(f"Effective complex-equivalent GFLOPS (best run): {gflops:.2f} GFLOPS")

    # Validate against complex64 reference on CPU/GPU (uses complex matmul in JAX)
    print("\nValidating accuracy against complex64 reference (this may be slower).")
    a_c = to_complex_pair(a).astype(jnp.complex64)
    b_c = to_complex_pair(b).astype(jnp.complex64)
    # Reference: batched complex matmul using einsum
    ref = jnp.einsum("bmk,bkn->bmn", a_c, b_c)
    # convert benchmark output to complex
    out_pair = out  # pair-of-reals float32
    out_c = to_complex_pair(out_pair).astype(jnp.complex64)
    # Compute max absolute error
    err = jnp.max(jnp.abs(ref - out_c))
    print(f"Max absolute error vs complex64 reference: {float(err):.6e}")

    # Print a small sample of values for manual inspection
    sample_index = (0, 0, 0)  # batch 0, row 0, col 0
    b_idx = 0
    i_idx = 0
    j_idx = 0
    print("\nSample (batch=0, row=0, col=0):")
    print(" ref (complex64):", ref[b_idx, i_idx, j_idx])
    print(" out (bf16->f32): ", out_c[b_idx, i_idx, j_idx])

    # Friendly note about hardware
    dev_list = jax.devices()
    if any(d.platform == "gpu" for d in dev_list):
        print("\nNote: BF16 path used device kernels when available. For highest throughput prefer H100 for large training workloads; L40S and other recent GPUs support BF16 too but performance characteristics differ.")
    else:
        print("\nWarning: no GPU found. This benchmark will run on CPU and will be much slower; BF16 performance kernels require a BF16-capable GPU for representative throughput.")

if __name__ == "__main__":
    main()
