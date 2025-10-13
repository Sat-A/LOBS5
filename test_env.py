#!/usr/bin/env python3
"""
Environment verification script for LOBS5 project
Tests JAX, Flax, PyTorch, and other dependencies
"""

import sys

def main():
    print("=" * 60)
    print("LOBS5 Environment Verification")
    print("=" * 60)

    # Test imports
    print("\n[1/4] Testing package imports...")
    try:
        import jax
        import jax.numpy as jnp
        import flax
        import optax
        import orbax
        import numpy as np
        import torch
        import pandas as pd
        import matplotlib
        import wandb
        print("✓ All core packages imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

    # Check versions
    print("\n[2/4] Package versions:")
    print(f"  JAX version:     {jax.__version__}")
    print(f"  Flax version:    {flax.__version__}")
    print(f"  Optax version:   {optax.__version__}")
    print(f"  NumPy version:   {np.__version__}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Pandas version:  {pd.__version__}")

    # Check devices
    print("\n[3/4] Device information:")
    print(f"  JAX devices:     {jax.devices()}")
    print(f"  JAX backend:     {jax.default_backend()}")
    print(f"  PyTorch CUDA:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  PyTorch CUDA version: {torch.version.cuda}")
        print(f"  PyTorch GPU count:    {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}:                 {torch.cuda.get_device_name(i)}")

    # Test GPU computation
    print("\n[4/4] GPU computation tests:")

    # Test JAX GPU
    try:
        if jax.default_backend() == 'gpu':
            x_jax = jnp.ones((1000, 1000))
            y_jax = jnp.dot(x_jax, x_jax)
            print(f"  ✓ JAX GPU test passed: output shape {y_jax.shape}")
        else:
            print("  ⚠ JAX: No GPU detected, using CPU")
    except Exception as e:
        print(f"  ✗ JAX GPU test failed: {e}")

    # Test PyTorch GPU
    try:
        if torch.cuda.is_available():
            x_torch = torch.ones(1000, 1000).cuda()
            y_torch = torch.matmul(x_torch, x_torch)
            print(f"  ✓ PyTorch GPU test passed: output shape {tuple(y_torch.shape)}")
        else:
            print("  ⚠ PyTorch: No GPU detected, using CPU")
    except Exception as e:
        print(f"  ✗ PyTorch GPU test failed: {e}")

    # Test project imports
    print("\n[Bonus] Testing LOBS5 project imports...")
    try:
        from s5.ssm import init_S5SSM
        from s5.seq_model import StackedEncoderModel
        from lob.lob_seq_model import BatchLobPredModel
        print("  ✓ LOBS5 project modules imported successfully")
    except ImportError as e:
        print(f"  ✗ LOBS5 import failed: {e}")

    print("\n" + "=" * 60)
    print("Environment verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
