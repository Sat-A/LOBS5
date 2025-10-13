#!/usr/bin/env python
"""
Test if CUDA libraries can be loaded by Python
Run this AFTER setting LD_LIBRARY_PATH
"""

import ctypes
import os
import sys

print("=== Testing CUDA Library Loading ===\n")

# Check LD_LIBRARY_PATH
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if ld_path:
    nvidia_paths = [p for p in ld_path.split(':') if 'nvidia' in p]
    print(f"✓ LD_LIBRARY_PATH has {len(nvidia_paths)} NVIDIA paths")
else:
    print("✗ LD_LIBRARY_PATH not set!")
    sys.exit(1)

# Test loading each library
libs_to_test = [
    'libcusparse.so.12',
    'libcublas.so.12',
    'libcudnn.so.9',
    'libcufft.so.11',
    'libcusolver.so.11',
]

print("\nTesting library loading:")
failed = []
for lib in libs_to_test:
    try:
        ctypes.CDLL(lib)
        print(f"  ✓ {lib}")
    except Exception as e:
        print(f"  ✗ {lib}: {e}")
        failed.append(lib)

if failed:
    print(f"\n✗ Failed to load {len(failed)} libraries")
    sys.exit(1)
else:
    print("\n✓ All CUDA libraries loaded successfully!")

print("\n=== Now testing JAX ===")
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    devices = jax.devices()
    print(f"Devices: {devices}")

    if any('cuda' in str(d).lower() for d in devices):
        print("\n✓✓✓ SUCCESS: GPU detected! ✓✓✓")
    else:
        print(f"\n✗ FAIL: No GPU detected, only {devices}")
except Exception as e:
    print(f"✗ Error with JAX: {e}")
    import traceback
    traceback.print_exc()
