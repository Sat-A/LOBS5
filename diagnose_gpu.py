#!/usr/bin/env python
import os
import sys

print("=== Environment Check ===")
print(f"LD_LIBRARY_PATH set: {'LD_LIBRARY_PATH' in os.environ}")

if 'LD_LIBRARY_PATH' in os.environ:
    paths = os.environ['LD_LIBRARY_PATH'].split(':')
    nvidia_paths = [p for p in paths if 'nvidia' in p]
    print(f"NVIDIA paths in LD_LIBRARY_PATH: {len(nvidia_paths)}")
    for p in nvidia_paths[:5]:
        print(f"  - {p}")
else:
    print("WARNING: LD_LIBRARY_PATH not set!")

print("\n=== Checking library files ===")
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    cusparse_lib = f"{conda_prefix}/lib/python3.11/site-packages/nvidia/cusparse/lib"
    print(f"cuSPARSE lib dir exists: {os.path.exists(cusparse_lib)}")
    if os.path.exists(cusparse_lib):
        files = os.listdir(cusparse_lib)
        print(f"Files in cuSPARSE lib: {[f for f in files if 'cusparse' in f][:3]}")

print("\n=== Trying to import JAX ===")
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
except Exception as e:
    print(f"ERROR importing JAX: {e}")
