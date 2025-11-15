#!/usr/bin/env python3
"""
Test memory_stats() in a context similar to actual training
"""

import jax
import jax.numpy as jnp

print("="*60)
print("Test 1: Fresh import (like your manual test)")
print("="*60)
devices = jax.local_devices()
print(f"Devices: {len(devices)} found")
for i, device in enumerate(devices):
    stats = device.memory_stats()
    print(f"Device {i}: stats = {stats}")
    if stats:
        print(f"  bytes_in_use: {stats['bytes_in_use'] / 1024**3:.2f} GB")
        print(f"  bytes_limit: {stats['bytes_limit'] / 1024**3:.2f} GB")
print()

print("="*60)
print("Test 2: After some JAX computation")
print("="*60)
# Do some JAX computation (similar to model initialization)
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
z = y.block_until_ready()  # Force computation

print("After computation:")
for i, device in enumerate(devices):
    stats = device.memory_stats()
    print(f"Device {i}: stats = {stats}")
    if stats:
        print(f"  bytes_in_use: {stats['bytes_in_use'] / 1024**3:.2f} GB")
print()

print("="*60)
print("Test 3: After pmap (like training)")
print("="*60)
# Create a pmapped function (this is what training uses)
@jax.pmap
def dummy_fn(x):
    return x * 2

# Allocate data across devices
data = jnp.ones((len(devices), 1000, 1000))
result = dummy_fn(data)
result[0].block_until_ready()

print("After pmap:")
for i, device in enumerate(devices):
    stats = device.memory_stats()
    print(f"Device {i}: stats = {stats}")
    if stats:
        print(f"  bytes_in_use: {stats['bytes_in_use'] / 1024**3:.2f} GB")
    else:
        print(f"  stats is None or empty!")
print()

print("="*60)
print("Test 4: Inside a function (like memory_profiler)")
print("="*60)
def test_from_function():
    for i, device in enumerate(jax.local_devices()):
        stats = device.memory_stats()
        print(f"Device {i}: stats type = {type(stats)}, value = {stats}")
        if stats:
            print(f"  ✓ Has data")
        else:
            print(f"  ✗ None or empty")

test_from_function()
