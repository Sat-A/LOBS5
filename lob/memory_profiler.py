"""
GPU Memory Profiling Utilities for LOBS5 Training
Based on: https://github.com/jax-ml/jax/issues/8096
"""

import os
import psutil
import jax
import jax.numpy as jnp


def print_memory_usage(step_name=""):
    """
    Print GPU and system memory usage
    Based on: https://github.com/jax-ml/jax/issues/8096
    """
    process = psutil.Process(os.getpid())
    cpu_mem_gb = process.memory_info().rss / 1024 ** 3

    print(f"\n{'='*60}")
    if step_name:
        print(f"Memory Usage @ {step_name}")
    else:
        print(f"Memory Usage")
    print(f"{'='*60}")
    print(f"CPU Memory: {cpu_mem_gb:.2f} GB")

    # Debug: Check what devices are available
    devices = jax.local_devices()
    print(f"JAX devices: {len(devices)} found")

    if len(devices) == 0:
        print("WARNING: No JAX devices found! jax.local_devices() returned empty list")
        print("This may indicate GPU is disabled via environment variables")
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'not set')}")
    else:
        # JAX device memory
        for device in devices:
            try:
                stats = device.memory_stats()
                if stats:
                    used_mb = stats['bytes_in_use'] / 1024**2
                    limit_gb = stats['bytes_limit'] / 1024**3
                    used_gb = stats['bytes_in_use'] / 1024**3
                    used_pct = (used_gb / limit_gb * 100) if limit_gb > 0 else 0
                    print(f"Device {device.id}: {used_gb:.2f} GB / {limit_gb:.2f} GB ({used_pct:.1f}%)")
                else:
                    print(f"Device {device.id}: memory_stats() returned None/empty")
            except Exception as e:
                print(f"Device {device.id}: Error - {e}")

    print(f"{'='*60}\n")


def log_memory(step_name="", verbose=True, device_id=0):
    """
    Log CPU and GPU memory usage

    Args:
        step_name: Identifier for this measurement point
        verbose: If True, print to console; if False, only return values
        device_id: Which GPU to report (default 0, since all GPUs have similar usage in pmap)

    Returns:
        dict with cpu_gb and gpu_stats
    """
    cpu_gb = get_process_memory_gb()
    gpu_stats = get_device_memory_stats(device_id)

    if verbose and gpu_stats:
        print(f"\n{'='*60}")
        print(f"Memory @ {step_name}")
        print(f"{'='*60}")
        print(f"CPU: {cpu_gb:.2f} GB")
        print(f"GPU {device_id}: {gpu_stats['used_gb']:.2f} / {gpu_stats['limit_gb']:.2f} GB ({gpu_stats['used_pct']:.1f}%)")
        print(f"{'='*60}\n")

    return {
        'step': step_name,
        'cpu_gb': cpu_gb,
        'gpu_stats': gpu_stats
    }


def log_all_devices(step_name=""):
    """Log memory usage for all GPU devices"""
    cpu_gb = get_process_memory_gb()
    device_stats = get_all_devices_memory()

    print(f"\n{'='*60}")
    print(f"All Devices Memory @ {step_name}")
    print(f"{'='*60}")
    print(f"CPU: {cpu_gb:.2f} GB")

    if len(device_stats) > 0:
        for device_id, stats in device_stats:
            print(f"GPU {device_id}: {stats['used_gb']:.2f} / {stats['limit_gb']:.2f} GB ({stats['used_pct']:.1f}%)")
    else:
        print("[WARNING] JAX device.memory_stats() not available, trying nvidia-smi fallback...")

        # Try nvidia-smi as fallback
        nvidia_stats = get_gpu_memory_nvidia_smi()
        if nvidia_stats:
            print("[INFO] Using nvidia-smi for GPU memory monitoring")
            for gpu_info in nvidia_stats:
                print(f"GPU {gpu_info['device_id']}: {gpu_info['used_gb']:.2f} / {gpu_info['limit_gb']:.2f} GB ({gpu_info['used_pct']:.1f}%)")
            device_stats = nvidia_stats  # Update return value
        else:
            print("[ERROR] Both device.memory_stats() and nvidia-smi failed")
            print("GPU memory monitoring not available on this platform")

    print(f"{'='*60}\n")

    return {
        'step': step_name,
        'cpu_gb': cpu_gb,
        'devices': device_stats
    }


def log_tensor_memory(tensor, name="tensor"):
    """
    Log memory footprint of a JAX tensor

    Args:
        tensor: JAX array
        name: Name for logging

    Returns:
        size in MB
    """
    size_bytes = tensor.size * tensor.itemsize
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)

    if size_gb > 1.0:
        print(f"Tensor '{name}': shape={tensor.shape}, size={size_gb:.2f} GB")
    else:
        print(f"Tensor '{name}': shape={tensor.shape}, size={size_mb:.2f} MB")

    return size_mb


# JAX-compatible debug print version for use inside jitted functions
def jax_debug_log_memory(step_name):
    """
    JAX debug print version - can be called inside jitted functions
    Note: This will print during trace/compilation, not during execution
    """
    cpu_gb = get_process_memory_gb()
    gpu_stats = get_device_memory_stats(0)

    if gpu_stats:
        jax.debug.print(
            "Memory @ {}: CPU={:.2f}GB, GPU={:.2f}/{:.2f}GB ({:.1f}%)",
            step_name, cpu_gb, gpu_stats['used_gb'], gpu_stats['limit_gb'], gpu_stats['used_pct']
        )


class MemoryTracker:
    """Context manager for tracking memory usage across a code block"""

    def __init__(self, name="", device_id=0):
        self.name = name
        self.device_id = device_id
        self.start_stats = None
        self.end_stats = None

    def __enter__(self):
        self.start_stats = log_memory(f"{self.name} [START]", verbose=False, device_id=self.device_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stats = log_memory(f"{self.name} [END]", verbose=False, device_id=self.device_id)

        # Calculate delta
        if self.start_stats['gpu_stats'] and self.end_stats['gpu_stats']:
            start_gb = self.start_stats['gpu_stats']['used_gb']
            end_gb = self.end_stats['gpu_stats']['used_gb']
            delta_gb = end_gb - start_gb

            print(f"\n{'='*60}")
            print(f"Memory Delta for '{self.name}'")
            print(f"{'='*60}")
            print(f"Start: {start_gb:.2f} GB")
            print(f"End:   {end_gb:.2f} GB")
            print(f"Delta: {delta_gb:+.2f} GB")
            print(f"{'='*60}\n")
