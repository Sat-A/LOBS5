"""
Common ES building blocks for S5 models.

This module provides ES-compatible versions of basic neural network components
(Parameter, Linear, LayerNorm, etc.) that work with the HyperscaleES noiser framework.

Based on: HyperscaleES/src/hyperscalees/models/common.py
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Optional
from functools import partial

# Import base classes from HyperscaleES
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../HyperscaleES/src'))

from hyperscalees.models.base_model import Model, CommonInit, CommonParams
from hyperscalees.models.common import (
    PARAM, MM_PARAM, EMB_PARAM, EXCLUDED,
    merge_inits, merge_frozen, call_submodule,
    simple_es_tree_key, recursive_scan_split,
)

# Re-export for convenience
__all__ = [
    'Model', 'CommonInit', 'CommonParams',
    'PARAM', 'MM_PARAM', 'EMB_PARAM', 'EXCLUDED',
    'merge_inits', 'merge_frozen', 'call_submodule',
    'simple_es_tree_key', 'recursive_scan_split',
    'ES_Parameter', 'ES_MM', 'ES_TMM', 'ES_Linear', 'ES_LayerNorm',
    'get_noisy_param', 'ACTIVATIONS',
]

# Activation functions
def layer_norm_fn(x, eps=1e-5):
    """Simple layer normalization without learnable parameters."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'gelu': jax.nn.gelu,
    'silu': jax.nn.silu,
    'tanh': jnp.tanh,
    'sigmoid': jax.nn.sigmoid,
}


def get_noisy_param(common_params: CommonParams, name: str):
    """
    Get a potentially noisy parameter from common_params.

    If iterinfo is None (inference mode), returns the original parameter.
    Otherwise, applies noise based on es_map classification.
    """
    param = common_params.params[name]
    es_key = common_params.es_tree_key[name]

    # Inference mode: no noise
    if common_params.iterinfo is None:
        return param

    # Get noisy version via noiser
    return common_params.noiser.get_noisy_standard(
        common_params.frozen_noiser_params,
        common_params.noiser_params,
        param,
        es_key,
        common_params.iterinfo
    )


class ES_Parameter(Model):
    """
    ES-compatible single parameter tensor.

    Used for biases, scales, and other small learnable parameters.
    Uses PARAM es_map by default (full perturbation).
    """

    @classmethod
    def rand_init(cls, key, shape=None, scale=1.0, raw_value=None,
                  dtype=jnp.float32, es_type=PARAM) -> CommonInit:
        """
        Initialize a parameter tensor.

        Args:
            key: JAX random key
            shape: Shape of parameter (ignored if raw_value provided)
            scale: Initialization scale
            raw_value: Pre-computed initial value (optional)
            dtype: Data type
            es_type: ES map classification (PARAM, EXCLUDED, etc.)

        Returns:
            CommonInit with params, es_map, scan_map
        """
        if raw_value is not None:
            params = jnp.asarray(raw_value).astype(dtype)
        else:
            params = (jax.random.normal(key, shape) * scale).astype(dtype)

        return CommonInit(
            frozen_params=None,
            params=params,
            scan_map=(),
            es_map=es_type
        )

    @classmethod
    def _forward(cls, common_params: CommonParams):
        """Return the (potentially noisy) parameter value."""
        return common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params,
            common_params.es_tree_key,
            common_params.iterinfo
        )


class ES_MM(Model):
    """
    ES-compatible matrix multiplication layer (out_dim, in_dim).

    Uses MM_PARAM es_map (LORA perturbation for efficiency).
    """

    @classmethod
    def rand_init(cls, key, in_dim: int, out_dim: int,
                  dtype=jnp.float32, scale=None) -> CommonInit:
        """
        Initialize matrix multiplication weights.

        Args:
            key: JAX random key
            in_dim: Input dimension
            out_dim: Output dimension
            dtype: Data type
            scale: Initialization scale (default: 1/sqrt(in_dim))

        Returns:
            CommonInit with weight matrix (out_dim, in_dim)
        """
        if scale is None:
            scale = 1.0 / jnp.sqrt(in_dim)

        params = (jax.random.normal(key, (out_dim, in_dim)) * scale).astype(dtype)

        return CommonInit(
            frozen_params=None,
            params=params,
            scan_map=(),
            es_map=MM_PARAM
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """Compute x @ W.T with noisy weights."""
        return common_params.noiser.do_mm(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params,
            common_params.es_tree_key,
            common_params.iterinfo,
            x
        )


class ES_TMM(Model):
    """
    ES-compatible transposed matrix multiplication layer (in_dim, out_dim).

    Computes x @ W (not x @ W.T like ES_MM).
    Uses MM_PARAM es_map (LORA perturbation).
    """

    @classmethod
    def rand_init(cls, key, in_dim: int, out_dim: int,
                  dtype=jnp.float32, scale=None) -> CommonInit:
        """
        Initialize transposed matrix multiplication weights.

        Args:
            key: JAX random key
            in_dim: Input dimension
            out_dim: Output dimension
            dtype: Data type
            scale: Initialization scale (default: 1/sqrt(in_dim))

        Returns:
            CommonInit with weight matrix (in_dim, out_dim)
        """
        if scale is None:
            scale = 1.0 / jnp.sqrt(in_dim)

        params = (jax.random.normal(key, (in_dim, out_dim)) * scale).astype(dtype)

        return CommonInit(
            frozen_params=None,
            params=params,
            scan_map=(),
            es_map=MM_PARAM
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """Compute x @ W with noisy weights."""
        return common_params.noiser.do_Tmm(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params,
            common_params.es_tree_key,
            common_params.iterinfo,
            x
        )


class ES_Linear(Model):
    """
    ES-compatible linear layer (Dense).

    Computes: y = x @ W.T + bias (if use_bias=True)
    """

    @classmethod
    def rand_init(cls, key, in_dim: int, out_dim: int,
                  use_bias: bool = True, dtype=jnp.float32,
                  scale=None) -> CommonInit:
        """
        Initialize linear layer.

        Args:
            key: JAX random key
            in_dim: Input dimension
            out_dim: Output dimension
            use_bias: Whether to include bias
            dtype: Data type
            scale: Initialization scale

        Returns:
            CommonInit with weight and optional bias
        """
        key_w, key_b = jax.random.split(key)

        if use_bias:
            return merge_inits(
                weight=ES_MM.rand_init(key_w, in_dim, out_dim, dtype, scale),
                bias=ES_Parameter.rand_init(
                    key_b,
                    raw_value=jnp.zeros(out_dim, dtype=dtype),
                    dtype=dtype
                )
            )
        else:
            return merge_inits(
                weight=ES_MM.rand_init(key_w, in_dim, out_dim, dtype, scale)
            )

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """Compute linear transformation."""
        out = call_submodule(ES_MM, 'weight', common_params, x)
        if 'bias' in common_params.params:
            out = out + call_submodule(ES_Parameter, 'bias', common_params)
        return out


class ES_LayerNorm(Model):
    """
    ES-compatible Layer Normalization.

    Computes: y = (x - mean) / std * weight + bias

    Note: weight and bias are typically EXCLUDED from ES perturbation
    to maintain stable normalization statistics.
    """

    @classmethod
    def rand_init(cls, key, dim: int, dtype=jnp.float32,
                  use_bias: bool = True, es_type=EXCLUDED) -> CommonInit:
        """
        Initialize layer normalization parameters.

        Args:
            key: JAX random key
            dim: Normalization dimension
            dtype: Data type
            use_bias: Whether to include bias
            es_type: ES map type (default EXCLUDED for stability)

        Returns:
            CommonInit with weight and optional bias
        """
        key_w, key_b = jax.random.split(key)

        if use_bias:
            return merge_inits(
                weight=ES_Parameter.rand_init(
                    key_w, raw_value=jnp.ones(dim, dtype=dtype),
                    dtype=dtype, es_type=es_type
                ),
                bias=ES_Parameter.rand_init(
                    key_b, raw_value=jnp.zeros(dim, dtype=dtype),
                    dtype=dtype, es_type=es_type
                )
            )
        else:
            return merge_inits(
                weight=ES_Parameter.rand_init(
                    key_w, raw_value=jnp.ones(dim, dtype=dtype),
                    dtype=dtype, es_type=es_type
                )
            )

    @classmethod
    def _forward(cls, common_params: CommonParams, x, eps=1e-5):
        """Apply layer normalization."""
        # Normalize
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + eps)

        # Scale and shift
        weight = call_submodule(ES_Parameter, 'weight', common_params)
        out = x_norm * weight

        if 'bias' in common_params.params:
            bias = call_submodule(ES_Parameter, 'bias', common_params)
            out = out + bias

        return out


class ES_GroupNorm(Model):
    """
    ES-compatible Group Normalization.

    Used in RWKV7's ln_x for attention normalization.
    """

    @classmethod
    def rand_init(cls, key, num_channels: int, num_groups: int,
                  dtype=jnp.float32, es_type=EXCLUDED) -> CommonInit:
        """
        Initialize group normalization parameters.
        """
        key_w, key_b = jax.random.split(key)

        return merge_frozen(
            merge_inits(
                weight=ES_Parameter.rand_init(
                    key_w, raw_value=jnp.ones(num_channels, dtype=dtype),
                    dtype=dtype, es_type=es_type
                ),
                bias=ES_Parameter.rand_init(
                    key_b, raw_value=jnp.zeros(num_channels, dtype=dtype),
                    dtype=dtype, es_type=es_type
                )
            ),
            num_groups=num_groups
        )

    @classmethod
    def _forward(cls, common_params: CommonParams, x, eps=1e-5):
        """Apply group normalization."""
        num_groups = common_params.frozen_params['num_groups']

        # x shape: (..., C)
        orig_shape = x.shape
        C = orig_shape[-1]
        G = num_groups

        # Reshape for group norm
        x = x.reshape(*orig_shape[:-1], G, C // G)

        # Normalize within groups
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + eps)

        # Reshape back
        x_norm = x_norm.reshape(*orig_shape)

        # Scale and shift
        weight = call_submodule(ES_Parameter, 'weight', common_params)
        bias = call_submodule(ES_Parameter, 'bias', common_params)

        return x_norm * weight + bias
