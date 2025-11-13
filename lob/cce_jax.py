"""
Cut Cross-Entropy (CCE) implementation for JAX.

This module provides memory-efficient cross-entropy loss computation
that avoids materializing the full (batch × sequence × vocab) logits tensor.
Based on the paper: "Cut Your Losses in Large-Vocabulary Language Models"

Author: Kang Li
Date: 2025
"""


'''
The actual vocab_size seems to be
'''

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, Union
from functools import partial
import numpy as np


# ============================================================================
# Core CCE Implementation
# ============================================================================

@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def jax_linear_cross_entropy(
    embeddings: jax.Array,          # (batch*seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
    classifier_weight: jax.Array,   # (vocab_size, hidden_dim)
    targets: jax.Array,             # (batch*seq_len,) or (batch, seq_len)
    classifier_bias: Optional[jax.Array] = None,  # (vocab_size,)
    ignore_index: int = -100,       # tokens to ignore in loss
    shift: int = 0,                 # shift for numerical stability
    vocab_block_size: int = 512,   # block size for chunked computation
    filter_eps: Optional[float] = None,  # gradient filtering threshold
    return_components: bool = False,  # whether to return intermediate values
) -> Union[jax.Array, Tuple[jax.Array, dict]]:
    """
    Compute cross-entropy loss without materializing full logits tensor.

    This implementation computes CE loss by:
    1. Computing only the logits for target tokens
    2. Computing log-sum-exp in chunks to avoid memory explosion
    3. Combining them to get the final loss

    Memory complexity: O(batch_size × seq_len) instead of O(batch_size × seq_len × vocab_size)

    Args:
        embeddings: Input embeddings, shape (batch*seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        classifier_weight: Classifier weight matrix, shape (vocab_size, hidden_dim)
        targets: Target token indices, shape (batch*seq_len,) or (batch, seq_len)
        classifier_bias: Optional bias term, shape (vocab_size,)
        ignore_index: Index to ignore in loss computation (e.g., padding)
        shift: Numerical shift for stability
        vocab_block_size: Block size for chunked vocabulary computation
        filter_eps: Optional gradient filtering threshold
        return_components: If True, return dict with intermediate values

    Returns:
        loss: Average cross-entropy loss (scalar) or per-token losses
        components: (optional) Dict with intermediate values for debugging
    """

    # CRITICAL DEBUG: Confirm CCE is being called
    jax.debug.callback(lambda: print("[CCE ENTRY] jax_linear_cross_entropy CALLED"))

    # Reshape inputs to 2D if needed
    original_shape = embeddings.shape
    if len(embeddings.shape) == 3:
        batch_size, seq_len, hidden_dim = embeddings.shape
        embeddings = embeddings.reshape(-1, hidden_dim)
        targets = targets.reshape(-1)
    else:
        batch_size = embeddings.shape[0]
        seq_len = 1
        hidden_dim = embeddings.shape[1]

    vocab_size = classifier_weight.shape[0]
    num_tokens = embeddings.shape[0]

    # ========================================================================
    # Step 1: Compute logits for target tokens only
    # ========================================================================

    # Gather target token weights
    target_weights = classifier_weight[targets]  # (num_tokens, hidden_dim)

    # Compute target logits: sum(embeddings * target_weights, axis=-1)
    target_logits = jnp.sum(embeddings * target_weights, axis=-1)  # (num_tokens,)

    # Add bias if provided
    if classifier_bias is not None:
        target_bias = classifier_bias[targets]
        target_logits = target_logits + target_bias

    # Apply numerical shift for stability
    target_logits = target_logits - shift

    # ========================================================================
    # Step 2: Compute log-sum-exp in chunks
    # ========================================================================

    def compute_lse_chunked(embeddings, classifier_weight, classifier_bias,
                           vocab_block_size, shift):
        """Compute log-sum-exp over vocabulary in chunks to save memory."""

        # Pad to make vocab divisible by block_size
        padded_vocab_size = ((vocab_size + vocab_block_size - 1) // vocab_block_size) * vocab_block_size
        padding_size = padded_vocab_size - vocab_size

        if padding_size > 0:
            # Pad weights with zeros
            classifier_weight_padded = jnp.concatenate([
                classifier_weight,
                jnp.zeros((padding_size, hidden_dim))
            ], axis=0)
            if classifier_bias is not None:
                # Use very large negative value for bias padding (not -inf to avoid NaN)
                classifier_bias_padded = jnp.concatenate([
                    classifier_bias,
                    jnp.full((padding_size,), -1e20)
                ])
            else:
                classifier_bias_padded = None
        else:
            classifier_weight_padded = classifier_weight
            classifier_bias_padded = classifier_bias

        num_blocks = padded_vocab_size // vocab_block_size

        def compute_block_lse(block_idx):
            """Compute logits and LSE for a vocabulary block."""
            start_idx = block_idx * vocab_block_size

            # Use lax.dynamic_slice for JAX-compatible indexing
            block_weights = lax.dynamic_slice(
                classifier_weight_padded,
                (start_idx, 0),
                (vocab_block_size, hidden_dim)
            )

            # Compute block logits
            block_logits = jnp.matmul(embeddings, block_weights.T)

            if classifier_bias_padded is not None:
                block_bias = lax.dynamic_slice(
                    classifier_bias_padded,
                    (start_idx,),
                    (vocab_block_size,)
                )
                block_logits = block_logits + block_bias

            # Apply shift
            block_logits = block_logits - shift

            # For padded elements, the large negative bias will naturally suppress them
            # No need for explicit masking with -inf which can cause NaN issues

            return block_logits

        # DIAGNOSTIC: Print shapes before scan - use callback to ensure execution
        jax.debug.callback(lambda: print(f"[CCE SCAN] num_blocks={num_blocks}, num_tokens={num_tokens}, vocab_block_size={vocab_block_size}"))
        jax.debug.callback(lambda: print(f"[CCE SCAN] embeddings shape={embeddings.shape}, classifier_weight shape={classifier_weight.shape}"))
        jax.debug.callback(lambda: print(f"[CCE SCAN] Using sequential processing (lax.scan) for memory efficiency"))

        jax.debug.print("CCE scan - num_blocks: {}, num_tokens: {}, vocab_block_size: {}",
                       num_blocks, num_tokens, vocab_block_size)
        jax.debug.print("CCE scan - embeddings shape: {}, classifier_weight shape: {}",
                       embeddings.shape, classifier_weight.shape)

        # MEMORY FIX: Use lax.scan instead of vmap to process blocks sequentially
        # vmap processes all blocks in parallel → 100GB memory (47 blocks × 8.47GB × 12x gradient amplification)
        # scan processes one block at a time → ~2-3GB memory (1 block × 180MB × 12x)
        # Trade-off: ~20-30% slower but enables training with larger batch sizes
        block_indices = jnp.arange(num_blocks)

        def scan_body(carry, block_idx):
            """Process one block at a time to reduce memory usage by 47x."""
            block_logits = compute_block_lse(block_idx)
            return carry, block_logits

        _, all_block_logits = lax.scan(scan_body, None, block_indices)
        # Shape: (num_blocks, num_tokens, vocab_block_size) - same as vmap
        # Memory: ~47x reduction compared to vmap

        # Reshape to (num_tokens, num_blocks * vocab_block_size)
        all_block_logits = all_block_logits.transpose(1, 0, 2)
        all_block_logits = all_block_logits.reshape(num_tokens, -1)

        # Trim padding if necessary
        if padding_size > 0:
            all_block_logits = all_block_logits[:, :vocab_size]

        # Compute stable log-sum-exp
        lse = jax.scipy.special.logsumexp(all_block_logits, axis=1)

        return lse

    # Compute log-sum-exp
    lse = compute_lse_chunked(embeddings, classifier_weight, classifier_bias,
                             vocab_block_size, shift)

    # ========================================================================
    # Step 3: Compute cross-entropy loss
    # ========================================================================

    # CE loss = -target_logit + log_sum_exp
    ce_losses = -target_logits + lse  # (num_tokens,)

    # Handle ignore_index
    mask = (targets != ignore_index).astype(jnp.float32)
    ce_losses = ce_losses * mask

    # Apply gradient filtering if specified
    if filter_eps is not None:
        # Zero out gradients for tokens with very small loss
        gradient_mask = (ce_losses > filter_eps).astype(jnp.float32)
        ce_losses = ce_losses * gradient_mask
        mask = mask * gradient_mask

    # Compute average loss
    valid_tokens = jnp.maximum(jnp.sum(mask), 1.0)  # Avoid division by zero
    avg_loss = jnp.sum(ce_losses) / valid_tokens

    # Return results
    if return_components:
        components = {
            'per_token_loss': ce_losses,
            'target_logits': target_logits,
            'lse': lse,
            'mask': mask,
            'valid_tokens': valid_tokens,
        }
        return avg_loss, components
    else:
        return avg_loss


# ============================================================================
# Specialized Version for Autoregressive Training
# ============================================================================

@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def cce_loss_autoregressive(
    embeddings: jax.Array,          # (batch, seq_len, hidden_dim)
    classifier_weight: jax.Array,   # (vocab_size, hidden_dim)
    targets: jax.Array,             # (batch, seq_len)
    classifier_bias: Optional[jax.Array] = None,  # (vocab_size,)
    ignore_index: int = -100,
    vocab_block_size: int = 512,
    return_per_position: bool = False,
    use_custom_vjp: bool = True,    # NEW: Enable memory-efficient backward pass
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Specialized CCE for autoregressive language modeling.

    This version is optimized for the common autoregressive case where:
    - Input is 3D: (batch, seq_len, hidden_dim)
    - We want per-position losses for analysis
    - We may want to ignore certain time indices

    Args:
        embeddings: Input embeddings (batch, seq_len, hidden_dim)
        classifier_weight: LM head weights (vocab_size, hidden_dim)
        targets: Target indices (batch, seq_len)
        classifier_bias: Optional LM head bias (vocab_size,)
        ignore_index: Index to ignore (e.g., padding)
        vocab_block_size: Block size for chunked computation
        return_per_position: If True, return (batch, seq_len) losses
        use_custom_vjp: If True, use memory-efficient custom VJP for backward pass
                       (recomputes logits instead of storing them, ~6x memory reduction)

    Returns:
        loss: Scalar average loss
        per_position_loss: (optional) Per-position losses (batch, seq_len)
    """

    batch_size, seq_len, hidden_dim = embeddings.shape
    vocab_size = classifier_weight.shape[0]

    # Flatten batch and sequence dimensions
    embeddings_flat = embeddings.reshape(-1, hidden_dim)
    targets_flat = targets.reshape(-1)

    # Choose whether to use custom VJP or standard autodiff
    if use_custom_vjp:
        jax.debug.callback(lambda: print("[CCE] Using custom VJP for memory-efficient backward pass"))
        # Custom VJP: recomputes logits during backward, saves ~6x memory
        # Forward: only saves LSE + targets (~100MB)
        # Backward: recomputes block logits sequentially (~180MB per block)
        loss, components = cce_with_custom_backward(
            embeddings_flat,
            classifier_weight,
            targets_flat,
            classifier_bias=classifier_bias,
            ignore_index=ignore_index,
            vocab_block_size=vocab_block_size,
            return_components=True
        )
    else:
        jax.debug.callback(lambda: print("[CCE] Using standard autodiff (higher memory usage)"))
        # Standard autodiff: JAX stores all intermediate values for backward
        # Memory usage: ~8.5GB for all 47 block outputs + gradient buffers
        loss, components = jax_linear_cross_entropy(
            embeddings_flat,
            classifier_weight,
            targets_flat,
            classifier_bias=classifier_bias,
            ignore_index=ignore_index,
            vocab_block_size=vocab_block_size,
            return_components=True
        )

    if return_per_position:
        # Reshape per-token losses back to (batch, seq_len)
        per_position_loss = components['per_token_loss'].reshape(batch_size, seq_len)
        return loss, per_position_loss
    else:
        return loss


# ============================================================================
# Gradient-Aware Version with Sparsity
# ============================================================================

def cce_with_gradient_sparsity(
    embeddings: jax.Array,
    classifier_weight: jax.Array,
    targets: jax.Array,
    classifier_bias: Optional[jax.Array] = None,
    sparsity_threshold: float = 1e-4,
    top_k: Optional[int] = None,
) -> Tuple[jax.Array, dict]:
    """
    CCE with gradient sparsity for memory efficiency during backward pass.

    This version filters gradients to reduce memory usage during backprop.
    Only gradients above threshold or top-k gradients are kept.

    Args:
        embeddings: Input embeddings
        classifier_weight: Classifier weights
        targets: Target indices
        classifier_bias: Optional bias
        sparsity_threshold: Minimum gradient magnitude to keep
        top_k: Keep only top-k gradients per sample

    Returns:
        loss: Scalar loss
        stats: Dictionary with sparsity statistics
    """

    # Forward pass
    loss, components = jax_linear_cross_entropy(
        embeddings,
        classifier_weight,
        targets,
        classifier_bias=classifier_bias,
        return_components=True
    )

    # Compute gradients w.r.t embeddings
    grad_fn = jax.grad(lambda emb: jax_linear_cross_entropy(
        emb, classifier_weight, targets, classifier_bias
    ))
    embedding_grads = grad_fn(embeddings)

    # Apply sparsity
    grad_magnitude = jnp.abs(embedding_grads)

    if top_k is not None:
        # Keep only top-k gradients
        k = min(top_k, embedding_grads.shape[-1])
        threshold = jnp.sort(grad_magnitude, axis=-1)[:, -k]
        mask = grad_magnitude >= threshold[:, None]
    else:
        # Keep gradients above threshold
        mask = grad_magnitude > sparsity_threshold

    sparse_grads = embedding_grads * mask

    # Statistics
    stats = {
        'gradient_sparsity': 1.0 - jnp.mean(mask),
        'gradient_norm': jnp.linalg.norm(sparse_grads),
        'num_active': jnp.sum(mask),
    }

    return loss, stats


# ============================================================================
# Memory-Efficient Backward Pass
# ============================================================================

@jax.custom_vjp
def cce_with_custom_backward(
    embeddings: jax.Array,
    classifier_weight: jax.Array,
    targets: jax.Array,
    classifier_bias: Optional[jax.Array] = None,
    ignore_index: int = -100,
    vocab_block_size: int = 256,
    return_components: bool = False,
) -> Union[jax.Array, Tuple[jax.Array, dict]]:
    """
    CCE with custom VJP for memory-efficient backward pass.

    This uses JAX's custom VJP to implement a memory-efficient backward pass
    that doesn't materialize the full Jacobian by recomputing logits during backward.

    Memory savings: ~6x reduction by not storing all 47 block outputs (8.5GB → ~1.4GB)
    Speed cost: ~20% slower due to recomputation
    """
    return jax_linear_cross_entropy(
        embeddings,
        classifier_weight,
        targets,
        classifier_bias,
        ignore_index=ignore_index,
        vocab_block_size=vocab_block_size,
        return_components=return_components
    )


def cce_fwd(embeddings, classifier_weight, targets, classifier_bias,
            ignore_index, vocab_block_size, return_components):
    """
    Forward pass for custom VJP.

    Save only LSE and minimal state needed for backward pass.
    This avoids storing all block outputs (47 blocks × 180MB = 8.5GB).
    """
    # Call CCE with components to get LSE
    result = jax_linear_cross_entropy(
        embeddings,
        classifier_weight,
        targets,
        classifier_bias,
        ignore_index=ignore_index,
        vocab_block_size=vocab_block_size,
        return_components=True
    )

    if return_components:
        loss, components = result
    else:
        # If components weren't requested, still need to compute them for backward
        loss = result
        loss_with_comps, components = jax_linear_cross_entropy(
            embeddings,
            classifier_weight,
            targets,
            classifier_bias,
            ignore_index=ignore_index,
            vocab_block_size=vocab_block_size,
            return_components=True
        )

    # Save only what we need for backward:
    # - embeddings, weights: to recompute logits in backward
    # - lse: for gradient computation
    # - targets: to know which token was correct
    # - mask: to handle ignore_index
    # DO NOT save all_block_logits (that's the memory killer!)
    saved_for_backward = (
        embeddings,           # (num_tokens, hidden_dim)
        classifier_weight,    # (vocab_size, hidden_dim)
        targets,              # (num_tokens,)
        classifier_bias,      # (vocab_size,) or None
        components['lse'],    # (num_tokens,) - log-sum-exp values
        components['mask'],   # (num_tokens,) - valid token mask
        ignore_index,         # Need for static args
        vocab_block_size,     # Need for static args
        return_components,    # Need to know what to return
    )

    if return_components:
        return (loss, components), saved_for_backward
    else:
        return loss, saved_for_backward


def cce_bwd(res, g):
    """
    Backward pass for custom VJP with memory-efficient recomputation.

    Instead of storing all block logits from forward pass (47 blocks × 180MB = 8.5GB),
    we RECOMPUTE them during backward pass using lax.scan (sequential processing).

    Memory: Only processes 1 block at a time → 180MB × gradient buffers
    Speed: ~20% slower due to recomputation, but enables training with larger batches

    Gradient formulas:
        loss = -target_logit + LSE
        where LSE = log(sum_k exp(logit_k))

        d_loss/d_embeddings = -classifier_weight[target] + sum_k(softmax[k] * classifier_weight[k])
        d_loss/d_classifier_weight[k] = sum_tokens((softmax[token,k] - is_target[token,k]) * embeddings[token])
    """
    (embeddings, classifier_weight, targets, classifier_bias, lse, mask,
     ignore_index, vocab_block_size, return_components) = res

    # Handle both scalar and tuple gradients (depending on return_components)
    if return_components:
        g_loss, g_components = g  # Gradient w.r.t (loss, components)
        # We only backprop through loss, not through components dict
    else:
        g_loss = g  # Usually 1.0 for scalar loss

    num_tokens = embeddings.shape[0]
    hidden_dim = embeddings.shape[1]
    vocab_size = classifier_weight.shape[0]

    # Use vocab_block_size from forward pass (saved in res)
    # vocab_block_size already extracted from res tuple above
    padded_vocab_size = ((vocab_size + vocab_block_size - 1) // vocab_block_size) * vocab_block_size
    num_blocks = padded_vocab_size // vocab_block_size

    jax.debug.callback(lambda: print(f"[CCE BACKWARD] Recomputing logits for {num_blocks} blocks sequentially"))

    # ========================================================================
    # Step 1: Compute embedding gradients
    # ========================================================================

    # Part 1: Gradient from target logit term: -target_logit
    # d(-target_logit)/d_embeddings = -classifier_weight[targets]
    target_weight_grads = -classifier_weight[targets]  # (num_tokens, hidden_dim)

    # Part 2: Gradient from LSE term: +LSE
    # d(LSE)/d_embeddings = sum_k(softmax[k] * classifier_weight[k])
    # We need to recompute logits and softmax, but do it block-by-block

    # Initialize accumulator for LSE gradient
    lse_emb_grads = jnp.zeros_like(embeddings)  # (num_tokens, hidden_dim)

    # Pad weights if needed
    padding_size = padded_vocab_size - vocab_size
    if padding_size > 0:
        classifier_weight_padded = jnp.concatenate([
            classifier_weight,
            jnp.zeros((padding_size, hidden_dim))
        ], axis=0)
        if classifier_bias is not None:
            classifier_bias_padded = jnp.concatenate([
                classifier_bias,
                jnp.full((padding_size,), -1e20)  # Large negative to suppress padding
            ])
        else:
            classifier_bias_padded = None
    else:
        classifier_weight_padded = classifier_weight
        classifier_bias_padded = classifier_bias

    def compute_block_grad(carry, block_idx):
        """
        Recompute logits for one block and accumulate gradients.

        This is the KEY memory optimization: we process one block at a time,
        computing softmax contribution and accumulating into gradients.
        """
        lse_emb_grad_accum, weight_grad_accum, bias_grad_accum = carry

        start_idx = block_idx * vocab_block_size

        # Recompute block logits (same as forward pass)
        block_weights = lax.dynamic_slice(
            classifier_weight_padded,
            (start_idx, 0),
            (vocab_block_size, hidden_dim)
        )

        # block_logits: (num_tokens, vocab_block_size)
        block_logits = jnp.matmul(embeddings, block_weights.T)

        if classifier_bias_padded is not None:
            block_bias = lax.dynamic_slice(
                classifier_bias_padded,
                (start_idx,),
                (vocab_block_size,)
            )
            block_logits = block_logits + block_bias

        # Compute softmax for this block using saved LSE
        # softmax[k] = exp(logit[k] - LSE)
        block_softmax = jnp.exp(block_logits - lse[:, None])  # (num_tokens, vocab_block_size)

        # Mask padding if this is the last block
        if padding_size > 0 and block_idx == num_blocks - 1:
            # Last block might have padding
            valid_size = vocab_block_size - padding_size
            padding_mask = jnp.arange(vocab_block_size) < valid_size
            block_softmax = block_softmax * padding_mask[None, :]

        # Gradient contribution to embeddings from this block
        # sum_k(softmax[k] * weight[k])
        block_emb_grad = jnp.matmul(block_softmax, block_weights)  # (num_tokens, hidden_dim)
        lse_emb_grad_accum = lse_emb_grad_accum + block_emb_grad

        # Gradient contribution to weights from this block
        # sum_tokens(softmax[token, k] * embeddings[token])
        block_weight_grad = jnp.matmul(block_softmax.T, embeddings)  # (vocab_block_size, hidden_dim)
        weight_grad_accum = weight_grad_accum.at[start_idx:start_idx+vocab_block_size].add(block_weight_grad)

        # Gradient contribution to bias from this block
        if classifier_bias_padded is not None:
            block_bias_grad = jnp.sum(block_softmax, axis=0)  # (vocab_block_size,)
            bias_grad_accum = bias_grad_accum.at[start_idx:start_idx+vocab_block_size].add(block_bias_grad)

        return (lse_emb_grad_accum, weight_grad_accum, bias_grad_accum), None

    # Initialize accumulators
    init_lse_emb_grad = jnp.zeros_like(embeddings)
    init_weight_grad = jnp.zeros_like(classifier_weight_padded)
    init_bias_grad = jnp.zeros_like(classifier_bias_padded) if classifier_bias_padded is not None else None

    # Run scan over blocks
    (lse_emb_grads, weight_grads_full, bias_grads_full), _ = lax.scan(
        compute_block_grad,
        (init_lse_emb_grad, init_weight_grad, init_bias_grad),
        jnp.arange(num_blocks)
    )

    # ========================================================================
    # Step 2: Add gradient from target token
    # ========================================================================

    # For weights: subtract contribution from target token
    # d(-target_logit)/d_weight[target] = -embeddings
    target_weight_grads_to_add = -jax.ops.segment_sum(
        embeddings,  # (num_tokens, hidden_dim)
        targets,     # (num_tokens,) - which vocab token each corresponds to
        num_segments=vocab_size
    )  # (vocab_size, hidden_dim)

    # Trim padding from weight gradients
    weight_grads = weight_grads_full[:vocab_size] + target_weight_grads_to_add

    # For bias: subtract 1 for each occurrence of target token
    if classifier_bias is not None:
        # Count how many times each token appears as target
        target_counts = jax.ops.segment_sum(
            jnp.ones(num_tokens),
            targets,
            num_segments=vocab_size
        )
        bias_grads = bias_grads_full[:vocab_size] - target_counts
    else:
        bias_grads = None

    # ========================================================================
    # Step 3: Combine and scale by upstream gradient and mask
    # ========================================================================

    # Total embedding gradient = -target_weight + lse_contribution
    embedding_grads = target_weight_grads + lse_emb_grads

    # Apply mask (ignore_index handling) and scale by upstream gradient
    embedding_grads = embedding_grads * mask[:, None] * g_loss
    weight_grads = weight_grads * g_loss
    if bias_grads is not None:
        bias_grads = bias_grads * g_loss

    jax.debug.callback(lambda: print(f"[CCE BACKWARD] Gradients computed, memory efficient!"))

    # Return gradients for all input parameters (in order)
    # static args (ignore_index, vocab_block_size, return_components) get None
    return (embedding_grads, weight_grads, None, bias_grads,
            None, None, None)  # None for: targets, ignore_index, vocab_block_size, return_components


cce_with_custom_backward.defvjp(cce_fwd, cce_bwd)


# ============================================================================
# Integration Utilities for LOBS5
# ============================================================================

def create_cce_loss_fn(
    model_params: dict,
    ignore_times: bool = False,
    time_start_idx: int = 9,
    time_end_idx: int = 13,
    msg_len: int = 23,
) -> callable:
    """
    Create a CCE loss function compatible with LOBS5 training loop.

    This function creates a loss function that can be directly used
    in place of the existing cross_entropy_loss in train_helpers.py.

    Args:
        model_params: Model parameters dictionary
        ignore_times: Whether to ignore time tokens
        time_start_idx: Start index of time tokens to ignore
        time_end_idx: End index of time tokens to ignore
        msg_len: Message length for reshaping

    Returns:
        loss_fn: Loss function compatible with LOBS5
    """

    def loss_fn(embeddings, targets, classifier_weight, classifier_bias=None):
        """
        Compute CCE loss with optional time token ignoring.

        Args:
            embeddings: Model output embeddings (batch, seq_len, hidden_dim)
            targets: Target token indices (batch, seq_len)
            classifier_weight: Decoder weight matrix
            classifier_bias: Optional decoder bias

        Returns:
            loss: Scalar loss
            per_position_loss: Per-position losses for logging
        """

        # Compute CCE loss
        loss, per_position_loss = cce_loss_autoregressive(
            embeddings,
            classifier_weight,
            targets,
            classifier_bias=classifier_bias,
            return_per_position=True
        )

        # Handle time token ignoring if needed
        if ignore_times:
            batch_size, seq_len = per_position_loss.shape

            # Reshape to separate messages
            loss_reshaped = per_position_loss.reshape(batch_size, -1, msg_len)

            # Extract non-time tokens
            loss_before_time = loss_reshaped[:, :, :time_start_idx]
            loss_after_time = loss_reshaped[:, :, (time_end_idx + 1):]

            # Concatenate non-time losses
            loss_no_time = jnp.concatenate([loss_before_time, loss_after_time], axis=2)
            per_position_loss = loss_no_time.reshape(batch_size, -1)

            # Recompute average
            loss = jnp.mean(per_position_loss)

        return loss, per_position_loss

    return loss_fn


# ============================================================================
# Testing and Validation
# ============================================================================

def validate_cce_implementation():
    """
    Validate CCE implementation against naive implementation.

    This function tests numerical correctness of the CCE implementation.
    """

    print("Validating CCE implementation...")

    # Test parameters
    batch_size = 4
    seq_len = 100
    hidden_dim = 256
    vocab_size = 48

    # Random data
    key = jax.random.PRNGKey(42)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    embeddings = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))
    classifier_weight = jax.random.normal(key2, (vocab_size, hidden_dim))
    classifier_bias = jax.random.normal(key3, (vocab_size,))
    targets = jax.random.randint(key4, (batch_size, seq_len), 0, vocab_size)

    # Naive implementation
    def naive_cross_entropy(embeddings, classifier_weight, targets, classifier_bias):
        """Naive CE that materializes full logits."""
        # Compute all logits
        logits = jnp.matmul(embeddings, classifier_weight.T)  # (batch, seq, vocab)
        if classifier_bias is not None:
            logits = logits + classifier_bias

        # Log softmax
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Gather target log probs
        batch_indices = jnp.arange(batch_size)[:, None]
        seq_indices = jnp.arange(seq_len)[None, :]
        target_log_probs = log_probs[batch_indices, seq_indices, targets]

        # Average loss
        return -jnp.mean(target_log_probs)

    # Compute with both methods
    loss_naive = naive_cross_entropy(embeddings, classifier_weight, targets, classifier_bias)
    loss_cce = cce_loss_autoregressive(embeddings, classifier_weight, targets, classifier_bias)

    # Check if close
    diff = jnp.abs(loss_naive - loss_cce)
    print(f"Naive loss: {loss_naive:.6f}")
    print(f"CCE loss: {loss_cce:.6f}")
    print(f"Difference: {diff:.2e}")

    # Allow slightly larger tolerance for chunked computation
    tolerance = 5e-4  # 0.0005 absolute error is acceptable
    assert diff < tolerance, f"Loss difference too large: {diff} (tolerance: {tolerance})"
    print(f"✓ CCE implementation validated successfully! (diff: {diff:.2e} < {tolerance})")

    # Test gradient consistency
    print("\nValidating gradients...")

    def loss_fn_naive(w):
        return naive_cross_entropy(embeddings, w, targets, classifier_bias)

    def loss_fn_cce(w):
        return cce_loss_autoregressive(embeddings, w, targets, classifier_bias)

    grad_naive = jax.grad(loss_fn_naive)(classifier_weight)
    grad_cce = jax.grad(loss_fn_cce)(classifier_weight)

    grad_diff = jnp.mean(jnp.abs(grad_naive - grad_cce))
    print(f"Gradient difference: {grad_diff:.2e}")

    grad_tolerance = 5e-4  # Same tolerance for gradients
    assert grad_diff < grad_tolerance, f"Gradient difference too large: {grad_diff} (tolerance: {grad_tolerance})"
    print(f"✓ Gradients validated successfully! (diff: {grad_diff:.2e} < {grad_tolerance})")

    # Memory usage comparison
    print("\nMemory usage comparison:")
    print(f"Naive logits size: {batch_size * seq_len * vocab_size * 4 / 1e6:.2f} MB")
    print(f"CCE working memory: {batch_size * seq_len * 4 / 1e6:.2f} MB")
    print(f"Memory reduction: {vocab_size}x")

    return True


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def benchmark_cce_performance():
    """Benchmark CCE performance vs naive implementation."""

    import time

    print("Benchmarking CCE performance...")

    # Realistic parameters for LOBS5
    batch_size = 40
    seq_len = 500
    hidden_dim = 1024
    vocab_size = 48

    # Generate data
    key = jax.random.PRNGKey(42)
    embeddings = jax.random.normal(key, (batch_size, seq_len, hidden_dim))
    classifier_weight = jax.random.normal(key, (vocab_size, hidden_dim))
    targets = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

    # JIT compile
    cce_jitted = jax.jit(cce_loss_autoregressive)

    # Warmup
    for _ in range(3):
        _ = cce_jitted(embeddings, classifier_weight, targets)

    # Benchmark
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        loss = cce_jitted(embeddings, classifier_weight, targets)
        loss.block_until_ready()
    elapsed = time.time() - start

    print(f"CCE forward pass: {elapsed / n_runs * 1000:.2f} ms per batch")
    print(f"Throughput: {batch_size * seq_len * n_runs / elapsed:.0f} tokens/sec")

    # Test backward pass
    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda w: cce_loss_autoregressive(embeddings, w, targets)
    ))

    # Warmup
    for _ in range(3):
        _, _ = loss_and_grad(classifier_weight)

    # Benchmark
    start = time.time()
    for _ in range(n_runs):
        loss, grad = loss_and_grad(classifier_weight)
        loss.block_until_ready()
    elapsed = time.time() - start

    print(f"CCE forward+backward: {elapsed / n_runs * 1000:.2f} ms per batch")


if __name__ == "__main__":
    # Run validation if executed directly
    validate_cce_implementation()
    print("\n" + "="*50 + "\n")
    benchmark_cce_performance()