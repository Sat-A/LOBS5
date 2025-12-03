"""
Fitness evaluation functions for ES training.

In Evolution Strategies, we need a fitness function that maps model outputs
to a scalar score. Higher scores indicate better performance.

For LOB prediction, we use negative cross-entropy loss as fitness.
"""

import jax
import jax.numpy as jnp
from functools import partial

__all__ = [
    'compute_fitness',
    'cross_entropy_loss',
    'accuracy',
    'per_token_cross_entropy',
    # PnL-based fitness for RL
    'compute_pnl_fitness',
    'compute_execution_fitness',
    'compute_advantage_fitness',
    'compute_normalized_pnl_fitness',
]


def cross_entropy_loss(log_probs, labels, mask=None):
    """
    Compute cross-entropy loss.

    Args:
        log_probs: Log probabilities (..., vocab_size)
        labels: Target labels (...,) int32
        mask: Optional mask (...,) to ignore certain positions

    Returns:
        Scalar loss (mean over all positions)
    """
    # Get log prob for correct label
    log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])
    labels_flat = labels.ravel()

    # Gather log probs at label positions
    ce = -jnp.take_along_axis(
        log_probs_flat,
        labels_flat[:, None],
        axis=-1
    ).squeeze(-1)

    if mask is not None:
        mask_flat = mask.ravel()
        ce = ce * mask_flat
        return jnp.sum(ce) / (jnp.sum(mask_flat) + 1e-8)
    else:
        return jnp.mean(ce)


def per_token_cross_entropy(log_probs, labels):
    """
    Compute per-token cross-entropy (no reduction).

    Args:
        log_probs: Log probabilities (L, vocab_size)
        labels: Target labels (L,) int32

    Returns:
        Per-token cross-entropy (L,)
    """
    return -jnp.take_along_axis(
        log_probs,
        labels[:, None],
        axis=-1
    ).squeeze(-1)


def accuracy(log_probs, labels, mask=None):
    """
    Compute prediction accuracy.

    Args:
        log_probs: Log probabilities (..., vocab_size)
        labels: Target labels (...,) int32
        mask: Optional mask

    Returns:
        Scalar accuracy
    """
    preds = jnp.argmax(log_probs, axis=-1)
    correct = (preds == labels).astype(jnp.float32)

    if mask is not None:
        correct = correct * mask
        return jnp.sum(correct) / (jnp.sum(mask) + 1e-8)
    else:
        return jnp.mean(correct)


def create_time_token_mask(seq_len, msg_len=24, time_token_start=9, time_token_end=14):
    """
    Create a mask that excludes time tokens.

    In LOB tokenization, tokens 9-13 (0-indexed) within each message
    are time-related and may be masked during evaluation.

    Args:
        seq_len: Total sequence length (L_m * msg_len)
        msg_len: Number of tokens per message (default 24)
        time_token_start: Start of time tokens within message (inclusive)
        time_token_end: End of time tokens within message (exclusive)

    Returns:
        Mask (seq_len,) with 0 at time token positions, 1 elsewhere
    """
    token_pos = jnp.arange(seq_len) % msg_len
    mask = ~((token_pos >= time_token_start) & (token_pos < time_token_end))
    return mask.astype(jnp.float32)


def compute_fitness(
    log_probs,
    labels,
    ignore_time_tokens: bool = False,
    msg_len: int = 24,
    time_token_range: tuple = (9, 14),
):
    """
    Compute fitness score for ES training.

    Fitness = -cross_entropy (higher is better)

    Args:
        log_probs: Model log probabilities (L, vocab_size) or (vocab_size,)
        labels: Target labels (L,) or scalar int32
        ignore_time_tokens: Whether to mask out time tokens
        msg_len: Tokens per message (for time masking)
        time_token_range: (start, end) of time tokens to mask

    Returns:
        Scalar fitness score (higher is better)
    """
    # Ensure 2D
    if log_probs.ndim == 1:
        log_probs = log_probs[None, :]
        labels = jnp.atleast_1d(labels)

    # Create mask if needed
    mask = None
    if ignore_time_tokens:
        mask = create_time_token_mask(
            log_probs.shape[0],
            msg_len,
            time_token_range[0],
            time_token_range[1]
        )

    # Compute cross-entropy
    ce = cross_entropy_loss(log_probs, labels, mask)

    # Fitness is negative loss (higher is better)
    return -ce


def compute_batch_fitness(
    model_forward_fn,
    params,
    batch_data,
    common_params_template,
    ignore_time_tokens: bool = False,
):
    """
    Compute fitness over a batch of data.

    Args:
        model_forward_fn: Model forward function
        params: Model parameters
        batch_data: Tuple of (x_m, x_b, labels, ...)
        common_params_template: Template CommonParams (with noiser, etc.)
        ignore_time_tokens: Whether to mask time tokens

    Returns:
        Mean fitness over batch
    """
    x_m, x_b, labels = batch_data[:3]

    # Update common_params with actual params
    common_params = common_params_template._replace(params=params)

    # Forward pass
    log_probs = model_forward_fn(common_params, x_m, x_b)

    # Compute fitness
    return compute_fitness(log_probs, labels, ignore_time_tokens)


# =============================================================================
# Fitness Normalization (from noiser)
# =============================================================================

def normalize_fitness_zscore(fitnesses, eps=1e-8):
    """
    Z-score normalize fitness values.

    This is typically done inside the noiser, but provided here for reference.

    Args:
        fitnesses: Raw fitness values (N,)
        eps: Small constant for numerical stability

    Returns:
        Normalized fitness values (N,)
    """
    mean = jnp.mean(fitnesses)
    std = jnp.std(fitnesses)
    return (fitnesses - mean) / (std + eps)


def normalize_fitness_rank(fitnesses):
    """
    Rank-based fitness normalization.

    Maps fitness values to their ranks, which can be more robust
    to outliers than z-score normalization.

    Args:
        fitnesses: Raw fitness values (N,)

    Returns:
        Rank-normalized fitness values (N,) in [-1, 1]
    """
    N = fitnesses.shape[0]
    ranks = jnp.argsort(jnp.argsort(fitnesses))  # Get ranks
    # Map ranks to [-1, 1]
    normalized = 2.0 * ranks / (N - 1) - 1.0
    return normalized


# =============================================================================
# PnL-based Fitness (for RL with JaxLOB)
# =============================================================================

def compute_pnl_fitness(total_revenue: float) -> float:
    """
    Compute fitness from JaxLOB execution total revenue.

    This is the primary fitness function for ES RL training.
    Higher revenue = better policy.

    Args:
        total_revenue: EnvState.total_revenue from JaxLOB

    Returns:
        Fitness score (same as total_revenue, higher is better)
    """
    return total_revenue


def compute_execution_fitness(
    total_revenue: float,
    slippage_rm: float = 0.0,
    vwap_rm: float = 0.0,
    init_price: float = 0.0,
    quant_executed: int = 0,
    slippage_weight: float = 0.0,
    vwap_weight: float = 0.0,
) -> float:
    """
    Compute combined execution quality fitness.

    Combines multiple execution metrics into a single fitness score.

    Args:
        total_revenue: Total revenue from execution
        slippage_rm: Rolling mean slippage
        vwap_rm: Rolling mean VWAP
        init_price: Initial price at episode start
        quant_executed: Total quantity executed
        slippage_weight: Weight for slippage penalty (0 = ignore)
        vwap_weight: Weight for VWAP deviation penalty (0 = ignore)

    Returns:
        Combined fitness score (higher is better)
    """
    fitness = total_revenue

    # Optionally penalize slippage
    if slippage_weight > 0:
        fitness = fitness - slippage_weight * jnp.abs(slippage_rm)

    # Optionally penalize VWAP deviation
    if vwap_weight > 0 and quant_executed > 0:
        avg_price = total_revenue / quant_executed
        vwap_deviation = jnp.abs(avg_price - vwap_rm)
        fitness = fitness - vwap_weight * vwap_deviation * quant_executed

    return fitness


def compute_advantage_fitness(
    total_revenue: float,
    vwap_rm: float,
    quant_executed: int,
) -> float:
    """
    Compute advantage over VWAP as fitness.

    Measures how much better/worse the policy performed compared
    to the market VWAP.

    Args:
        total_revenue: Total revenue from execution
        vwap_rm: Rolling mean VWAP
        quant_executed: Total quantity executed

    Returns:
        Advantage fitness (positive = beat VWAP, negative = underperformed)
    """
    if quant_executed == 0:
        return 0.0
    return total_revenue - vwap_rm * quant_executed


def compute_normalized_pnl_fitness(
    total_revenue: float,
    task_size: int,
    init_price: float,
    scale: float = 10000.0,
) -> float:
    """
    Compute normalized PnL fitness.

    Normalizes revenue by task size and initial price for comparable
    fitness across different market conditions.

    Args:
        total_revenue: Total revenue from execution
        task_size: Target execution quantity
        init_price: Initial price (tick-adjusted)
        scale: Normalization scale factor

    Returns:
        Normalized fitness in reasonable range
    """
    expected_revenue = task_size * init_price
    if expected_revenue == 0:
        return 0.0
    return (total_revenue - expected_revenue) / scale
