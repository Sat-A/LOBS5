"""
ES-compatible S5 State Space Model.

This module provides an ES-compatible implementation of the S5 SSM layer
that works with the HyperscaleES noiser framework.

The key challenge is handling complex-valued parameters (Lambda, B, C)
which are stored as real tensors with shape (..., 2) for (real, imag).

Based on: s5/ssm.py
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh
from functools import partial

from .common import (
    Model, CommonInit, CommonParams,
    PARAM, MM_PARAM, EXCLUDED,
    merge_inits, merge_frozen, call_submodule,
    ES_Parameter,
)

__all__ = [
    'ES_S5SSM',
    'make_DPLR_HiPPO',
    'init_hippo_matrices',
    'discretize_zoh',
    'discretize_bilinear',
    'apply_ssm',
]


# =============================================================================
# HiPPO Initialization (from s5/ssm_init.py)
# =============================================================================

def make_HiPPO(N):
    """Create a HiPPO-LegS matrix."""
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """Make components for NPLR representation of HiPPO-LegS."""
    hippo = make_HiPPO(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Make components for DPLR representation of HiPPO-LegS.

    Returns:
        Lambda: eigenvalues (complex)
        P: low-rank term
        B: conjugated input matrix
        V: eigenvectors
        B_orig: original B
    """
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def init_hippo_matrices(ssm_size, blocks, conj_sym=True):
    """
    Initialize HiPPO matrices for S5.

    Args:
        ssm_size: Total state space size (H * blocks)
        blocks: Number of SSM blocks (J)
        conj_sym: Whether to enforce conjugate symmetry

    Returns:
        Lambda_re_init, Lambda_im_init, V, Vinv as jnp arrays
    """
    block_size = ssm_size // blocks
    Lambda_list, V_list = [], []

    for _ in range(blocks):
        Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
        if conj_sym:
            # Take only upper half (conjugate pairs)
            Lambda = Lambda[:block_size // 2]
            V = V[:, :block_size // 2]
        Lambda_list.append(Lambda)
        V_list.append(V)

    # Stack blocks diagonally
    Lambda = jnp.concatenate(Lambda_list)
    V = jax.scipy.linalg.block_diag(*V_list)
    Vinv = V.conj().T

    Lambda_re_init = Lambda.real
    Lambda_im_init = Lambda.imag

    return Lambda_re_init, Lambda_im_init, V, Vinv


# =============================================================================
# Discretization Functions
# =============================================================================

def discretize_bilinear(Lambda, B_tilde, Delta):
    """
    Discretize continuous-time SSM using bilinear transform.

    Args:
        Lambda: diagonal state matrix eigenvalues (P,) complex
        B_tilde: input matrix (P, H) complex
        Delta: discretization step sizes (P,) real

    Returns:
        Lambda_bar, B_bar (discretized)
    """
    Identity = jnp.ones(Lambda.shape[0])
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """
    Discretize continuous-time SSM using zero-order hold.

    Args:
        Lambda: diagonal state matrix eigenvalues (P,) complex
        B_tilde: input matrix (P, H) complex
        Delta: discretization step sizes (P,) real

    Returns:
        Lambda_bar, B_bar (discretized)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# =============================================================================
# Parallel Scan Operations
# =============================================================================

@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """
    Compute SSM output using parallel scan.

    Args:
        Lambda_bar: discretized diagonal state matrix (P,) complex
        B_bar: discretized input matrix (P, H) complex
        C_tilde: output matrix (H, P) complex
        input_sequence: input (L, H) float
        conj_sym: whether conjugate symmetry is enforced
        bidirectional: whether to use bidirectional processing

    Returns:
        output sequence (L, H) float
    """
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(
            binary_operator, (Lambda_elements, Bu_elements), reverse=True
        )
        xs = jnp.concatenate((xs, xs2), axis=-1)

    if conj_sym:
        return jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


# =============================================================================
# Parameter Initialization Helpers
# =============================================================================

def init_log_steps(key, P, dt_min, dt_max):
    """Initialize learnable timescale parameters."""
    log_steps = []
    for i in range(P):
        key, skey = jax.random.split(key)
        log_step = jax.random.uniform(skey, (1,)) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)
        log_steps.append(log_step)
    return jnp.array(log_steps)


def init_VinvB(key, shape, Vinv, dtype=jnp.float32):
    """
    Initialize B_tilde = V^{-1} @ B.

    Returns tensor of shape (P, H, 2) storing real and imag parts.
    """
    P, H = shape
    B = lecun_normal()(key, (P, H))
    VinvB = Vinv @ B
    VinvB_real = VinvB.real.astype(dtype)
    VinvB_imag = VinvB.imag.astype(dtype)
    return jnp.stack([VinvB_real, VinvB_imag], axis=-1)


def init_CV(key, shape, V, dtype=jnp.float32):
    """
    Initialize C_tilde = C @ V.

    Returns tensor of shape (H, P, 2) storing real and imag parts.
    """
    H, P = shape

    # Sample C as (H, P, 2) with lecun_normal per row
    Cs = []
    for i in range(H):
        key, skey = jax.random.split(key)
        C_row = lecun_normal()(skey, (1, P, 2))
        Cs.append(C_row)
    C_ = jnp.concatenate(Cs, axis=0)  # (H, P, 2)

    # Reconstruct complex and multiply by V
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real.astype(dtype)
    CV_imag = CV.imag.astype(dtype)
    return jnp.stack([CV_real, CV_imag], axis=-1)


# =============================================================================
# ES_S5SSM Model
# =============================================================================

class ES_S5SSM(Model):
    """
    ES-compatible S5 State Space Model.

    Parameters are classified for ES as follows:
    - Lambda_re, Lambda_im: EXCLUDED (stability critical)
    - log_step: EXCLUDED (stability critical)
    - B: PARAM (safe to perturb)
    - C: MM_PARAM (LORA for efficiency, used in matmul)
    - D: PARAM (simple feedthrough)

    Complex parameters (B, C) are stored as (..., 2) real tensors.
    """

    @classmethod
    def rand_init(
        cls,
        key,
        H: int,
        P: int,
        Lambda_re_init: jnp.ndarray,
        Lambda_im_init: jnp.ndarray,
        V: jnp.ndarray,
        Vinv: jnp.ndarray,
        C_init: str = 'trunc_standard_normal',
        discretization: str = 'zoh',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conj_sym: bool = True,
        clip_eigs: bool = True,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        dtype=jnp.float32,
    ) -> CommonInit:
        """
        Initialize S5 SSM parameters.

        Args:
            key: JAX random key
            H: Feature dimension (hidden size)
            P: State space dimension
            Lambda_re_init, Lambda_im_init: Initial eigenvalues (from HiPPO)
            V, Vinv: Eigenvector matrices (from HiPPO)
            C_init: C matrix initialization method
            discretization: 'zoh' or 'bilinear'
            dt_min, dt_max: Timescale range
            conj_sym: Conjugate symmetry (reduces params by half)
            clip_eigs: Clip eigenvalues to left-half plane
            bidirectional: Use bidirectional processing
            step_rescale: Timescale scaling factor
            dtype: Data type

        Returns:
            CommonInit with params, frozen_params, es_map, scan_map
        """
        keys = jax.random.split(key, 4)

        local_P = 2 * P if conj_sym else P

        # Initialize B (input matrix)
        B = init_VinvB(keys[0], (local_P, H), Vinv, dtype)

        # Initialize C (output matrix)
        if bidirectional:
            C_shape = (H, 2 * local_P)
        else:
            C_shape = (H, local_P)

        C = init_CV(keys[1], C_shape, V, dtype)

        # Initialize D (feedthrough)
        D = (jax.random.normal(keys[2], (H,)) * 1.0).astype(dtype)

        # Initialize log_step
        log_step = init_log_steps(keys[3], P, dt_min, dt_max).astype(dtype)

        # Build params dict
        params = {
            'Lambda_re': Lambda_re_init.astype(dtype),
            'Lambda_im': Lambda_im_init.astype(dtype),
            'B': B,
            'C': C,
            'D': D,
            'log_step': log_step,
        }

        # ES map: which params get which perturbation type
        es_map = {
            'Lambda_re': EXCLUDED,  # Stability critical
            'Lambda_im': EXCLUDED,  # Stability critical
            'B': PARAM,             # Safe to perturb
            'C': PARAM,             # Note: Could use MM_PARAM but C is used in complex matmul
            'D': PARAM,             # Simple feedthrough
            'log_step': EXCLUDED,   # Stability critical
        }

        # Scan map: no vmap dimensions for these params
        scan_map = {k: () for k in params}

        # Frozen params (not learned, but needed for forward)
        frozen_params = {
            'V': V,
            'Vinv': Vinv,
            'H': H,
            'P': P,
            'conj_sym': conj_sym,
            'clip_eigs': clip_eigs,
            'bidirectional': bidirectional,
            'discretization': discretization,
            'step_rescale': step_rescale,
        }

        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params: CommonParams, x):
        """
        Compute S5 SSM forward pass.

        Args:
            common_params: CommonParams with noiser and params
            x: Input sequence (L, H) float

        Returns:
            Output sequence (L, H) float
        """
        fp = common_params.frozen_params
        noiser = common_params.noiser

        # Get (potentially noisy) parameters
        def get_param(name):
            param = common_params.params[name]
            es_key = common_params.es_tree_key[name]
            if common_params.iterinfo is None:
                return param
            return noiser.get_noisy_standard(
                common_params.frozen_noiser_params,
                common_params.noiser_params,
                param, es_key, common_params.iterinfo
            )

        Lambda_re = get_param('Lambda_re')
        Lambda_im = get_param('Lambda_im')
        B = get_param('B')
        C = get_param('C')
        D = get_param('D')
        log_step = get_param('log_step')

        # Reconstruct complex eigenvalues
        Lambda = Lambda_re + 1j * Lambda_im
        if fp['clip_eigs']:
            Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

        # Reconstruct complex B and C from (real, imag) storage
        B_tilde = B[..., 0] + 1j * B[..., 1]
        C_tilde = C[..., 0] + 1j * C[..., 1]

        # Compute discretization step
        step = fp['step_rescale'] * jnp.exp(log_step[:, 0])

        # Discretize
        if fp['discretization'] == 'zoh':
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        else:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

        # Cast input to float32 for SSM computation
        input_dtype = x.dtype
        x_fp32 = x.astype(jnp.float32)

        # Apply SSM
        ys = apply_ssm(
            Lambda_bar, B_bar, C_tilde, x_fp32,
            fp['conj_sym'], fp['bidirectional']
        )

        # Add feedthrough
        Du = jax.vmap(lambda u: D * u)(x_fp32)
        output = ys + Du

        return output.astype(input_dtype)

    @classmethod
    def _forward_rnn(cls, common_params: CommonParams, hidden, x, resets=None):
        """
        Compute S5 SSM forward pass in RNN mode (step-by-step).

        Args:
            common_params: CommonParams with noiser and params
            hidden: Hidden state (1, P) complex
            x: Input sequence (L, H) float
            resets: Optional reset signals (L,) bool

        Returns:
            new_hidden, output_sequence
        """
        fp = common_params.frozen_params
        noiser = common_params.noiser

        # Get (potentially noisy) parameters (same as _forward)
        def get_param(name):
            param = common_params.params[name]
            es_key = common_params.es_tree_key[name]
            if common_params.iterinfo is None:
                return param
            return noiser.get_noisy_standard(
                common_params.frozen_noiser_params,
                common_params.noiser_params,
                param, es_key, common_params.iterinfo
            )

        Lambda_re = get_param('Lambda_re')
        Lambda_im = get_param('Lambda_im')
        B = get_param('B')
        C = get_param('C')
        D = get_param('D')
        log_step = get_param('log_step')

        # Reconstruct complex
        Lambda = Lambda_re + 1j * Lambda_im
        if fp['clip_eigs']:
            Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

        B_tilde = B[..., 0] + 1j * B[..., 1]
        C_tilde = C[..., 0] + 1j * C[..., 1]

        # Discretize
        step = fp['step_rescale'] * jnp.exp(log_step[:, 0])
        if fp['discretization'] == 'zoh':
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        else:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

        # RNN scan
        Lambda_elements = Lambda_bar * jnp.ones((x.shape[0], Lambda_bar.shape[0]))
        Bu_elements = jax.vmap(lambda u: B_bar @ u)(x.astype(jnp.float32))

        # Prepend hidden state
        Lambda_elements = jnp.concatenate([
            jnp.ones((1, Lambda_bar.shape[0])),
            Lambda_elements,
        ])
        Bu_elements = jnp.concatenate([hidden, Bu_elements])

        if resets is not None:
            # Handle resets with modified binary operator
            resets = jnp.concatenate([jnp.zeros(1), resets])

            @jax.vmap
            def binary_operator_reset(q_i, q_j):
                A_i, b_i, c_i = q_i
                A_j, b_j, c_j = q_j
                return (
                    (A_j * A_i) * (1 - c_j) + A_j * c_j,
                    (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
                    c_i * (1 - c_j) + c_j,
                )

            _, xs, _ = jax.lax.associative_scan(
                binary_operator_reset,
                (Lambda_elements, Bu_elements, resets)
            )
        else:
            _, xs = jax.lax.associative_scan(
                binary_operator, (Lambda_elements, Bu_elements)
            )

        # Extract hidden state and outputs
        hidden_out = xs[jnp.newaxis, -1]
        xs = xs[1:]

        # Apply C matrix
        if fp['conj_sym']:
            ys = jax.vmap(lambda s: 2 * (C_tilde @ s).real)(xs)
        else:
            ys = jax.vmap(lambda s: (C_tilde @ s).real)(xs)

        # Add feedthrough
        Du = jax.vmap(lambda u: D * u)(x.astype(jnp.float32))
        output = ys + Du

        return hidden_out, output.astype(x.dtype)
