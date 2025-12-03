"""
ES RL Training Loop with JaxLOB.

This module implements Evolution Strategies training for trading policies
using JaxLOB as the execution environment.

Architecture:
    ┌──────────────┐        ┌──────────────┐
    │ Frozen S5    │        │ S5 Policy    │←── Perturbed by EGGROLL
    │ World Model  │        │ (generates   │
    │ (exogenous   │        │  message     │
    │  order flow) │        │  as action)  │
    └──────┬───────┘        └──────┬───────┘
           │                       │
           ▼                       ▼
    ┌─────────────────────────────────────────┐
    │              JaxLOB                     │
    │   (order book simulation)               │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Reward/PnL  │ → Fitness score
              └──────────────┘

Key Features:
- Frozen world model predicts background market order flow
- Policy model (ES-trained) generates trading actions
- Policy output = message tokens, directly used as actions
- Fitness = total_revenue (PnL)
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial
import argparse
from tqdm import tqdm
import time
from typing import Tuple, Optional, NamedTuple

# Import centralized utilities
from ..utils.import_utils import get_all_noisers
from ..utils import convert_and_load_checkpoint
from ..models import ES_PaddedLobPredModel
from ..models.common import CommonParams, simple_es_tree_key
from ..envs import JaxLOBESWrapper
from .fitness import compute_pnl_fitness, compute_execution_fitness

all_noisers = get_all_noisers()

__all__ = ['ESRLTrainer', 'create_esrl_config', 'es_rl_train']


class FrozenWorldModel(NamedTuple):
    """Frozen world model container."""
    params: dict
    frozen_params: dict
    es_tree_key: dict
    config: dict


def create_esrl_config():
    """Create argument parser for ES RL training configuration."""
    parser = argparse.ArgumentParser(description='ES RL Training for LOBS5')

    # Model architecture (shared between world model and policy)
    parser.add_argument('--d_model', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--d_output', type=int, default=150, help='Output vocab size')
    parser.add_argument('--d_book', type=int, default=503, help='Book feature dimension')
    parser.add_argument('--n_message_layers', type=int, default=2)
    parser.add_argument('--n_fused_layers', type=int, default=4)
    parser.add_argument('--n_book_pre_layers', type=int, default=1)
    parser.add_argument('--n_book_post_layers', type=int, default=1)
    parser.add_argument('--ssm_size', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=8)

    # World model checkpoint
    parser.add_argument('--world_model_checkpoint', type=str, required=True,
                        help='Path to LOBS5 checkpoint for frozen world model')

    # ES configuration
    parser.add_argument('--noiser', type=str, default='eggroll',
                        choices=['open_es', 'eggroll', 'eggrollbs', 'sparse'])
    parser.add_argument('--sigma', type=float, default=0.01, help='Noise std')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=4, help='LORA rank')

    # Training configuration
    parser.add_argument('--n_threads', type=int, default=128, help='Population size')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--antithetic', type=bool, default=True)

    # JaxLOB environment
    parser.add_argument('--alphatrade_path', type=str, required=True,
                        help='Path to AlphaTrade data')
    parser.add_argument('--task', type=str, default='sell',
                        choices=['sell', 'buy'])
    parser.add_argument('--task_size', type=int, default=500,
                        help='Shares to execute')
    parser.add_argument('--window_index', type=int, default=-1,
                        help='Data window (-1 for random)')

    # Fitness configuration
    parser.add_argument('--fitness_type', type=str, default='pnl',
                        choices=['pnl', 'execution', 'advantage'])
    parser.add_argument('--slippage_weight', type=float, default=0.0)
    parser.add_argument('--vwap_weight', type=float, default=0.0)

    return parser


class ESRLTrainer:
    """
    ES RL Trainer with JaxLOB environment.

    Trains a policy model using Evolution Strategies with
    JaxLOB execution simulation for fitness evaluation.
    """

    def __init__(self, config):
        """
        Initialize ES RL trainer.

        Args:
            config: Namespace with training configuration
        """
        self.config = config

        # Load frozen world model
        self.world_model = self._load_frozen_world_model(
            config.world_model_checkpoint
        )

        # Initialize policy model (same architecture as world model)
        self.policy_init = self._init_policy_model()

        # Initialize noiser
        self.noiser = all_noisers[config.noiser]

        # Initialize JaxLOB environment
        self.env = JaxLOBESWrapper(
            alphatrade_path=config.alphatrade_path,
            task=config.task,
            window_index=config.window_index,
            action_type='delta',
            task_size=config.task_size,
        )

    def _load_frozen_world_model(self, checkpoint_path: str) -> FrozenWorldModel:
        """
        Load frozen world model from LOBS5 gradient-trained checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            FrozenWorldModel with params and config
        """
        es_params, es_tree_key, config = convert_and_load_checkpoint(
            checkpoint_path,
            return_config=True,
        )

        # Extract frozen_params from the init structure
        # For frozen model, we don't need noiser params
        return FrozenWorldModel(
            params=es_params['params'],
            frozen_params=es_params.get('frozen_params', {}),
            es_tree_key=es_tree_key,
            config=config,
        )

    def _init_policy_model(self):
        """
        Initialize policy model with same architecture as world model.

        Returns:
            CommonInit for policy model
        """
        key = jax.random.PRNGKey(42)
        config = self.config

        return ES_PaddedLobPredModel.rand_init(
            key,
            d_model=config.d_model,
            d_output=config.d_output,
            d_book=config.d_book,
            n_message_layers=config.n_message_layers,
            n_fused_layers=config.n_fused_layers,
            n_book_pre_layers=config.n_book_pre_layers,
            n_book_post_layers=config.n_book_post_layers,
            ssm_size=config.ssm_size,
            blocks=config.blocks,
        )

    def _create_noiser_params(self, policy_init):
        """
        Create noiser parameters for policy.

        Args:
            policy_init: CommonInit from policy model

        Returns:
            (frozen_noiser_params, noiser_params)
        """
        frozen_noiser_params = self.noiser.get_frozen_noiser_params(
            policy_init.params,
            policy_init.es_map,
            self.config.sigma,
            lora_rank=self.config.lora_rank,
        )

        noiser_params = self.noiser.get_noiser_params(
            policy_init.params,
            policy_init.es_map,
            frozen_noiser_params,
        )

        return frozen_noiser_params, noiser_params

    def _eval_single_thread(
        self,
        key: jnp.ndarray,
        policy_params: dict,
        noiser_params: dict,
        frozen_noiser_params: dict,
        es_tree_key: dict,
        frozen_params: dict,
        epoch: int,
        thread_id: int,
    ) -> float:
        """
        Evaluate one perturbed policy on a single episode.

        Args:
            key: JAX random key
            policy_params: Policy model parameters
            noiser_params: Noiser parameters
            frozen_noiser_params: Frozen noiser parameters
            es_tree_key: ES tree key structure
            frozen_params: Frozen model parameters
            epoch: Current epoch
            thread_id: Thread ID for noise generation

        Returns:
            Fitness score (PnL)
        """
        iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))

        # Create CommonParams for policy (with noise)
        policy_common_params = CommonParams(
            noiser=self.noiser,
            frozen_noiser_params=frozen_noiser_params,
            noiser_params=noiser_params,
            params=policy_params,
            es_tree_key=es_tree_key,
            frozen_params=frozen_params,
            iterinfo=iterinfo,
        )

        # Reset environment
        key, reset_key = jax.random.split(key)
        obs, state = self.env.reset(reset_key)

        # Run episode
        def step_fn(carry, _):
            key, state, obs, hidden = carry
            key, step_key = jax.random.split(key)

            # World model: predict background order flow (frozen, no noise)
            # world_common_params = CommonParams(
            #     noiser=all_noisers['noop'](),  # No noise
            #     params=self.world_model.params,
            #     frozen_params=self.world_model.frozen_params,
            #     iterinfo=None,
            # )
            # world_messages = ES_PaddedLobPredModel._forward_ar(
            #     world_common_params, obs, ...
            # )

            # Policy: generate action with ES perturbation
            # action_logits = ES_PaddedLobPredModel._forward_ar(
            #     policy_common_params, obs, ...
            # )
            # action = jax.random.categorical(key, action_logits)

            # Placeholder: simple action for now
            action = jnp.zeros(2)

            obs, state, reward, done, info = self.env.step(step_key, state, action)

            return (key, state, obs, hidden), (reward, done)

        max_steps = state.max_steps_in_episode
        (_, final_state, _, _), _ = jax.lax.scan(
            step_fn,
            (key, state, obs, None),
            None,
            length=max_steps,
        )

        # Compute fitness
        metrics = self.env.get_execution_metrics(final_state)

        if self.config.fitness_type == 'pnl':
            return compute_pnl_fitness(metrics['total_revenue'])
        elif self.config.fitness_type == 'execution':
            return compute_execution_fitness(
                metrics['total_revenue'],
                metrics['slippage_rm'],
                metrics['vwap_rm'],
                slippage_weight=self.config.slippage_weight,
                vwap_weight=self.config.vwap_weight,
            )
        else:
            return compute_pnl_fitness(metrics['total_revenue'])

    def train_epoch(
        self,
        key: jnp.ndarray,
        policy_params: dict,
        noiser_params: dict,
        frozen_noiser_params: dict,
        es_tree_key: dict,
        frozen_params: dict,
        epoch: int,
    ) -> Tuple[dict, dict, float]:
        """
        Run one training epoch.

        Args:
            key: JAX random key
            policy_params: Current policy parameters
            noiser_params: Current noiser parameters
            frozen_noiser_params: Frozen noiser parameters
            es_tree_key: ES tree key structure
            frozen_params: Frozen model parameters
            epoch: Current epoch number

        Returns:
            (updated_params, updated_noiser_params, mean_fitness)
        """
        n_threads = self.config.n_threads

        # Evaluate all threads in parallel
        keys = jax.random.split(key, n_threads)
        thread_ids = jnp.arange(n_threads)

        # vmap over threads
        eval_fn = partial(
            self._eval_single_thread,
            policy_params=policy_params,
            noiser_params=noiser_params,
            frozen_noiser_params=frozen_noiser_params,
            es_tree_key=es_tree_key,
            frozen_params=frozen_params,
            epoch=epoch,
        )

        fitnesses = jax.vmap(eval_fn)(keys, thread_ids)

        # ES update
        updated_params, updated_noiser_params = self.noiser.es_grad(
            frozen_noiser_params,
            noiser_params,
            policy_params,
            es_tree_key,
            fitnesses,
            self.config.lr,
        )

        return updated_params, updated_noiser_params, jnp.mean(fitnesses)

    def train(self, n_epochs: Optional[int] = None):
        """
        Run full training loop.

        Args:
            n_epochs: Number of epochs (uses config if None)
        """
        n_epochs = n_epochs or self.config.n_epochs

        # Initialize
        key = jax.random.PRNGKey(0)
        policy_init = self.policy_init
        policy_params = policy_init.params
        es_tree_key = simple_es_tree_key(policy_init.es_map)
        frozen_params = policy_init.frozen_params or {}

        frozen_noiser_params, noiser_params = self._create_noiser_params(policy_init)

        # Training loop
        for epoch in tqdm(range(n_epochs), desc='ES RL Training'):
            key, epoch_key = jax.random.split(key)

            policy_params, noiser_params, mean_fitness = self.train_epoch(
                epoch_key,
                policy_params,
                noiser_params,
                frozen_noiser_params,
                es_tree_key,
                frozen_params,
                epoch,
            )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: mean_fitness = {mean_fitness:.4f}")

        return policy_params, noiser_params


def es_rl_train(config):
    """
    Main entry point for ES RL training.

    Args:
        config: Namespace with training configuration
    """
    trainer = ESRLTrainer(config)
    return trainer.train()


if __name__ == '__main__':
    parser = create_esrl_config()
    args = parser.parse_args()
    es_rl_train(args)
