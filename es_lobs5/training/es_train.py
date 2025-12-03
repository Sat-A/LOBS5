"""
ES Training Loop for LOBS5 Models.

This module implements the main training loop for Evolution Strategies
based training of LOBS5 S5 models.

Key features:
- Multi-GPU parallel evaluation
- Antithetic sampling
- Integration with HyperscaleES noiser
- Compatible with existing LOBS5 data pipeline
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial
import argparse
from tqdm import tqdm
import time

# Import centralized utilities
from ..utils.import_utils import get_all_noisers

all_noisers = get_all_noisers()

# simple_es_tree_key is imported from common.py which we already set up
from ..models.common import simple_es_tree_key

from ..models import ES_PaddedLobPredModel
from ..models.common import CommonParams
from .fitness import compute_fitness, cross_entropy_loss, accuracy

__all__ = ['es_train', 'create_es_config', 'ESTrainer']


def create_es_config():
    """Create argument parser for ES training configuration."""
    parser = argparse.ArgumentParser(description='ES Training for LOBS5')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--d_output', type=int, default=150, help='Output vocab size')
    parser.add_argument('--d_book', type=int, default=503, help='Book feature dimension')
    parser.add_argument('--n_message_layers', type=int, default=2, help='Message encoder layers')
    parser.add_argument('--n_fused_layers', type=int, default=4, help='Fused encoder layers')
    parser.add_argument('--n_book_pre_layers', type=int, default=1, help='Book pre-layers')
    parser.add_argument('--n_book_post_layers', type=int, default=1, help='Book post-layers')
    parser.add_argument('--ssm_size', type=int, default=256, help='SSM state size')
    parser.add_argument('--blocks', type=int, default=8, help='Number of SSM blocks')

    # SSM configuration
    parser.add_argument('--C_init', type=str, default='trunc_standard_normal')
    parser.add_argument('--discretization', type=str, default='zoh')
    parser.add_argument('--dt_min', type=float, default=0.001)
    parser.add_argument('--dt_max', type=float, default=0.1)
    parser.add_argument('--conj_sym', type=bool, default=True)
    parser.add_argument('--clip_eigs', type=bool, default=True)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--prenorm', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='none', help='Pooling mode')

    # ES configuration
    parser.add_argument('--noiser', type=str, default='eggroll',
                        choices=['open_es', 'eggroll', 'eggrollbs', 'sparse'])
    parser.add_argument('--sigma', type=float, default=0.01, help='Noise standard deviation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=4, help='LORA rank for eggroll')
    parser.add_argument('--threads_per_gpu', type=int, default=64, help='Perturbations per GPU')
    parser.add_argument('--noise_reuse', type=int, default=0, help='Epochs to reuse noise')
    parser.add_argument('--freeze_nonlora', type=bool, default=False)

    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--validate_every', type=int, default=10, help='Validation frequency')
    parser.add_argument('--save_every', type=int, default=50, help='Checkpoint frequency')
    parser.add_argument('--ignore_time_tokens', type=bool, default=True)

    # Data configuration
    parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--msg_seq_len', type=int, default=500, help='Message sequence length')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./es_checkpoints')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    # Precision
    parser.add_argument('--use_bf16', type=bool, default=True)

    # Checkpoint loading
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Path to gradient-trained LOBS5 checkpoint for initialization')

    return parser


class ESTrainer:
    """
    ES Trainer for LOBS5 models.

    Handles model initialization, parallel evaluation, and parameter updates.
    """

    def __init__(self, config):
        """
        Initialize ES trainer.

        Args:
            config: Namespace with training configuration
        """
        self.config = config
        self.dtype = jnp.bfloat16 if config.use_bf16 else jnp.float32

        # Initialize random keys
        self.master_key = jax.random.key(config.seed)
        self.model_key, self.gen_key = jax.random.split(self.master_key)

        # Setup devices and mesh
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        self.mesh = jax.make_mesh((self.num_devices,), ('data',))

        # Total number of parallel evaluations
        self.total_threads = self.num_devices * config.threads_per_gpu

        print(f"ESTrainer initialized with {self.num_devices} devices, "
              f"{self.total_threads} total threads")

        # Initialize model
        self._init_model()

        # Initialize noiser
        self._init_noiser()

        # Compile functions
        self._compile_functions()

    def _init_model(self):
        """Initialize model parameters."""
        config = self.config

        print("Initializing model...")
        model_init = ES_PaddedLobPredModel.rand_init(
            self.model_key,
            d_output=config.d_output,
            d_model=config.d_model,
            d_book=config.d_book,
            n_message_layers=config.n_message_layers,
            n_fused_layers=config.n_fused_layers,
            n_book_pre_layers=config.n_book_pre_layers,
            n_book_post_layers=config.n_book_post_layers,
            ssm_size=config.ssm_size,
            blocks=config.blocks,
            C_init=config.C_init,
            discretization=config.discretization,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            conj_sym=config.conj_sym,
            clip_eigs=config.clip_eigs,
            bidirectional=config.bidirectional,
            activation=config.activation,
            prenorm=config.prenorm,
            mode=config.mode,
            dtype=self.dtype,
        )

        self.frozen_params = model_init.frozen_params
        self.params = model_init.params
        self.scan_map = model_init.scan_map
        self.es_map = model_init.es_map

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"Model initialized with {param_count:,} parameters")

        # Generate ES keys
        self.es_keys = simple_es_tree_key(self.params, self.model_key, self.scan_map)

    def _init_noiser(self):
        """Initialize noiser for ES updates."""
        config = self.config

        print(f"Initializing noiser: {config.noiser}")
        NOISER = all_noisers[config.noiser]
        self.noiser_cls = NOISER

        noiser_kwargs = {
            'sigma': config.sigma,
            'lr': config.lr,
            'noise_reuse': config.noise_reuse,
            'freeze_nonlora': config.freeze_nonlora,
        }

        if config.noiser in ['eggroll', 'eggrollbs']:
            noiser_kwargs['rank'] = config.lora_rank

        self.frozen_noiser_params, self.noiser_params = NOISER.init_noiser(
            self.params, **noiser_kwargs
        )

        print(f"Noiser initialized with sigma={config.sigma}, lr={config.lr}")

    def _compile_functions(self):
        """Compile JIT functions for training."""
        config = self.config
        NOISER = self.noiser_cls

        # Thread indices for parallel evaluation
        self.thread_indices = jnp.arange(self.total_threads)

        def _eval_single(params, x_m, x_b, labels, thread_id, epoch):
            """Evaluate one perturbed model on one sample."""
            iterinfo = (jnp.int32(epoch), jnp.int32(thread_id))

            # Create common params
            common_params = CommonParams(
                noiser=NOISER,
                frozen_noiser_params=self.frozen_noiser_params,
                noiser_params=self.noiser_params,
                frozen_params=self.frozen_params,
                params=params,
                es_tree_key=self.es_keys,
                iterinfo=iterinfo,
            )

            # Forward pass
            log_probs = ES_PaddedLobPredModel._forward_ar(common_params, x_m, x_b)

            # Compute fitness
            fitness = compute_fitness(
                log_probs, labels,
                ignore_time_tokens=config.ignore_time_tokens
            )
            return fitness

        # Vmap over threads
        self._eval_threads = jax.vmap(
            _eval_single,
            in_axes=(None, None, None, None, 0, None)
        )

        # JIT compile
        self.eval_batch = jax.jit(self._eval_threads)

        def _do_update(noiser_params, params, fitnesses, epoch):
            """Update parameters based on fitness."""
            iterinfos = (
                jnp.full(self.total_threads, epoch, dtype=jnp.int32),
                self.thread_indices
            )

            # Convert fitnesses
            normalized_fitnesses = NOISER.convert_fitnesses(
                self.frozen_noiser_params, noiser_params, fitnesses
            )

            # Update parameters
            noiser_params, new_params = NOISER.do_updates(
                self.frozen_noiser_params, noiser_params, params,
                self.es_keys, normalized_fitnesses, iterinfos, self.es_map
            )
            return noiser_params, new_params

        self.do_update = jax.jit(_do_update)

        print("Functions compiled")

    def train_step(self, batch_data, epoch):
        """
        Execute one training step.

        Args:
            batch_data: Tuple of (x_m, x_b, labels)
            epoch: Current epoch number

        Returns:
            Dict with training metrics
        """
        x_m, x_b, labels = batch_data

        # Evaluate all perturbations
        fitnesses = self.eval_batch(
            self.params, x_m, x_b, labels,
            self.thread_indices, epoch
        )

        # Update parameters
        self.noiser_params, self.params = self.do_update(
            self.noiser_params, self.params, fitnesses, epoch
        )

        # Compute metrics
        mean_fitness = jnp.mean(fitnesses)
        max_fitness = jnp.max(fitnesses)
        min_fitness = jnp.min(fitnesses)
        std_fitness = jnp.std(fitnesses)

        return {
            'mean_fitness': float(mean_fitness),
            'max_fitness': float(max_fitness),
            'min_fitness': float(min_fitness),
            'std_fitness': float(std_fitness),
        }

    def validate(self, val_data):
        """
        Validate model (without noise).

        Args:
            val_data: Validation data tuple (x_m, x_b, labels)

        Returns:
            Dict with validation metrics
        """
        x_m, x_b, labels = val_data

        # Create common params with iterinfo=None (no noise)
        common_params = CommonParams(
            noiser=self.noiser_cls,
            frozen_noiser_params=self.frozen_noiser_params,
            noiser_params=self.noiser_params,
            frozen_params=self.frozen_params,
            params=self.params,
            es_tree_key=self.es_keys,
            iterinfo=None,  # No noise for validation
        )

        # Forward pass
        log_probs = ES_PaddedLobPredModel._forward_ar(common_params, x_m, x_b)

        # Compute metrics
        ce = cross_entropy_loss(log_probs, labels)
        acc = accuracy(log_probs, labels)

        return {
            'val_loss': float(ce),
            'val_accuracy': float(acc),
        }

    def save_checkpoint(self, epoch, path):
        """Save checkpoint."""
        import pickle

        checkpoint = {
            'epoch': epoch,
            'params': self.params,
            'noiser_params': self.noiser_params,
            'config': vars(self.config),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load checkpoint."""
        import pickle

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint['params']
        self.noiser_params = checkpoint['noiser_params']

        return checkpoint['epoch']

    def load_from_flax_checkpoint(self, checkpoint_path: str):
        """
        Load parameters from a gradient-trained LOBS5 checkpoint.

        Converts Flax params to ES format and re-initializes the noiser.

        Args:
            checkpoint_path: Path to Orbax checkpoint directory
        """
        from ..utils.checkpoint_converter import load_params_for_es_trainer

        print(f"\nLoading gradient-trained checkpoint from: {checkpoint_path}")

        # Load and convert checkpoint
        es_params, ckpt_config = load_params_for_es_trainer(
            checkpoint_path,
            trainer_config=self.config
        )

        # Replace current params
        self.params = es_params

        # Re-generate ES keys for new params
        self.es_keys = simple_es_tree_key(self.params, self.model_key, self.scan_map)

        # Re-initialize noiser with new params structure
        config = self.config
        noiser_kwargs = {
            'sigma': config.sigma,
            'lr': config.lr,
            'noise_reuse': config.noise_reuse,
            'freeze_nonlora': config.freeze_nonlora,
        }

        if config.noiser in ['eggroll', 'eggrollbs']:
            noiser_kwargs['rank'] = config.lora_rank

        self.frozen_noiser_params, self.noiser_params = self.noiser_cls.init_noiser(
            self.params, **noiser_kwargs
        )

        # Re-compile functions with new params
        self._compile_functions()

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"Loaded checkpoint with {param_count:,} parameters")
        print(f"Checkpoint config: d_model={ckpt_config.get('d_model')}, "
              f"n_message_layers={ckpt_config.get('n_message_layers')}, "
              f"n_fused_layers={ckpt_config.get('n_fused_layers')}")


def es_train(config):
    """
    Main ES training function.

    Args:
        config: Training configuration (from argparse or dict)
    """
    # Convert dict to namespace if needed
    if isinstance(config, dict):
        config = argparse.Namespace(**config)

    # Initialize trainer
    trainer = ESTrainer(config)

    # Load from gradient-trained checkpoint if provided
    if hasattr(config, 'init_checkpoint') and config.init_checkpoint:
        trainer.load_from_flax_checkpoint(config.init_checkpoint)

    # Initialize wandb if requested
    if config.wandb_project:
        import wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config),
        )

    # Create dummy data for testing (replace with actual dataloader)
    print("Creating dummy data for testing...")
    dummy_x_m = jnp.zeros((config.msg_seq_len * 24,), dtype=jnp.int32)
    dummy_x_b = jnp.zeros((config.msg_seq_len, config.d_book), dtype=trainer.dtype)
    dummy_labels = jnp.zeros((config.msg_seq_len * 24,), dtype=jnp.int32)

    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    for epoch in tqdm(range(config.num_epochs)):
        start_time = time.time()

        # Training step
        metrics = trainer.train_step(
            (dummy_x_m, dummy_x_b, dummy_labels),
            epoch
        )

        elapsed = time.time() - start_time

        # Log metrics
        if config.wandb_project:
            import wandb
            wandb.log({**metrics, 'epoch': epoch, 'time_per_epoch': elapsed})

        # Print progress
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}: fitness={metrics['mean_fitness']:.4f} "
                  f"(+/-{metrics['std_fitness']:.4f}), time={elapsed:.2f}s")

        # Validation
        if epoch % config.validate_every == 0 and epoch > 0:
            val_metrics = trainer.validate((dummy_x_m, dummy_x_b, dummy_labels))
            print(f"  Validation: loss={val_metrics['val_loss']:.4f}, "
                  f"acc={val_metrics['val_accuracy']:.4f}")

            if config.wandb_project:
                import wandb
                wandb.log(val_metrics)

        # Save checkpoint
        if epoch % config.save_every == 0 and epoch > 0:
            ckpt_path = os.path.join(
                config.output_dir,
                f'checkpoint_epoch_{epoch}.pkl'
            )
            trainer.save_checkpoint(epoch, ckpt_path)

    print("\nTraining complete!")

    # Save final checkpoint
    final_path = os.path.join(config.output_dir, 'checkpoint_final.pkl')
    trainer.save_checkpoint(config.num_epochs, final_path)


if __name__ == '__main__':
    parser = create_es_config()
    args = parser.parse_args()
    es_train(args)
