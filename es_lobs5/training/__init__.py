# ES training infrastructure for LOBS5
"""
Training loop and fitness evaluation for ES-based LOBS5 training.

Two modes:
1. es_train: Next-token prediction with cross-entropy fitness
2. es_rl_train: RL with JaxLOB execution, PnL-based fitness
"""

# Next-token prediction mode
from .fitness import compute_fitness, cross_entropy_loss

# PnL-based fitness for RL mode
from .fitness import (
    compute_pnl_fitness,
    compute_execution_fitness,
    compute_advantage_fitness,
)

# Training loops
from .es_train import es_train, ESTrainer, create_es_config
from .es_rl_train import es_rl_train, ESRLTrainer, create_esrl_config
