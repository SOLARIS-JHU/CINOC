import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

class SparsePolynomialLayer(nn.Module):
    """
    A Flax layer that applies a sparse mask to polynomial features.
    This is the JAX equivalent of the L0Dense logic.
    """
    out_features: int = 1
    
    @nn.compact
    def __call__(self, x):
        # x is the polynomial feature vector
        # We learn weights and a 'gate' for sparsity
        weights = self.param('weights', 
                            nn.initializers.glorot_uniform(), 
                            (x.shape[-1], self.out_features))
        
        # In JAX, we can use a soft-concrete distribution or a simple 
        # mask for sparsity during training.
        log_alpha = self.param('log_alpha', 
                              nn.initializers.constant(0.0), 
                              (x.shape[-1],))
        
        # Gate logic (simplified concrete distribution for JAX)
        gate = jax.nn.sigmoid(log_alpha) 
        masked_weights = weights * gate[:, None]
        
        return jnp.dot(x, masked_weights)

    def regularization(self):
        # L0 proxy: sum of the sigmoid gates
        log_alpha = self.get_variable('params', 'log_alpha')
        return jnp.sum(jax.nn.sigmoid(log_alpha))

class MARLPolynomialActor(nn.Module):
    """Global Actor using Polynomial Features in JAX."""
    max_action: float = 1.0

    @nn.compact
    def __call__(self, poly_features):
        # Change out_features to 8 (N_AGENTS)
        x = SparsePolynomialLayer(out_features=8)(poly_features)
        return self.max_action * jnp.tanh(x)

class CentralizedMARLCritic(nn.Module):
    """Centralized Critic taking Global State + Joint Actions."""
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_state, joint_actions):
        # global_state: (B, 130), joint_actions: (B, 8)
        xu = jnp.concatenate([global_state, joint_actions], axis=-1)
        
        def q_net(name):
            y = nn.Dense(self.hidden_dim, name=f"{name}_1")(xu)
            y = nn.relu(y)
            y = nn.Dense(self.hidden_dim, name=f"{name}_2")(y)
            y = nn.relu(y)
            return nn.Dense(1, name=f"{name}_3")(y)

        return q_net("q1"), q_net("q2")