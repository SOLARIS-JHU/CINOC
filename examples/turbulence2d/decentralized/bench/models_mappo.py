import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class CNNFeatureExtractor(nn.Module):
    """Matches the branch_net from DecentralizedTurbulenceNet"""
    features: Sequence[int] = (16, 32)
    
    @nn.compact
    def __call__(self, x):
        # x expected shape: (..., 20, 20, 3)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)

        # Flatten spatial dimensions
        x = x.reshape((*x.shape[:-3], -1))
        
        # Soft Normalization
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0) 
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x

class MAPPOActorTurb(nn.Module):
    n_agents: int
    u_max: float = 75.0
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, patches, pos_enc):
        """
        patches: (..., N_AGENTS, 20, 20, 3) 
        pos_enc: (..., N_AGENTS, pe_dim)
        """
        branch_out = CNNFeatureExtractor()(patches)
        combined = jnp.concatenate([branch_out, pos_enc], axis=-1)
        
        h = nn.Dense(self.hidden_dim)(combined)
        h = nn.tanh(h)
        mean_raw = nn.Dense(1)(h)
        
        # Bound the mean to the control limit
        mean = jnp.tanh(mean_raw) * self.u_max
        
        # Initialize std dev to be tight around the bounded mean
        log_std = self.param('log_std', lambda rng, shape: jnp.full(shape, -0.5), (self.n_agents, 1))
        
        batch_shape = mean.shape[:-2]
        log_std_b = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents, 1))
        
        return mean, log_std_b

class MAPPOCriticTurb(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_w):
        """
        global_w: (..., N_GRID, N_GRID) -> Centralized state for CTDE
        """
        # Add channel dimension for CNN
        x = jnp.expand_dims(global_w, axis=-1)
        
        # Lightweight CNN to encode the 64x64 global grid
        x = nn.Conv(16, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        x = x.reshape((*x.shape[:-3], -1))
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        v = nn.Dense(self.n_agents)(x) 
        return v