import jax
import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraint (Matches DecentralizedKS2DControlNet config)
U_MAX = 5.0  

class CentralizedActorKS2D(nn.Module):
    """
    Centralized Actor for 2D KS Stabilization (Fixed Actuators).
    Maps the full 2D state directly to all 100 1D actions [u].
    """
    hidden_dim: int = 256
    n_agents: int = 100

    @nn.compact
    def __call__(self, z):
        # Flatten the spatial grid
        z_flat = z.reshape((*z.shape[:-2], -1))
        
        x = nn.Dense(self.hidden_dim)(z_flat)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # DPC-style Soft Normalization trick for stability (+ 1.0)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Single Head for Forcing (u), outputting for all agents
        u_raw = nn.Dense(self.n_agents)(x)
        
        u_out = U_MAX * jnp.tanh(u_raw)
        
        # Shape: (..., n_agents, 1) -> match action dimensions
        return jnp.expand_dims(u_out, axis=-1)

class CentralizedCriticKS2D(nn.Module):
    """
    Centralized Critic network estimating a global Q-value.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z, actions):
        # Flatten the state
        z_flat = z.reshape((*z.shape[:-2], -1))
        
        # actions is shape (..., n_agents, 1). Flatten it.
        batch_shape = actions.shape[:-2]
        actions_flat = actions.reshape((*batch_shape, -1))

        xu = jnp.concatenate([z_flat, actions_flat], axis=-1)
        
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2