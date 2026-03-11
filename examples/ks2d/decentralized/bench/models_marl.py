import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraint (Matches DecentralizedKS2DControlNet config)
U_MAX = 5.0  

class MARLActor2DKS(nn.Module):
    """
    Standard Decentralized Actor adapted for 2D KS (Fixed Actuators).
    Maps concatenated [y_local, mu, PE_2d(x, y)] directly to 1D action [u].
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # DPC-style Soft Normalization trick for stability (+ 1.0)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Single Head for Forcing (u), no velocity since agents are fixed
        u_raw = nn.Dense(1)(x)
        
        u_out = U_MAX * jnp.tanh(u_raw)
        
        # Shape: (..., 1) -> [u]
        return u_out

class MARLCritic2DKS(nn.Module):
    """
    Critic network estimating Q-values for the 1D action space.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, u):
        # Gracefully handle cases where 'u' might be passed as shape (...) instead of (..., 1)
        if u.ndim == x.ndim - 1:
            u = jnp.expand_dims(u, axis=-1)

        # u is now shape (..., 1)
        xu = jnp.concatenate([x, u], axis=-1)
        
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