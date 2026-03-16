import jax.numpy as jnp
import flax.linen as nn

class MARLActor2DKS(nn.Module):
    hidden_dim: int = 256
    u_max: float = 75.0  # Configurable max action

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # DPC-style Soft Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        u_raw = nn.Dense(1)(x)
        # Actor outputs actions natively scaled to [-u_max, u_max]
        u_out = self.u_max * jnp.tanh(u_raw)
        return u_out

class MARLCritic2DKS(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x, u):
        if u.ndim == x.ndim - 1:
            u = jnp.expand_dims(u, axis=-1)

        xu = jnp.concatenate([x, u], axis=-1)
        
        # Q1 Architecture
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2 Architecture
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2