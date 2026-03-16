import jax
import jax.numpy as jnp
import flax.linen as nn

# Action scaling constraints (Matches NS2D centralized config)
V_MAX = 0.8  # Push max limit from the baseline

class CentralizedTD3Actor(nn.Module):
    """
    Single-agent Actor for NS2D Density Control.
    Maps the single flat observation to [vx, vy] commands.
    """
    hidden_dim: int = 256
    n_agents: int = 9 

    @nn.compact
    def __call__(self, obs):
        # Environment already flattened and concatenated the inputs
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Normalization trick for stability
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
        # Dual-Heads for Velocity (vx, vy) for ALL agents simultaneously
        vx_raw = nn.Dense(self.n_agents)(x)
        vy_raw = nn.Dense(self.n_agents)(x)
        
        vx_out = V_MAX * jnp.tanh(vx_raw)
        vy_out = V_MAX * jnp.tanh(vy_raw)
        
        return jnp.stack([vx_out, vy_out], axis=-1)

class CentralizedTD3Critic(nn.Module):
    """
    Single-agent Critic.
    Maps [joint_obs, joint_actions] to a single Q-value.
    """
    hidden_dim: int = 256
    n_agents: int = 9

    @nn.compact
    def __call__(self, obs, actions):
        # Both inputs are already flattened joint vectors in train_rl.py
        xu = jnp.concatenate([obs, actions], axis=-1)
        
        # Q1
        q1 = nn.Dense(self.hidden_dim)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2
        q2 = nn.Dense(self.hidden_dim)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2