import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0

class MAPPOActor2DKS(nn.Module):
    """Decentralized Actor for MAPPO in 2D. Outputs 1D action distribution parameters per agent."""
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs):
        # obs shape: (batch, n_agents, obs_dim)
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Mean head for u (1 control variable per agent)
        mean_raw = nn.Dense(1)(x)
        
        # Bounded and scaled to [-U_MAX, U_MAX] for KS2D control
        mean = jnp.tanh(mean_raw) * U_MAX 
        
        # State-independent learned standard deviation
        log_std = self.param('log_std', nn.initializers.zeros, (self.n_agents, 1))
        
        # Broadcast log_std to match the batch dimensions of mean
        batch_shape = mean.shape[:-2]
        log_std_b = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents, 1))
        
        return mean, log_std_b

class MAPPOCritic2DKS(nn.Module):
    """Centralized State-Value Critic for 2D MAPPO. Outputs V(s) per agent based on global state."""
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, global_u):
        # Flatten the 2D global state: (batch, N_GRID, N_GRID) -> (batch, N_GRID * N_GRID)
        x = global_u.reshape((*global_u.shape[:-2], -1))
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output one value per agent based on the global state
        v = nn.Dense(self.n_agents)(x) 
        return v