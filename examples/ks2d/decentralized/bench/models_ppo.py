import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0

class PPOActor2DKS(nn.Module):
    n_agents: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u):
        # Flatten the 2D spatial grid (batch, N_GRID, N_GRID) -> (batch, N_GRID * N_GRID)
        x = u.reshape((*u.shape[:-2], -1)) 
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Mean bounded and scaled to [-U_MAX, U_MAX] for KS2D control
        mean_raw = nn.Dense(self.n_agents)(x)
        mean = jnp.tanh(mean_raw) * U_MAX
        
        # State-independent learned standard deviation
        log_std = self.param('log_std', nn.initializers.zeros, (self.n_agents,))
        
        # Broadcast to match batch size
        batch_shape = mean.shape[:-1]
        log_std = jnp.broadcast_to(log_std, (*batch_shape, self.n_agents))
        
        return mean, log_std

class PPOCritic2DKS(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, u):
        # Flatten the 2D spatial grid
        x = u.reshape((*u.shape[:-2], -1))
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        v = nn.Dense(1)(x)
        return v