import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0  

class CentralizedActor(nn.Module):
    n_agents: int = 64

    @nn.compact
    def __call__(self, z):
        # Add channel dimension for CNN: (..., N_grid, N_grid, 1)
        x = jnp.expand_dims(z, -1) 
        
        # Spatial feature extraction
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        
        # Dimension-agnostic flatten: Flattens Height x Width x Channels
        # Safely handles both batched (Batch, 4096) and unbatched (4096,) calls
        x = x.reshape((*x.shape[:-3], -1)) 
        
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        # Output directly for all agents
        out = nn.Dense(self.n_agents)(x)
        return jnp.tanh(out) * U_MAX

class CentralizedCritic(nn.Module):
    n_agents: int = 64

    @nn.compact
    def __call__(self, z, actions):
        x = jnp.expand_dims(z, -1)
        
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        
        # Dimension-agnostic flatten
        z_features = x.reshape((*x.shape[:-3], -1))
        
        # Concat spatial features with the continuous actions
        xu = jnp.concatenate([z_features, actions], axis=-1)
        
        # Q1
        q1 = nn.Dense(256)(xu)
        q1 = nn.relu(q1)
        q1 = nn.Dense(256)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        # Q2
        q2 = nn.Dense(256)(xu)
        q2 = nn.relu(q2)
        q2 = nn.Dense(256)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2