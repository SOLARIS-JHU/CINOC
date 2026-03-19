import jax.numpy as jnp
import flax.linen as nn

class CentralizedActor(nn.Module):
    n_agents: int = 64

    @nn.compact
    def __call__(self, z):
        x = jnp.expand_dims(z, -1) 
        
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        
        x = x.reshape((*x.shape[:-3], -1)) 
        
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        
        out = nn.Dense(self.n_agents)(x)
        # ONLY return the normalized [-1.0, 1.0] percentage!
        return jnp.tanh(out) 

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
        
        z_features = x.reshape((*x.shape[:-3], -1))
        
        # --- NEW: Compress state features so they don't drown out actions ---
        z_features = nn.Dense(256)(z_features)
        z_features = nn.relu(z_features)
        
        # Now concatenate 256 state dims with 64 action dims (much more balanced)
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