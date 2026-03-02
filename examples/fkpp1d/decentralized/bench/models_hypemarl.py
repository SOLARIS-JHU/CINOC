import jax
import jax.numpy as jnp
import flax.linen as nn

U_MAX = 40.0
V_MAX = 2.0

class HyperActor(nn.Module):
    """
    Hypernetwork-based Actor for FKPP (Unbatched/1D).
    Maps z = PE(xi_i) to the parameters of a local policy network.
    The local policy maps local state y_i to a 2D action [u_i, v_i].
    """
    hidden_dim: int = 256
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, z, y):
        # Assumes inputs are strictly 1D vectors for a single agent/env combination
        y_dim = y.shape[-1]
        w1_size = y_dim * self.hidden_dim
        b1_size = self.hidden_dim
        w2_size = self.hidden_dim * self.action_dim
        b2_size = self.action_dim
        total_params = w1_size + b1_size + w2_size + b2_size
        
        # Hypernetwork forward pass (predicts primary network weights)
        h_out = nn.Dense(total_params, kernel_init=nn.initializers.xavier_uniform())(z)
        
        # Unpack weights for the single instance (NO batch dimension slicing)
        idx = 0
        w1 = h_out[idx : idx+w1_size].reshape(y_dim, self.hidden_dim)
        idx += w1_size
        b1 = h_out[idx : idx+b1_size]
        idx += b1_size
        w2 = h_out[idx : idx+w2_size].reshape(self.hidden_dim, self.action_dim)
        idx += w2_size
        b2 = h_out[idx : idx+b2_size]
        
        # DPC Normalization trick for stable gradients
        y_norm = y / (jnp.linalg.norm(y) + 1.0)
        
        # Primary local network forward pass 
        hidden = nn.relu(jnp.matmul(y_norm, w1) + b1)
        out = jnp.matmul(hidden, w2) + b2 
        
        # Bounded outputs based on actuator physical limits
        u_out = U_MAX * jnp.tanh(out[0:1])
        v_out = V_MAX * jnp.tanh(out[1:2])
        
        return jnp.concatenate([u_out, v_out], axis=-1)

class HyperCritic(nn.Module):
    """
    Hypernetwork-based Critic (Unbatched/1D).
    Maps z = PE(xi_i) to parameters of a local Q-network.
    """
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, z, y, u):
        yu = jnp.concatenate([y, u], axis=-1)
        yu_dim = yu.shape[-1]
        
        w1_size = yu_dim * self.hidden_dim
        b1_size = self.hidden_dim
        w2_size = self.hidden_dim * 1
        b2_size = 1
        total_params = w1_size + b1_size + w2_size + b2_size
        
        h_out = nn.Dense(total_params, kernel_init=nn.initializers.xavier_uniform())(z)
        
        idx = 0
        w1 = h_out[idx : idx+w1_size].reshape(yu_dim, self.hidden_dim)
        idx += w1_size
        b1 = h_out[idx : idx+b1_size]
        idx += b1_size
        w2 = h_out[idx : idx+w2_size].reshape(self.hidden_dim, 1)
        idx += w2_size
        b2 = h_out[idx : idx+b2_size]
        
        hidden = nn.relu(jnp.matmul(yu, w1) + b1)
        q_val = jnp.matmul(hidden, w2) + b2
        
        return q_val[0] # Return as scalar for the single instance

class SurrogateModel(nn.Module):
    """
    Shallow NN surrogate model for MB-HypeMARL on moving actuators (Unbatched/1D).
    """
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, y, u, xi):
        x = jnp.concatenate([y, u, xi], axis=-1)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        out = nn.Dense(y.shape[-1] + 1)(x)
        next_y = out[:-1]
        next_xi = out[-1:]
        
        return next_y, next_xi