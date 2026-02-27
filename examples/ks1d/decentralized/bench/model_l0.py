import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from functools import partial
import numpy as np

# ==========================================
# 1. NETWORK ARCHITECTURES
# ==========================================

class SparsePolynomialLayer(nn.Module):
    """
    A Flax layer that applies a sparse mask to polynomial features.
    This is the JAX equivalent of the Hard-Concrete L0Dense logic.
    """
    out_features: int = 1
    # Hyperparameters from Louizos et al. 2018
    gamma: float = -0.1
    zeta: float = 1.1
    
    @nn.compact
    def __call__(self, x):
        weights = self.param('weights', 
                            nn.initializers.glorot_uniform(), 
                            (x.shape[-1], self.out_features))
        
        # log_alpha now matches the shape of weights (independent mask per actuator)
        log_alpha = self.param('log_alpha', 
                              nn.initializers.constant(0.0), 
                              (x.shape[-1], self.out_features))
        
        # Hard-Concrete stretched sigmoid
        s = jax.nn.sigmoid(log_alpha)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        
        # Clip to allow exactly 0.0 and 1.0
        gate = jnp.clip(s_stretched, 0.0, 1.0) 
        
        masked_weights = weights * gate
        
        return jnp.dot(x, masked_weights)


class MARLPolynomialActor(nn.Module):
    """Global Actor using Polynomial Features in JAX."""
    max_action: float = 1.0

    @nn.compact
    def __call__(self, poly_features):
        # We give the layer an explicit name so we can easily find 'log_alpha' later
        x = SparsePolynomialLayer(out_features=8, name="sparse_layer")(poly_features)
        return self.max_action * jnp.tanh(x)


class StandardCritic(nn.Module):
    """Standard Critic taking Local Observation + Joint Actions."""
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs, joint_actions):
        # obs: (B, 10), joint_actions: (B, 8)
        xu = jnp.concatenate([obs, joint_actions], axis=-1)
        
        def q_net(name):
            y = nn.Dense(self.hidden_dim, name=f"{name}_1")(xu)
            y = nn.relu(y)
            y = nn.Dense(self.hidden_dim, name=f"{name}_2")(y)
            y = nn.relu(y)
            return nn.Dense(1, name=f"{name}_3")(y)

        return q_net("q1"), q_net("q2")

# ==========================================
# 2. TD3 TRAINING WRAPPER
# ==========================================

class TD3TrainState(train_state.TrainState):
    """Extension of TrainState to hold target network parameters."""
    target_params: dict


class JAXMARLTD3:
    def __init__(self, poly_feat_dim, obs_dim, joint_act_dim, 
                 max_action=1.0, tau=0.005, lr=3e-4, seed=0, device='cpu'):
        
        self.max_action = max_action
        self.tau = tau
        
        # Instantiate the models
        self.actor_model = MARLPolynomialActor(max_action=max_action)
        self.critic_model = StandardCritic(hidden_dim=256)

        # Initialize PRNG keys
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # Dummy inputs to initialize Flax parameter shapes
        dummy_poly = jnp.zeros((1, poly_feat_dim))
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_actions = jnp.zeros((1, joint_act_dim))

        # Initialize Actor State
        actor_params = self.actor_model.init(actor_key, dummy_poly)['params']
        self.actor_state = TD3TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            target_params=actor_params,
            tx=optax.adam(learning_rate=lr)
        )

        # Initialize Critic State
        critic_params = self.critic_model.init(critic_key, dummy_obs, dummy_actions)['params']
        self.critic_state = TD3TrainState.create(
            apply_fn=self.critic_model.apply,
            params=critic_params,
            target_params=critic_params,
            tx=optax.adam(learning_rate=lr)
        )

    # ---------------------------------------------------------
    # ACTION SELECTION
    # ---------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def select_action(self, params, poly_features):
        poly_features = jnp.atleast_2d(poly_features)
        return self.actor_model.apply({'params': params}, poly_features)[0]

    def get_action(self, poly_features):
        """Wrapper for interacting with the environment (returns numpy)."""
        return np.array(self.select_action(self.actor_state.params, poly_features))

    # ---------------------------------------------------------
    # CRITIC UPDATE
    # ---------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def update_critic(self, actor_state, critic_state, batch, rng, policy_noise, noise_clip, discount):
        obs, poly_features, action, next_obs, next_poly_features, reward, done = batch
        
        # Target action + Clipped Noise
        rng, noise_key = jax.random.split(rng)
        next_action = self.actor_model.apply({'params': actor_state.target_params}, next_poly_features)
        noise = jax.random.normal(noise_key, next_action.shape) * policy_noise
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        next_action = jnp.clip(next_action + noise, -self.max_action, self.max_action)

        # Target Q Values (Min of Double Q)
        target_q1, target_q2 = self.critic_model.apply(
            {'params': critic_state.target_params}, next_obs, next_action
        )
        target_q = jnp.minimum(target_q1, target_q2)
        target_q = reward + (1.0 - done) * discount * target_q

        # Critic Loss
        def critic_loss_fn(params):
            current_q1, current_q2 = self.critic_model.apply({'params': params}, obs, action)
            loss_q1 = jnp.mean((current_q1 - target_q) ** 2)
            loss_q2 = jnp.mean((current_q2 - target_q) ** 2)
            return loss_q1 + loss_q2

        loss, grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
        new_critic_state = critic_state.apply_gradients(grads=grads)
        
        return new_critic_state, loss, rng

    # ---------------------------------------------------------
    # ACTOR UPDATE (With Hard-Concrete L0 Regularization)
    # ---------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def update_actor(self, actor_state, critic_state, batch, lambda_l0=0.01):
        obs, poly_features, _, _, _, _, _ = batch

        def actor_loss_fn(params):
            # 1. Deterministic Policy Gradient Loss
            actions = self.actor_model.apply({'params': params}, poly_features)
            q1, _ = self.critic_model.apply({'params': critic_state.params}, obs, actions)
            dpg_loss = -jnp.mean(q1)

            # 2. L0 Regularization Loss (Hard-Concrete Proxy)
            gamma, zeta = -0.1, 1.1
            log_alpha = params['sparse_layer']['log_alpha']
            
            # This computes the continuous penalty pushing weights towards 0
            penalty = jax.nn.sigmoid(log_alpha - jnp.log(-gamma / zeta))
            l0_loss = jnp.sum(penalty)
            
            # Equation 14 from the paper: J(θ) = -Q + λ*L0
            return dpg_loss + (lambda_l0 * l0_loss)

        loss, grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        new_actor_state = actor_state.apply_gradients(grads=grads)

        # Soft updates for target networks
        def soft_update(params, target_params, tau):
            return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), params, target_params)

        new_actor_state = new_actor_state.replace(
            target_params=soft_update(new_actor_state.params, new_actor_state.target_params, self.tau)
        )
        new_critic_state = critic_state.replace(
            target_params=soft_update(critic_state.params, critic_state.target_params, self.tau)
        )

        return new_actor_state, new_critic_state, loss

    # ---------------------------------------------------------
    # MAIN TRAINING LOOP
    # ---------------------------------------------------------
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, 
              policy_noise=0.2, noise_clip=0.3, policy_freq=2, lambda_l0=0.01):
        
        rng = jax.random.PRNGKey(np.random.randint(0, 10000))

        for it in range(iterations):
            batch = replay_buffer.sample(batch_size)
            
            # Update Critic
            self.critic_state, critic_loss, rng = self.update_critic(
                self.actor_state, self.critic_state, batch, rng, policy_noise, noise_clip, discount
            )

            # Delayed Policy Update
            if it % policy_freq == 0:
                self.actor_state, self.critic_state, actor_loss = self.update_actor(
                    self.actor_state, self.critic_state, batch, lambda_l0
                )