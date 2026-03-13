# import jax
# import jax.numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt
# import flax.linen as nn
# import flax.serialization
# from flax.serialization import msgpack_restore, from_state_dict
# import sys
# import os
# from pathlib import Path
# from typing import Sequence
# from functools import partial

# # --- 1. Model Definitions (Flax) ---

# # Action scaling constraints
# U_MAX = 40.0
# V_MAX = 2.0

# class MARLActor(nn.Module):
#     """
#     Standard Decentralized Actor adapted for Dual Outputs (from new training script).
#     Maps concatenated [y_i, mu, PE(p_i)] directly to action [u_i, v_i].
#     """
#     hidden_dim: int = 256

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(self.hidden_dim)(x)
#         x = nn.relu(x)
#         x = nn.Dense(self.hidden_dim)(x)
#         x = nn.relu(x)
        
#         # DPC-style Normalization trick for stability
#         x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        
#         # Dual Heads for Forcing (u) and Velocity (v)
#         u_raw = nn.Dense(1)(x)
#         v_raw = nn.Dense(1)(x)
        
#         u_out = U_MAX * jnp.tanh(u_raw)
#         v_out = V_MAX * jnp.tanh(v_raw)
        
#         # Shape: (..., 2)
#         return jnp.concatenate([u_out, v_out], axis=-1)

# # class HyperActor(nn.Module):
# #     """
# #     Hypernetwork-based Actor for FKPP (Unbatched/1D).
# #     Maps z = PE(xi_i) to the parameters of a local policy network.
# #     The local policy maps local state y_i to a 2D action [u_i, v_i].
# #     """
# #     hidden_dim: int = 256
# #     action_dim: int = 2
    
# #     @nn.compact
# #     def __call__(self, z, y):
# #         # Assumes inputs are strictly 1D vectors for a single agent/env combination
# #         y_dim = y.shape[-1]
# #         w1_size = y_dim * self.hidden_dim
# #         b1_size = self.hidden_dim
# #         w2_size = self.hidden_dim * self.action_dim
# #         b2_size = self.action_dim
# #         total_params = w1_size + b1_size + w2_size + b2_size
        
# #         # Hypernetwork forward pass (predicts primary network weights)
# #         h_out = nn.Dense(total_params, kernel_init=nn.initializers.xavier_uniform())(z)
        
# #         # Unpack weights for the single instance (NO batch dimension slicing)
# #         idx = 0
# #         w1 = h_out[idx : idx+w1_size].reshape(y_dim, self.hidden_dim)
# #         idx += w1_size
# #         b1 = h_out[idx : idx+b1_size]
# #         idx += b1_size
# #         w2 = h_out[idx : idx+w2_size].reshape(self.hidden_dim, self.action_dim)
# #         idx += w2_size
# #         b2 = h_out[idx : idx+b2_size]
        
# #         # DPC Normalization trick for stable gradients
# #         y_norm = y / (jnp.linalg.norm(y) + 1.0)
        
# #         # Primary local network forward pass 
# #         hidden = nn.relu(jnp.matmul(y_norm, w1) + b1)
# #         out = jnp.matmul(hidden, w2) + b2 
        
# #         # Bounded outputs based on actuator physical limits
# #         u_out = U_MAX * jnp.tanh(out[0:1])
# #         v_out = V_MAX * jnp.tanh(out[1:2])
        
# #         return jnp.concatenate([u_out, v_out], axis=-1)

# class CentralizedActor(nn.Module):
#     """Centralized baseline that observes the whole domain and outputs all actions."""
#     hidden_dim: int = 256
#     n_agents: int = 20
    
#     @nn.compact
#     def __call__(self, obs_flat):
#         x = nn.Dense(self.hidden_dim)(obs_flat)
#         x = nn.relu(x)
#         x = nn.Dense(self.hidden_dim)(x)
#         x = nn.relu(x)
#         u = U_MAX * jnp.tanh(nn.Dense(self.n_agents)(x))
#         v = V_MAX * jnp.tanh(nn.Dense(self.n_agents)(x))
#         return jnp.stack([u, v], axis=-1)

# # --- 2. Configuration & Paths ---
# script_dir = Path(__file__).resolve().parent.parent.parent.parent
# sys.path.append(str(script_dir))

# output_dir = Path("figures/bench_fkpp")
# output_dir.mkdir(parents=True, exist_ok=True)

# bench_models_dir = Path("bench/models")
# bench_models_dir.mkdir(parents=True, exist_ok=True) 

# # FKPP Imports
# from examples.fkpp1d.decentralized.dynamics_dual import PDEDynamics 
# from models.policy import DecentralizedControlNet
# from data_utils import generate_grf

# def get_sinusoidal_encoding(p, d=2048, n=1000.0):
#     """
#     Computes the sinusoidal positional encoding for the agents' relative positions.
    
#     Args:
#         p: Array of agent positions of shape (N_agents,).
#         d: Dimension of the positional embedding vector.
#         n: Constant scaling value.
        
#     Returns:
#         pe: Positional encoding matrix of shape (N_agents, d).
#     """
#     # Create an array of j values: 1 to d/2
#     j_vals = jnp.arange(1, (d // 2) + 1)
    
#     # Calculate omega_j = n^(2j/d)
#     omega_j = jnp.power(n, 2 * j_vals / d)
    
#     # Expand dimensions for broadcasting: (N_agents, 1) and (1, d/2)
#     p_expanded = p[:, None]
#     omega_expanded = omega_j[None, :]
    
#     # Calculate arguments for sin and cos
#     args = p_expanded / omega_expanded
    
#     # Calculate sin and cos components
#     sin_enc = jnp.sin(args)
#     cos_enc = jnp.cos(args)
    
#     # Interleave sin and cos: [sin1, cos1, sin2, cos2, ...]
#     pe = jnp.stack([sin_enc, cos_enc], axis=-1).reshape(p.shape[0], d)
#     return pe

# # FKPP specific configuration
# N_grid, L_domain, n_agents = 100, 1.0, 20
# T_steps, N_eval = 300, 50

# # PDE params to inject into local obs
# ENV_MU = jnp.array([0.02, 2.0]) # Example Nu, Rho. Adjust if your env uses different defaults

# # --- Shared Patch Extractor ---
# @partial(jax.jit, static_argnames=['window_size'])
# def extract_patches_jit(full_state, target_state, xi_norm, window_size):
#     """Updated to use constant padding (0.0) as per new training script logic."""
#     error = full_state - target_state
#     error_grad = jnp.gradient(error)
#     half_window = window_size // 2
    
#     padded_error = jnp.pad(error, (half_window, half_window), mode='constant', constant_values=0.0)
#     padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant', constant_values=0.0)
    
#     def get_local_obs(xi_n):
#         center_idx = jax.lax.stop_gradient((xi_n * (N_grid - 1)).astype(int)) + half_window
#         start = center_idx - half_window
#         p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
#         p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
#         p_err = jax.image.resize(p_err, (20,), method='bilinear')
#         p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
#         return jnp.concatenate([p_err, p_grad])
        
#     return jax.vmap(get_local_obs)(xi_norm)

# bench_registry = {}

# # --- 3. Loading Logic ---
# def load_params(filename, model, dummy_input):
#     if not os.path.exists(filename):
#         print(f"[-] {filename} not found.")
#         return None
#     with open(filename, 'rb') as f: bytes_data = f.read()
#     variables = model.init(jax.random.PRNGKey(0), *dummy_input)
#     try:
#         state_dict = msgpack_restore(bytes_data)
#         if 'actor' in state_dict: state_dict = state_dict['actor']
#         if 'params' in variables and 'params' not in state_dict: state_dict = {'params': state_dict}
#         elif 'params' not in variables and 'params' in state_dict: state_dict = state_dict['params']
#         return from_state_dict(variables, state_dict)
#     except Exception as e:
#         print(f"[-] Failed to load {filename}: {e}")
#         return None

# print("Loading Models...")
# xi_init = jnp.linspace(0.2, 0.8, n_agents)

# # 1. DPC
# dpc_model = DecentralizedControlNet(features=(64, 64))
# dpc_p = load_params('decentralized_params.msgpack', dpc_model, (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_init))
# if dpc_p:
#     bench_registry['DPC'] = {'apply': dpc_model.apply, 'params': dpc_p, 'color': 'blue'}

# # 2. MARL (Updated to new architecture)
# marl_model = MARLActor()
# # Dummy Input calculation: Patch (20+20=40) + Mu (2) + PE (128) = 170
# marl_dummy_input = jnp.zeros((n_agents, 170))
# marl_p = load_params(bench_models_dir / 'marl_fkpp_params.msgpack', marl_model, (marl_dummy_input,))

# if marl_p:
#     def marl_apply(p, z, target, xi):
#         y = extract_patches_jit(z, target, xi/L_domain, window_size=8)
#         mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
#         pe = get_sinusoidal_encoding(xi, d=128)
        
#         # Concat exact same way as training script build_marl_obs_batch + get_batch_actions
#         obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
#         action = marl_model.apply(p, obs)
#         return action[:, 0], action[:, 1]
    
#     bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# # 2.5 HypeMARL
# # hypemarl_model = HyperActor()
# # # Unbatched 1D dummy inputs for HyperActor initialization
# # dummy_z = jnp.zeros((2048,))
# # dummy_y = jnp.zeros((40,))

# # hypemarl_p = load_params(bench_models_dir / 'hypemarl_fkpp_params.msgpack', hypemarl_model, (dummy_z, dummy_y))

# # if hypemarl_p:
# #     def hypemarl_apply(p, z, target, xi):
# #         y = extract_patches_jit(z, target, xi/L_domain, window_size=8)
# #         pe = get_sinusoidal_encoding(xi, d=2048)
        
# #         # Vectorize the unbatched module over the n_agents dimension
# #         vmap_actor = jax.vmap(hypemarl_model.apply, in_axes=(None, 0, 0))
# #         action = vmap_actor(p, pe, y)
        
# #         return action[:, 0], action[:, 1]
    
# #     bench_registry['HypeMARL'] = {'apply': hypemarl_apply, 'params': hypemarl_p, 'color': 'green'}

# # 3. RL Centralized
# rl_model = CentralizedActor(n_agents=n_agents)
# rl_p = load_params(bench_models_dir / 'rl_fkpp_params.msgpack', rl_model, (jnp.zeros(N_grid*2 + n_agents),))
# if rl_p:
#     def rl_apply(p, z, target, xi):
#         obs = jnp.concatenate([z, target, xi])
#         action = rl_model.apply(p, obs)
#         return action[:, 0], action[:, 1]
#     bench_registry['RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'purple'}

# # 4. Uncontrolled Baseline
# bench_registry['Uncontrolled'] = {
#     'apply': lambda p, z, t, xi: (jnp.zeros(n_agents), jnp.zeros(n_agents)), 
#     'params': None, 'color': 'red'
# }

# # --- 4. Simulation ---
# print(f"Running Simulations for {list(bench_registry.keys())}...")
# key = jax.random.PRNGKey(42)
# keys_init = jax.random.split(key, N_eval)
# keys_target = jax.random.split(jax.random.PRNGKey(100), N_eval)

# _, z_init_batch = jax.vmap(partial(generate_grf, n_points=N_grid, length_scale=0.2))(keys_init)
# _, z_target_batch = jax.vmap(partial(generate_grf, n_points=N_grid, length_scale=0.4))(keys_target)
# xi_batch = jnp.tile(xi_init, (N_eval, 1))

# @jax.jit(static_argnames=['name'])
# def run_sim(name, z_i, target_i, xi_i):
#     dyn = PDEDynamics(policy_apply_fn=bench_registry[name]['apply'])
#     z_traj, xi_traj, u_traj, v_traj = dyn.unroll_controlled(
#         z_init=z_i, xi_init=xi_i, z_target=target_i, 
#         params=bench_registry[name]['params'], t_steps=T_steps
#     )
#     return z_traj, xi_traj

# for name in bench_registry:
#     print(f"Running {name} unrolls...")
#     z_res, xi_res = jax.vmap(lambda z, t, x: run_sim(name, z, t, x))(z_init_batch, z_target_batch, xi_batch)
#     bench_registry[name]['z_data'] = z_res
#     bench_registry[name]['xi_data'] = xi_res

# # --- 5. Metrics & Results Printing ---
# print("\n" + "="*70)
# print(f"{'Method':<15} | {'Mean Track Error':<20} | {'2-Sigma':<20}")
# print("-" * 70)

# for name in bench_registry:
#     # Error is MSE between final state and target
#     final_err = jnp.mean((bench_registry[name]['z_data'][:, -1] - z_target_batch)**2, axis=1)
#     mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
#     print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
# print("="*70)

# # --- 6. Plotting ---
# plt.figure(figsize=(18, 8))

# # 1. Boxplot of Tracking Error
# plt.subplot(1, 2, 1)
# data_boxplot = [jnp.mean((bench_registry[n]['z_data'][:, -1] - z_target_batch)**2, axis=1) for n in bench_registry]
# plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
# plt.yscale('log')
# plt.title('Final Tracking Error (MSE)')
# plt.ylabel('Mean Squared Error')
# plt.grid(True, alpha=0.3)

# # 2. Error Evolution
# plt.subplot(1, 2, 2)
# time_axis = jnp.arange(T_steps)
# for name in bench_registry:
#     # MSE at each timestep across the batch
#     evol = jnp.mean(jnp.mean((bench_registry[name]['z_data'] - z_target_batch[:, None, :])**2, axis=2), axis=0)
#     plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

# plt.yscale('log')
# plt.title('Tracking Error Evolution')
# plt.xlabel('Time Step')
# plt.ylabel('MSE (Log)')
# plt.legend()
# plt.grid(True, which="both", alpha=0.3)

# plt.tight_layout()
# plt.savefig(output_dir / "fkpp_tracking_results.png")
# print(f"\nSummary results saved to {output_dir}/fkpp_tracking_results.png")

# # --- 7. Individual Field Plots (PDF Export) ---
# print("Saving individual field plots to PDF...")
# for name in bench_registry:
#     plt.figure(figsize=(8, 5))
    
#     # Visualize the first sample [0] from the evaluation batch
#     field_data = bench_registry[name]['z_data'][0]
    
#     plt.imshow(field_data.T, aspect='auto', origin='lower', 
#                extent=[0, T_steps, 0, L_domain], 
#                cmap='magma', vmin=0, vmax=float(jnp.max(z_target_batch)*1.2))
    
#     plt.colorbar(label='Concentration u(x,t)')
#     plt.title(f'FKPP Controlled Field: {name}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Space (x)')
#     plt.tight_layout()
#     plt.savefig(output_dir / f"field_{name.lower()}.pdf")
#     plt.close()


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
from pathlib import Path
from typing import Sequence
from functools import partial

# --- 1. Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_fkpp")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# FKPP Imports
from examples.fkpp1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

# Import Models
from examples.fkpp1d.decentralized.bench.models_marl import MARLActor
from examples.fkpp1d.decentralized.bench.models_rl import CentralizedActor
from examples.fkpp1d.decentralized.bench.models_ppo import PPOActor
from examples.fkpp1d.decentralized.bench.models_mappo import MAPPOActor

# FKPP specific configuration
N_grid, L_domain, n_agents = 100, 1.0, 20
T_steps, N_eval = 300, 50

# PDE params to inject into local obs
ENV_MU = jnp.array([0.005, 3.0]) 

def get_sinusoidal_encoding(p, d=128, n=1000.0):
    j_vals = jnp.arange(1, (d // 2) + 1)
    omega_j = jnp.power(n, 2 * j_vals / d)
    p_expanded = p[:, None]
    omega_expanded = omega_j[None, :]
    args = p_expanded / omega_expanded
    sin_enc = jnp.sin(args)
    cos_enc = jnp.cos(args)
    pe = jnp.stack([sin_enc, cos_enc], axis=-1).reshape(p.shape[0], d)
    return pe

# --- Shared Patch Extractor ---
@partial(jax.jit, static_argnames=['window_size'])
def extract_patches_jit(full_state, target_state, xi_norm, window_size):
    """Uses constant padding (0.0) for zero boundary conditions."""
    error = full_state - target_state
    error_grad = jnp.gradient(error)
    half_window = window_size // 2
    
    padded_error = jnp.pad(error, (half_window, half_window), mode='constant', constant_values=0.0)
    padded_grad = jnp.pad(error_grad, (half_window, half_window), mode='constant', constant_values=0.0)
    
    def get_local_obs(xi_n):
        center_idx = jax.lax.stop_gradient((xi_n * (N_grid - 1)).astype(int)) + half_window
        start = center_idx - half_window
        p_err = jax.lax.dynamic_slice(padded_error, (start,), (window_size,))
        p_grad = jax.lax.dynamic_slice(padded_grad, (start,), (window_size,))
        p_err = jax.image.resize(p_err, (20,), method='bilinear')
        p_grad = jax.image.resize(p_grad, (20,), method='bilinear')
        return jnp.concatenate([p_err, p_grad])
        
    return jax.vmap(get_local_obs)(xi_norm)

bench_registry = {}

# --- 3. Loading Logic ---
def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found.")
        return None
    with open(filename, 'rb') as f: bytes_data = f.read()
    variables = model.init(jax.random.PRNGKey(0), *dummy_input)
    try:
        state_dict = msgpack_restore(bytes_data)
        if 'actor' in state_dict: state_dict = state_dict['actor']
        if 'params' in variables and 'params' not in state_dict: state_dict = {'params': state_dict}
        elif 'params' not in variables and 'params' in state_dict: state_dict = state_dict['params']
        return from_state_dict(variables, state_dict)
    except Exception as e:
        print(f"[-] Failed to load {filename}: {e}")
        return None

print("Loading Models...")
xi_init = jnp.linspace(0.2, 0.8, n_agents)

# 1. DPC
dpc_model = DecentralizedControlNet(features=(64, 64))
dpc_p = load_params('decentralized_params.msgpack', dpc_model, (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_init))
if dpc_p:
    bench_registry['DPC'] = {'apply': dpc_model.apply, 'params': dpc_p, 'color': 'blue'}

# 2. MARL (DDPG/TD3)
marl_model = MARLActor()
marl_dummy_input = jnp.zeros((n_agents, 170))
marl_p = load_params(bench_models_dir / 'marl_fkpp_params.msgpack', marl_model, (marl_dummy_input,))
if marl_p:
    def marl_apply(p, z, target, xi):
        y = extract_patches_jit(z, target, xi/L_domain, window_size=4)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        pe = get_sinusoidal_encoding(xi, d=128)
        obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        action = marl_model.apply(p, obs)
        return action[:, 0], action[:, 1]
    
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 3. RL (Centralized DDPG/TD3)
rl_model = CentralizedActor(n_agents=n_agents)
rl_dummy_z = jnp.zeros((1, N_grid))
rl_dummy_xi = jnp.zeros((1, n_agents))
rl_p = load_params(bench_models_dir / 'rl_fkpp_params.msgpack', rl_model, (rl_dummy_z, rl_dummy_z, rl_dummy_xi))
if rl_p:
    def rl_apply(p, z, target, xi):
        # Apply adds batch dim [None, ...] since model expects batched inputs
        action = rl_model.apply(p, z[None, ...], target[None, ...], xi[None, ...])
        return action[0, :, 0], action[0, :, 1]
    bench_registry['RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'purple'}

# 4. PPO (Centralized)
ppo_model = PPOActor(n_agents=n_agents)
ppo_p = load_params(bench_models_dir / 'ppo_fkpp_params.msgpack', ppo_model, (rl_dummy_z, rl_dummy_z, rl_dummy_xi))
if ppo_p:
    def ppo_apply(p, z, target, xi):
        mean, _ = ppo_model.apply(p, z[None, ...], target[None, ...], xi[None, ...])
        return mean[0, :, 0], mean[0, :, 1]
    bench_registry['PPO'] = {'apply': ppo_apply, 'params': ppo_p, 'color': 'green'}

# 5. MAPPO (Decentralized)
mappo_model = MAPPOActor(n_agents=n_agents)
mappo_dummy_input = jnp.zeros((1, n_agents, 170))
mappo_p = load_params(bench_models_dir / 'mappo_fkpp_params.msgpack', mappo_model, (mappo_dummy_input,))
if mappo_p:
    def mappo_apply(p, z, target, xi):
        y = extract_patches_jit(z, target, xi/L_domain, window_size=4)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        pe = get_sinusoidal_encoding(xi, d=128)
        obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        mean, _ = mappo_model.apply(p, obs[None, ...])
        return mean[0, :, 0], mean[0, :, 1]
    bench_registry['MAPPO'] = {'apply': mappo_apply, 'params': mappo_p, 'color': 'cyan'}

# 6. Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, z, t, xi: (jnp.zeros(n_agents), jnp.zeros(n_agents)), 
    'params': None, 'color': 'red'
}

# --- 4. Simulation ---
print(f"Running Simulations for {list(bench_registry.keys())}...")
key = jax.random.PRNGKey(42)
keys_init = jax.random.split(key, N_eval)
keys_target = jax.random.split(jax.random.PRNGKey(100), N_eval)

_, z_init_batch = jax.vmap(partial(generate_grf, n_points=N_grid, length_scale=0.2))(keys_init)
_, z_target_batch = jax.vmap(partial(generate_grf, n_points=N_grid, length_scale=0.4))(keys_target)
xi_batch = jnp.tile(xi_init, (N_eval, 1))

@jax.jit(static_argnames=['name'])
def run_sim(name, z_i, target_i, xi_i):
    dyn = PDEDynamics(policy_apply_fn=bench_registry[name]['apply'])
    z_traj, xi_traj, u_traj, v_traj = dyn.unroll_controlled(
        z_init=z_i, xi_init=xi_i, z_target=target_i, 
        params=bench_registry[name]['params'], t_steps=T_steps, nu=ENV_MU[0], rho=ENV_MU[1]
    )
    return z_traj, xi_traj

for name in bench_registry:
    print(f"Running {name} unrolls...")
    z_res, xi_res = jax.vmap(lambda z, t, x: run_sim(name, z, t, x))(z_init_batch, z_target_batch, xi_batch)
    bench_registry[name]['z_data'] = z_res
    bench_registry[name]['xi_data'] = xi_res

# --- 5. Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Mean Track Error':<20} | {'2-Sigma':<20}")
print("-" * 70)

for name in bench_registry:
    # Error is MSE between final state and target
    final_err = jnp.mean((bench_registry[name]['z_data'][:, -1] - z_target_batch)**2, axis=1)
    mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
    print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
print("="*70)

# --- 6. Plotting ---
plt.figure(figsize=(18, 8))

# 1. Boxplot of Tracking Error
plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['z_data'][:, -1] - z_target_batch)**2, axis=1) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final Tracking Error (MSE)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

# 2. Error Evolution
plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps)
for name in bench_registry:
    # MSE at each timestep across the batch
    evol = jnp.mean(jnp.mean((bench_registry[name]['z_data'] - z_target_batch[:, None, :])**2, axis=2), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Tracking Error Evolution')
plt.xlabel('Time Step')
plt.ylabel('MSE (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "fkpp_tracking_results.png")
print(f"\nSummary results saved to {output_dir}/fkpp_tracking_results.png")

# --- 7. Individual Field Plots (PDF Export) ---
print("Saving individual field plots to PDF...")
for name in bench_registry:
    plt.figure(figsize=(8, 5))
    
    # Visualize the first sample [0] from the evaluation batch
    field_data = bench_registry[name]['z_data'][0]
    
    plt.imshow(field_data.T, aspect='auto', origin='lower', 
               extent=[0, T_steps, 0, L_domain], 
               cmap='magma', vmin=0, vmax=float(jnp.max(z_target_batch)*1.2))
    
    plt.colorbar(label='Concentration u(x,t)')
    plt.title(f'FKPP Controlled Field: {name}')
    plt.xlabel('Time Step')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    plt.savefig(output_dir / f"field_{name.lower()}.pdf")
    plt.close()