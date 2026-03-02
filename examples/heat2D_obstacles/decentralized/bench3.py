import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import flax.linen as nn
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
from pathlib import Path
from functools import partial

# --- Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_heat2d_obstacles")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# Heat 2D Imports
from examples.heat2D.decentralized.dynamics_dual import PDEDynamics 
from models.policy import DecentralizedHeat2DControlNet
from examples.heat2D.decentralized.data_utils import get_training_data
from examples.heat2D.decentralized.bench.env_heat2d import extract_patches_heat2d_jit
from examples.heat2D.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.heat2D.decentralized.bench.models_marl import MARLActor2D, U_MAX, V_MAX

# 2D Specific Configuration
N_grid = 32
L_domain = 1.0
n_agents = 16
T_steps = 100 
N_eval = 50
ENV_MU = jnp.array([0.01]) 

# --- NEW: Obstacle Definitions for Visualization ---
OBSTACLES = np.array([
    [0.30, 0.30, 0.06],   # [x_center, y_center, radius]
    [0.50, 0.50, 0.06],   
    [0.70, 0.70, 0.06],   
])

def get_2d_sinusoidal_encoding(p_2d, d=1024, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

# --- 1. Baseline Model Definitions (Flax) ---

class CentralizedActor(nn.Module):
    hidden_dim: int = 256
    n_agents: int = 16 
    
    @nn.compact
    def __call__(self, obs_flat):
        x = nn.Dense(self.hidden_dim)(obs_flat)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        u = U_MAX * jnp.tanh(nn.Dense(self.n_agents)(x))
        vx = V_MAX * jnp.tanh(nn.Dense(self.n_agents)(x))
        vy = V_MAX * jnp.tanh(nn.Dense(self.n_agents)(x))
        return jnp.stack([u, vx, vy], axis=-1)

class SparsePolynomialLayer(nn.Module):
    out_features: int = 1
    gamma: float = -0.1
    zeta: float = 1.1
    @nn.compact
    def __call__(self, x):
        weights = self.param('weights', nn.initializers.glorot_uniform(), (x.shape[-1], self.out_features))
        log_alpha = self.param('log_alpha', nn.initializers.constant(0.0), (x.shape[-1], self.out_features))
        s = jax.nn.sigmoid(log_alpha)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate = jnp.clip(s_stretched, 0.0, 1.0) 
        return jnp.dot(x, weights * gate)

class MARLPolynomialActor(nn.Module):
    n_agents: int = 16
    @nn.compact
    def __call__(self, poly_features):
        x = SparsePolynomialLayer(out_features=self.n_agents * 3, name="sparse_layer")(poly_features)
        x = x.reshape(-1, self.n_agents, 3)
        u = U_MAX * jnp.tanh(x[..., 0])
        v = V_MAX * jnp.tanh(x[..., 1:3])
        return jnp.concatenate([u[..., None], v], axis=-1)

@jax.jit
def get_poly_features_jax(x):
    is_1d = x.ndim == 1
    x_2d = jnp.atleast_2d(x)
    n_feat = x_2d.shape[-1]
    _r, _c = np.triu_indices(n_feat)
    def poly_single(feat):
        bias = jnp.ones((1,))
        quad = jnp.outer(feat, feat)[_r, _c] 
        return jnp.concatenate([bias, feat, quad])
    res = jax.vmap(poly_single)(x_2d)
    return res[0] if is_1d else res

bench_registry = {}

# --- 2. Loading Logic ---
def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found. Skipping.")
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
n_side = int(np.sqrt(n_agents))
pos_1d = np.linspace(0.2, 0.8, n_side)
X, Y = np.meshgrid(pos_1d, pos_1d)
xi_init = jnp.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)

# 1. DPC (Loaded with _obstacles suffix)
dpc_model = DecentralizedHeat2DControlNet(features=(16, 32))
dpc_p = load_params('decentralized_params_heat2d_obstacles.msgpack', dpc_model, (jnp.zeros((N_grid, N_grid)), jnp.zeros((N_grid, N_grid)), xi_init))
if dpc_p:
    bench_registry['DPC'] = {'apply': dpc_model.apply, 'params': dpc_p, 'color': 'blue'}

# 2. MARL
marl_model = MARLActor2D()
marl_dummy_input = jnp.zeros((n_agents, 2349))
marl_p = load_params(bench_models_dir / 'marl_heat2d_params.msgpack', marl_model, (marl_dummy_input,))
if marl_p:
    def marl_apply(p, z, target, xi):
        y = extract_patches_heat2d_jit(z, target, xi/L_domain, window_size=6, resized_dim=10)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        pe = get_2d_sinusoidal_encoding(xi/L_domain, d=1024) 
        obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        action = marl_model.apply(p, obs)
        return action[:, 0], action[:, 1:3]
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 3. RL Centralized (Loaded with _obstacles suffix)
rl_model = CentralizedActor(n_agents=n_agents)
rl_input_dim = N_grid * N_grid * 2 + n_agents * 2
rl_p = load_params(bench_models_dir / 'rl_heat2d_params.msgpack', rl_model, (jnp.zeros(rl_input_dim),))
if rl_p:
    def rl_apply(p, z, target, xi):
        obs = jnp.concatenate([z.flatten(), target.flatten(), xi.flatten()])
        action = rl_model.apply(p, obs)
        return action[:, 0], action[:, 1:3]
    bench_registry['RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'purple'}

# 4. Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, z, t, xi: (jnp.zeros(n_agents), jnp.zeros((n_agents, 2))), 
    'params': None, 'color': 'red'
}

# --- 3. Simulation ---
print(f"Loading 2D dataset and Running Simulations for {list(bench_registry.keys())}...")
z_init_all, z_target_all, _ = get_training_data(n_samples=N_eval, n_grid=N_grid, dataset_dir='../../heat2D/data')
z_init_batch = jnp.array(z_init_all[:N_eval])
z_target_batch = jnp.array(z_target_all[:N_eval])
xi_batch = jnp.tile(xi_init, (N_eval, 1, 1))

@jax.jit(static_argnames=['name'])
def run_sim(name, z_i, target_i, xi_i):
    dyn = PDEDynamics(policy_apply_fn=bench_registry[name]['apply'])
    z_traj, xi_traj, u_traj, v_traj = dyn.unroll_controlled(
        z_init=z_i, xi_init=xi_i, z_target=target_i, 
        params=bench_registry[name]['params'], t_steps=T_steps
    )
    return z_traj, xi_traj

for name in bench_registry:
    print(f"Running {name} unrolls...")
    z_res, xi_res = jax.vmap(lambda z, t, x: run_sim(name, z, t, x))(z_init_batch, z_target_batch, xi_batch)
    bench_registry[name]['z_data'] = z_res
    bench_registry[name]['xi_data'] = xi_res

# --- 4. Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Mean Track Error':<20} | {'2-Sigma':<20}")
print("-" * 70)

for name in bench_registry:
    final_err = jnp.mean((bench_registry[name]['z_data'][:, -1] - z_target_batch)**2, axis=(1, 2))
    mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
    print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
print("="*70)

# --- 5. Individual Field Plots (With Obstacle Overlays) ---
print("Saving individual field plots to PDF...")
for name in bench_registry:
    fig = plt.figure(figsize=(15, 5))
    
    final_state = bench_registry[name]['z_data'][0, -1]
    target_state = z_target_batch[0]
    initial_state = z_init_batch[0]
    
    vmin = float(jnp.min(z_target_batch))
    vmax = float(jnp.max(z_target_batch))
    
    # Helper to plot obstacles
    def add_obstacles(ax):
        for obs in OBSTACLES:
            circle = patches.Circle((obs[0], obs[1]), obs[2], edgecolor='cyan', facecolor='none', linewidth=2, linestyle='--')
            ax.add_patch(circle)

    # Plot Initial
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(initial_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='hot', vmin=vmin, vmax=vmax)
    plt.title('Initial State')
    plt.colorbar(im1, label='Temperature')
    add_obstacles(ax1)
    
    # Plot Target
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(target_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='hot', vmin=vmin, vmax=vmax)
    plt.title('Target State')
    plt.colorbar(im2, label='Temperature')
    add_obstacles(ax2)

    # Plot Final
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(final_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='hot', vmin=vmin, vmax=vmax)
    plt.title(f'Final Controlled State: {name}')
    plt.colorbar(im3, label='Temperature')
    add_obstacles(ax3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"field_{name.lower()}.pdf")
    plt.close()

# --- 6. Plotting Trendlines ---
plt.figure(figsize=(18, 8))

# 1. Boxplot of Tracking Error
plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['z_data'][:, -1] - z_target_batch)**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final Tracking Error (MSE)')
plt.ylabel('Mean Squared Error')
plt.grid(True, alpha=0.3)

# 2. Error Evolution
plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps)
for name in bench_registry:
    evol = jnp.mean(jnp.mean((bench_registry[name]['z_data'] - z_target_batch[:, None, :, :])**2, axis=(2, 3)), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Tracking Error Evolution')
plt.xlabel('Time Step')
plt.ylabel('MSE (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "heat2d_obstacles_tracking_results.png")
print(f"\nSummary results saved to {output_dir}/heat2d_obstacles_tracking_results.png")