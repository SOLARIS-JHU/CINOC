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
from functools import partial

# --- Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_ks2d")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# KS2D Imports
from examples.ks2d.decentralized.dynamics_dual import PDEDynamics2D 
from models.policy_ks2d import DecentralizedKS2DControlNet
from examples.ks2d.decentralized.data_utils import get_batch_initial_conditions
from examples.ks2d.decentralized.bench.env_ks2d import extract_patches_2d_jit
from examples.ks2d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.ks2d.decentralized.bench.models_marl import MARLActor2DKS
from examples.ks2d.decentralized.bench.models_rl import CentralizedActorKS2D

# 2D Specific Configuration
N_grid = 64
L_domain = 32.0
n_agents = 100
T_steps = 50 
substeps = 20
dt = 0.005
N_eval = 20 # Reduced evaluation batch size to save VRAM on 2D
ENV_MU = jnp.array([L_domain, dt]) 

def get_2d_sinusoidal_encoding(p_2d, d=1024, n=1000.0):
    pe_x = get_sinusoidal_encoding(p_2d[:, 0], d=d, n=n)
    pe_y = get_sinusoidal_encoding(p_2d[:, 1], d=d, n=n)
    return jnp.concatenate([pe_x, pe_y], axis=-1)

bench_registry = {}

# --- 1. Loading Logic ---
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
# Initialize 10x10 grid of agents (100 agents)
grid_dim = int(np.sqrt(n_agents))
x_lin = np.linspace(0, L_domain, grid_dim, endpoint=False) + (L_domain/grid_dim)/2
xv, yv = np.meshgrid(x_lin, x_lin)
xi_init = jnp.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)

# 1. DPC (Centralized)
dpc_model = DecentralizedKS2DControlNet(features=(64, 128), domain_size=(L_domain, L_domain), u_max=10.0)
dpc_p = load_params('ks2d_centralized_params.msgpack', dpc_model, (jnp.zeros((N_grid, N_grid)), jnp.zeros((N_grid, N_grid)), xi_init))
if dpc_p:
    bench_registry['DPC'] = {'apply': dpc_model.apply, 'params': dpc_p, 'color': 'blue'}

# 2. MARL (Decentralized Multi-Agent)
marl_model = MARLActor2DKS()
# Dummy Input calculation: Patch 3 channels * 12x12 (432) + Mu (2) + PE (2048) = 2482
marl_dummy_input = jnp.zeros((n_agents, 2482))
marl_p = load_params(bench_models_dir / 'marl_ks2d_params.msgpack', marl_model, (marl_dummy_input,))

if marl_p:
    def marl_apply(p, u_curr, u_target, xi_fixed):
        y = extract_patches_2d_jit(u_curr, u_target, xi_fixed/L_domain, patch_size=12, n_grid=N_grid)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        pe = get_2d_sinusoidal_encoding(xi_fixed/L_domain, d=1024) 
        obs = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        action = marl_model.apply(p, obs)
        return action[..., 0] 
    
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 3. RL (Centralized God-View)
rl_model = CentralizedActorKS2D()
rl_dummy_input = jnp.zeros((1, N_grid, N_grid)) 
rl_p = load_params(bench_models_dir / 'rl_ks2d_params.msgpack', rl_model, (rl_dummy_input,))

if rl_p:
    def rl_apply(p, u_curr, u_target, xi_fixed):
        action = rl_model.apply(p, u_curr[None, ...])
        return action.squeeze() 
    
    bench_registry['RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'green'}


# 4. Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, u_curr, u_target, xi_fixed: jnp.zeros(n_agents), 
    'params': None, 'color': 'red'
}

# --- 2. Simulation ---
print(f"Generating KS2D Initial Conditions and Running Simulations for {list(bench_registry.keys())}...")
key = jax.random.PRNGKey(123)
u_init_batch = get_batch_initial_conditions(jax.random.split(key)[1], N_eval, N_grid, L_domain)
xi_batch = jnp.tile(xi_init, (N_eval, 1, 1))
u_target_batch = jnp.zeros_like(u_init_batch) 

@jax.jit(static_argnames=['name'])
def run_sim(name, u_i, xi_i, target_i):
    dyn = PDEDynamics2D(policy_apply_fn=bench_registry[name]['apply'])
    traj = dyn.unroll_controlled(
        u_init=u_i, 
        xi_fixed=xi_i, 
        u_target=target_i, 
        params=bench_registry[name]['params'], 
        t_steps=T_steps,
        substeps=substeps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        sigma=1.2
    )
    return traj[0] 

for name in bench_registry:
    print(f"Running {name} unrolls...")
    z_res = jax.vmap(lambda u, x, t: run_sim(name, u, x, t))(u_init_batch, xi_batch, u_target_batch)
    bench_registry[name]['data'] = z_res

# --- 3. Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Mean Energy':<20} | {'2-Sigma':<20}")
print("-" * 70)

for name in bench_registry:
    final_err = jnp.mean(bench_registry[name]['data'][:, -1]**2, axis=(1, 2))
    mean_val, std_val = jnp.mean(final_err), jnp.std(final_err)
    print(f"{name:<15} | {mean_val:.6f}             | ±{2*std_val:.6f}")
print("="*70)

# --- 4. Individual Field Plots (PDF Export) ---
print("Saving individual state plots to PDF...")
for name in bench_registry:
    fig = plt.figure(figsize=(10, 5))
    
    initial_state = u_init_batch[0]
    final_state = bench_registry[name]['data'][0, -1]
    
    vmin, vmax = float(jnp.min(initial_state)), float(jnp.max(initial_state))
    
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(initial_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title('Initial Chaotic State')
    plt.colorbar(im1, label='u(x,y)')
    
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(final_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f'Final Controlled State: {name}')
    plt.colorbar(im2, label='u(x,y)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"state_{name.lower()}.pdf")
    plt.close()

# --- 5. Plotting Trendlines ---
plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['data'][:, -1])**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final System Energy (Log Scale)')
plt.ylabel('Mean L2 Energy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps) * substeps * dt
for name in bench_registry:
    evol = jnp.mean(jnp.mean(bench_registry[name]['data']**2, axis=(2, 3)), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Stabilization Energy Evolution (Controlled Only)')
plt.xlabel('Time (s)')
plt.ylabel('Mean Energy (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "ks2d_stabilization_results.png")
print(f"\nSummary results saved to {output_dir}/ks2d_stabilization_results.png")