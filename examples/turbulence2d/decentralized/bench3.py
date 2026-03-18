import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
import pickle
from pathlib import Path
from functools import partial

# Enable x64 for Spectral Solvers
jax.config.update("jax_enable_x64", True)

# --- Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/bench_turb")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

# Turbulence Imports
from examples.turbulence2d.decentralized.dynamics_dual import PDEDynamics2D 
from models.policy_turb import DecentralizedTurbulenceNet
from examples.turbulence2d.decentralized.data_utils import get_batch_initial_conditions

# MARL & RL Model Imports
from examples.turbulence2d.decentralized.bench.env_turb2d import extract_patches_2d_jit
from examples.turbulence2d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.turbulence2d.decentralized.bench.models_marl import MARLActor2DKS
from examples.turbulence2d.decentralized.bench.models_rl import CentralizedActor

# 2D Specific Configuration
N_grid = 64
L_domain = 1.0
n_agents = 64 # 8x8 Actuator Grid
T_steps = 150 
substeps = 5
dt = 0.01
viscosity = 5e-4
N_eval = 20 # Evaluation batch size
ENV_MU = jnp.array([L_domain, dt, viscosity])
STATE_NORM_FACTOR = 50.0 # From RL script

# Adapted: d defaults to 64 to match MARL training script
def get_2d_sinusoidal_encoding(p_2d, d=64, n=1000.0):
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
    
    # Init expects (xi_fixed, obs) for TurbulenceNet 
    # Or (dummy_input) for Actor models
    if isinstance(model, DecentralizedTurbulenceNet):
        variables = model.init(jax.random.PRNGKey(0), dummy_input[0], dummy_input[1])
    else:
        variables = model.init(jax.random.PRNGKey(0), dummy_input)
        
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
# Initialize 8x8 grid of agents
grid_dim = int(np.sqrt(n_agents))
x_lin = np.linspace(0, L_domain, grid_dim, endpoint=False) + (L_domain/grid_dim)/2
xv, yv = np.meshgrid(x_lin, x_lin)
xi_init = jnp.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)
target_state = jnp.zeros((N_grid, N_grid))

# 1. DPC (Centralized)
dpc_model = DecentralizedTurbulenceNet(features=(32, 64), patch_size=16, domain_size=(L_domain, L_domain), u_max=40.0)
dpc_p = load_params('turbulence_params.msgpack', dpc_model, (xi_init, jnp.zeros((1, N_grid, N_grid))))
if dpc_p:
    def dpc_apply_wrapped(p, xi_fixed, obs):
        return dpc_model.apply(p, xi_fixed, obs)
    bench_registry['DPC'] = {'apply': dpc_apply_wrapped, 'params': dpc_p, 'color': 'blue'}

# 2. MARL (Decentralized Multi-Agent)
marl_model = MARLActor2DKS()
# Adapted: Observation dim is 387 -> (16*16 patch) + (3 mu) + (64*2 PE)
marl_dummy_input = jnp.zeros((n_agents, 387))
marl_p = load_params(bench_models_dir / 'marl_turb_params.msgpack', marl_model, marl_dummy_input)

if marl_p:
    def marl_apply(p, xi_fixed, obs):
        w_phys = obs.squeeze()
        xi_norm = xi_fixed / L_domain
        
        y = extract_patches_2d_jit(w_phys, target_state, xi_norm, 16, N_grid)
        mu_broadcast = jnp.tile(ENV_MU, (n_agents, 1))
        # Adapted: ensure d=64 matches training
        pe = get_2d_sinusoidal_encoding(xi_norm, d=64) 
        
        obs_cat = jnp.concatenate([y, mu_broadcast, pe], axis=-1)
        action = marl_model.apply(p, obs_cat)
        
        return action.squeeze(-1)
    
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 3. RL (Centralized God-View)
rl_model = CentralizedActor(n_agents=n_agents)
# FIX: Force float32 for dummy input to ensure proper Flax Network initialization
rl_dummy_input = jnp.zeros((1, N_grid, N_grid), dtype=jnp.float32) 

rl_p = load_params(bench_models_dir / 'rl_turb_params.msgpack', rl_model, rl_dummy_input)
if not rl_p:
    # Fallback to check the root models directory if not moved yet
    rl_p = load_params(Path('models/rl_turb_params.msgpack'), rl_model, rl_dummy_input)

if rl_p:
    def rl_apply(p, xi_fixed, obs):
        # Match training script: ensure 2D, scale, cast to float32
        obs_squeeze = obs.squeeze()
        obs_norm = (obs_squeeze / STATE_NORM_FACTOR).astype(jnp.float32)
        
        # Add batch dim manually: (1, N_grid, N_grid)
        action = rl_model.apply(p, obs_norm[None, ...])
        
        # Squeeze batch dim and cast back to float64 for PDE solver
        return action.squeeze().astype(jnp.float64) 
    
    bench_registry['RL'] = {'apply': rl_apply, 'params': rl_p, 'color': 'green'}

# 4. Uncontrolled Baseline
bench_registry['Uncontrolled'] = {
    'apply': lambda p, xi_fixed, obs: jnp.zeros(n_agents), 
    'params': None, 'color': 'red'
}

# --- 2. Simulation ---
print(f"Loading Turbulence Spectral Data and Running Simulations...")

data_dir = Path('../../data')
file_path = data_dir / 'turbulence_chaotic_ics_64_more.pkl'
if file_path.exists():
    with open(file_path, 'rb') as f:
        w_hat_pool = jnp.array(pickle.load(f)[:N_eval])
else:
    print("Generating ICs on the fly...")
    key = jax.random.PRNGKey(1234)
    w_hat_pool = get_batch_initial_conditions(key, N_eval, N_grid, L_domain, viscosity=5e-4)

xi_batch = jnp.tile(xi_init, (N_eval, 1, 1))

def run_sim(name, w_hat_init, xi_i):
    apply_fn = bench_registry[name]['apply']
    params = bench_registry[name]['params']
    dyn = PDEDynamics2D(policy_apply_fn=apply_fn)
    
    w_phys_traj, u_ctrl_traj = dyn.unroll_controlled(
        w_hat_init=w_hat_init, 
        xi_fixed=xi_i, 
        params=params, 
        t_steps=T_steps,
        substeps=substeps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        viscosity=viscosity,
        actuator_grid_shape=(8, 8) 
    )
    return w_phys_traj 

for name in bench_registry:
    print(f"Running {name} unrolls...")
    
    @jax.jit
    def batched_sim(w_batch, x_batch):
        return jax.vmap(lambda w, x: run_sim(name, w, x))(w_batch, x_batch)
        
    w_phys_res = batched_sim(w_hat_pool, xi_batch)
    bench_registry[name]['data'] = w_phys_res

# --- 3. Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Final Enstrophy':<20} | {'2-Sigma':<20}")
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
    
    initial_state = jnp.fft.ifft2(w_hat_pool[0]).real
    final_state = bench_registry[name]['data'][0, -1]
    
    vmin, vmax = float(jnp.min(initial_state)), float(jnp.max(initial_state))
    
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(initial_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title('Initial Vorticity')
    plt.colorbar(im1, label='ω(x,y)')
    
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(final_state, aspect='auto', origin='lower', extent=[0, L_domain, 0, L_domain], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.title(f'Final Controlled State: {name}')
    plt.colorbar(im2, label='ω(x,y)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"state_{name.lower()}.pdf")
    plt.close()

# --- 5. Plotting Trendlines ---
plt.figure(figsize=(18, 8))

plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean((bench_registry[n]['data'][:, -1])**2, axis=(1, 2)) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final System Enstrophy (Log Scale)')
plt.ylabel('Mean L2 Vorticity')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps) * substeps * dt
for name in bench_registry:
    evol = jnp.mean(jnp.mean(bench_registry[name]['data']**2, axis=(2, 3)), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Stabilization Enstrophy Evolution')
plt.xlabel('Time (s)')
plt.ylabel('Mean Enstrophy (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "turbulence_stabilization_results.png")
print(f"\nSummary results saved to {output_dir}/turbulence_stabilization_results.png")