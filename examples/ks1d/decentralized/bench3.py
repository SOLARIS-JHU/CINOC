import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax.linen as nn
import flax.serialization
from flax.serialization import msgpack_restore, from_state_dict
import sys
import os
from pathlib import Path

# --- 1. Model Definitions (Flax) ---

class MARLActor(nn.Module):
    hidden_dim: int = 256
    action_dim: int = 1
    u_max: float = 1.0
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1.0)
        out = nn.Dense(self.action_dim)(x)
        return jnp.tanh(out) * self.u_max

class CentralizedActor(nn.Module):
    hidden_dim: int = 256
    n_agents: int = 8
    u_max: float = 1.0
    @nn.compact
    def __call__(self, u_field):
        x = nn.Dense(self.hidden_dim)(u_field)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        out = nn.Dense(self.n_agents)(x)
        return jnp.tanh(out) * self.u_max

# --- 2. Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

# Ensure output directory exists
output_dir = Path("figures/images/bench")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the specific folder for benchmark weights
bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

from examples.ks1d.decentralized.bench.models_hypemarl import HyperActor
from examples.ks1d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.ks1d.decentralized.bench.env_ks import extract_patches_jit

N_grid, L_domain, n_agents = 128, 22.0, 8
T_steps, N_eval, dt = 200, 100, 0.05

# Registry for available benchmarks
bench_registry = {}

# --- 3. Loading Logic ---

def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found. Skipping benchmark.")
        return None
    with open(filename, 'rb') as f:
        bytes_data = f.read()
    variables = model.init(jax.random.PRNGKey(0), *dummy_input)
    try:
        state_dict = msgpack_restore(bytes_data)
        if 'actor' in state_dict: state_dict = state_dict['actor']
        return from_state_dict(variables, state_dict)
    except:
        return flax.serialization.from_bytes(variables, bytes_data)

print("Loading Models...")

# 1. HypeMARL (Look in bench/models/)
hm_model = HyperActor()
xi_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
hm_pe = get_sinusoidal_encoding(xi_single, d=2048)
hm_z = jnp.concatenate([hm_pe, jnp.tile(jnp.array([L_domain, 0.05]), (n_agents, 1))], axis=-1)

hm_p = load_params(bench_models_dir / 'hypemarl_params.msgpack', hm_model, (hm_z, jnp.zeros((n_agents, 40))))

if hm_p:
    def hm_apply(p, u, target, xi):
        y = extract_patches_jit(u, target, xi/L_domain, 4)
        return hm_model.apply(p, hm_z, y)
    bench_registry['HypeMARL'] = {'apply': hm_apply, 'params': hm_p, 'color': 'green'}

# 2. DPC (LOCATION UNCHANGED - Root directory)
dpc_model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)
dpc_p = load_params('ks_centralized_params.msgpack', dpc_model, (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_single))
if dpc_p:
    bench_registry['DPC'] = {'apply': dpc_model.apply, 'params': dpc_p, 'color': 'blue'}

# 3. MARL (Look in bench/models/)
marl_model = MARLActor()
marl_mu_val = jnp.array([L_domain, 0.05]) 
marl_pe_val = get_sinusoidal_encoding(xi_single, d=2048)

marl_p = load_params(bench_models_dir / 'marl_standard_params.msgpack', marl_model, (jnp.zeros((n_agents, 2090)),))

if marl_p:
    def marl_apply(p, u, target, xi):
        y = extract_patches_jit(u, target, xi/L_domain, window_size=4)
        mu_context = jnp.tile(marl_mu_val, (n_agents, 1))
        full_input = jnp.concatenate([y, mu_context, marl_pe_val], axis=-1)
        return marl_model.apply(p, full_input).flatten()
    bench_registry['MARL'] = {'apply': marl_apply, 'params': marl_p, 'color': 'orange'}

# 4. RL (Look in bench/models/)
rl_model = CentralizedActor()
rl_p = load_params(bench_models_dir / 'rl_params.msgpack', rl_model, (jnp.zeros(N_grid),))

if rl_p:
    bench_registry['RL'] = {'apply': lambda p, u, t, xi: rl_model.apply(p, u), 'params': rl_p, 'color': 'purple'}
    
# Uncontrolled
bench_registry['Uncontrolled'] = {
    'apply': lambda p, u, t, xi: jnp.zeros((n_agents,)), 
    'params': None, 'color': 'red'
}

# --- 4. Simulation ---
print(f"Generating Data & Running Simulations for {list(bench_registry.keys())}...")
key = jax.random.PRNGKey(42)
u_init_batch = get_batch_initial_conditions(jax.random.split(key)[1], N_eval, N_grid, L_domain)
xi_batch = jnp.tile(xi_single, (N_eval, 1))

@jax.jit(static_argnames=['name'])
def run_sim(name, u_i, xi_i):
    dyn = PDEDynamics(policy_apply_fn=bench_registry[name]['apply'])
    u, _, _, _ = dyn.unroll_controlled(
        u_i, xi_i, jnp.zeros_like(u_i), 
        bench_registry[name]['params'], 
        T_steps, N_grid, L_domain, dt=dt
    )
    return u

for name in bench_registry:
    print(f"Running {name} unrolls...")
    results_data = jax.vmap(lambda u, x: run_sim(name, u, x))(u_init_batch, xi_batch)
    bench_registry[name]['data'] = results_data

# --- 5. Metrics & Results Printing ---
print("\n" + "="*70)
print(f"{'Method':<15} | {'Mean Energy':<15} | {'2-Sigma':<20}")
print("-" * 70)

for name in bench_registry:
    # Mean squared field at final timestep across batch
    final_e = jnp.mean(bench_registry[name]['data'][:, -1]**2, axis=1)
    mean_val, std_val = jnp.mean(final_e), jnp.std(final_e)
    print(f"{name:<15} | {mean_val:.6f}      | ±{2*std_val:.6f}")
print("="*70)

# --- 6. Plotting & PDF Export ---
print("Saving individual state plots to PDF...")
for name in bench_registry:
    plt.figure(figsize=(8, 5))
    plt.imshow(bench_registry[name]['data'][0].T, aspect='auto', origin='lower', 
               extent=[0, T_steps*dt, 0, L_domain], cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar(label='u(x,t)')
    plt.title(f'Controlled State: {name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Space (x)')
    plt.tight_layout()
    plt.savefig(output_dir / f"state_{name.lower()}.pdf")
    plt.close()

plt.figure(figsize=(18, 8))

# 1. Boxplot (Includes Uncontrolled)
plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean(bench_registry[n]['data'][:, -1]**2, axis=1) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final System Energy (Log Scale)')
plt.ylabel('L2 Energy')
plt.grid(True, alpha=0.3)

# 2. Energy Evolution (Log Scale, Excludes Uncontrolled)
plt.subplot(1, 2, 2)
time_axis = jnp.arange(T_steps) * dt
for name in bench_registry:
    if name == 'Uncontrolled': continue 
    
    evol = jnp.mean(jnp.mean(bench_registry[name]['data']**2, axis=2), axis=0)
    plt.plot(time_axis, evol, label=name, color=bench_registry[name]['color'], lw=2.5)

plt.yscale('log')
plt.title('Stabilization Energy Evolution (Controlled Only)')
plt.xlabel('Time (s)')
plt.ylabel('Mean Energy (Log)')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "ks_resilient_results.png")
print(f"\nSummary results saved to {output_dir}/ks_resilient_results.png")