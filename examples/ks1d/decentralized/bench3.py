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

# --- NEW: L0 Sparse Polynomial Definitions ---
class SparsePolynomialLayer(nn.Module):
    out_features: int = 1
    gamma: float = -0.1
    zeta: float = 1.1
    @nn.compact
    def __call__(self, x):
        weights = self.param('weights', nn.initializers.glorot_uniform(), (x.shape[-1], self.out_features))
        # 2D log_alpha for independent actuator masks
        log_alpha = self.param('log_alpha', nn.initializers.constant(0.0), (x.shape[-1], self.out_features))
        
        s = jax.nn.sigmoid(log_alpha)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate = jnp.clip(s_stretched, 0.0, 1.0) 
        
        return jnp.dot(x, weights * gate)

class MARLPolynomialActor(nn.Module):
    max_action: float = 1.0
    n_agents: int = 8
    @nn.compact
    def __call__(self, poly_features):
        # We must name this "sparse_layer" to match the training checkpoint
        x = SparsePolynomialLayer(out_features=self.n_agents, name="sparse_layer")(poly_features)
        return self.max_action * jnp.tanh(x)

# Local state is 8 sensors + 2 mu params = 10
N_grid, L_domain, n_agents = 128, 22.0, 8
n_sensors = 8
OBS_DIM_FOR_L0 = n_sensors + 2 

# The exact number of features for degree=2: 1 + N + N*(N+1)//2
POLY_DIM = 1 + OBS_DIM_FOR_L0 + (OBS_DIM_FOR_L0 * (OBS_DIM_FOR_L0 + 1)) // 2

# Precompute indices for cross-terms statically to guarantee trace safety
_r, _c = np.triu_indices(OBS_DIM_FOR_L0)

@jax.jit
def get_poly_features_jax(x):
    """Pure JAX equivalent of PolynomialFeatures(degree=2, include_bias=True)"""
    is_1d = x.ndim == 1
    x_2d = jnp.atleast_2d(x)
    
    def poly_single(feat):
        bias = jnp.ones((1,))
        linear = feat
        outer = jnp.outer(feat, feat)
        quad = outer[_r, _c] # Extracts cross terms and squares
        return jnp.concatenate([bias, linear, quad])
        
    res = jax.vmap(poly_single)(x_2d)
    return res[0] if is_1d else res
# ---------------------------------------------

# --- 2. Configuration & Paths ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

output_dir = Path("figures/images/bench_new")
output_dir.mkdir(parents=True, exist_ok=True)

bench_models_dir = Path("bench/models")
bench_models_dir.mkdir(parents=True, exist_ok=True) 

from examples.ks1d.decentralized.dynamics_dual import PDEDynamics 
from models.policy_ks1d import DecentralizedControlNet
from examples.ks1d.decentralized.data_utils import get_batch_initial_conditions

# Make sure these paths are correct for your repo structure
from examples.ks1d.decentralized.bench.models_hypemarl import HyperActor
from examples.ks1d.decentralized.bench.utils_hypemarl import get_sinusoidal_encoding
from examples.ks1d.decentralized.bench.env_ks import extract_patches_jit

T_steps, N_eval, dt = 200, 100, 0.05

bench_registry = {}

# --- 3. Loading Logic ---

def load_params(filename, model, dummy_input):
    if not os.path.exists(filename):
        print(f"[-] {filename} not found. Skipping benchmark.")
        return None
        
    with open(filename, 'rb') as f:
        bytes_data = f.read()
        
    # Get the expected variable structure from Flax
    variables = model.init(jax.random.PRNGKey(0), *dummy_input)
    
    try:
        # Unpack the raw bytes into a python dictionary
        state_dict = msgpack_restore(bytes_data)
    except Exception as e:
        print(f"[-] Could not parse msgpack for {filename}: {e}")
        return None

    # Handle common nesting from different training scripts
    if 'actor' in state_dict: 
        state_dict = state_dict['actor']
        
    # FIX: If Flax expects a 'params' root key, but our saved dict doesn't have it, wrap it.
    if 'params' in variables and 'params' not in state_dict:
        state_dict = {'params': state_dict}
    # (Optional safeguard) If saved dict has 'params' but Flax doesn't expect it
    elif 'params' not in variables and 'params' in state_dict:
        state_dict = state_dict['params']

    try:
        # Restore the state dict safely
        return from_state_dict(variables, state_dict)
    except Exception as e:
        print(f"[-] Failed to load weights for {filename}. Mismatch error: {e}")
        # Fallback to direct byte deserialization just in case
        try:
            return flax.serialization.from_bytes(variables, bytes_data)
        except Exception as e2:
            print(f"[-] Fallback also failed: {e2}")
            return None

print("Loading Models...")

# 1. HypeMARL 
hm_model = HyperActor()
xi_single = jnp.linspace(0.0, L_domain, n_agents, endpoint=False) + (L_domain/n_agents)/2
hm_pe = get_sinusoidal_encoding(xi_single, d=2048)
# Handle the dummy input gracefully if it crashes during init.
try:
    hm_z = jnp.concatenate([hm_pe, jnp.tile(jnp.array([L_domain, 0.05]), (n_agents, 1))], axis=-1)
    hm_p = load_params(bench_models_dir / 'hypemarl_params.msgpack', hm_model, (hm_z, jnp.zeros((n_agents, 40))))
except Exception as e:
    print(f"[-] HypeMARL initialization failed: {e}")
    hm_p = None

if hm_p:
    def hm_apply(p, u, target, xi):
        y = extract_patches_jit(u, target, xi/L_domain, 4)
        return hm_model.apply(p, hm_z, y)
    bench_registry['HypeMARL'] = {'apply': hm_apply, 'params': hm_p, 'color': 'green'}

# 2. DPC 
dpc_model = DecentralizedControlNet(features=(64, 64), L_domain=L_domain)
dpc_p = load_params('ks_centralized_params.msgpack', dpc_model, (jnp.zeros(N_grid), jnp.zeros(N_grid), xi_single))
if dpc_p:
    bench_registry['DPC'] = {'apply': dpc_model.apply, 'params': dpc_p, 'color': 'blue'}

# 3. MARL 
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

# 4. RL 
rl_model = CentralizedActor()
rl_p = load_params(bench_models_dir / 'rl_centralized_params.msgpack', rl_model, (jnp.zeros(N_grid),))

if rl_p:
    bench_registry['RL'] = {'apply': lambda p, u, t, xi: rl_model.apply(p, u), 'params': rl_p, 'color': 'purple'}

# --- NEW: 5. L0 Sparse Poly ---
l0_model = MARLPolynomialActor(n_agents=n_agents)
dummy_poly = jnp.ones((POLY_DIM,))

# Update filename to whatever your final L0 epoch checkpoint is named
l0_p = load_params(bench_models_dir / 'actor_poly_ep490.msgpack', l0_model, (dummy_poly,))

# Pre-calculate sensor indices exactly as done in the KSEnv
sensor_indices = np.linspace(0, N_grid, n_sensors, endpoint=False, dtype=int)

if l0_p:
    def l0_apply(p, u, target, xi):
        # Extract the partial observation from the full PDE grid
        sensor_readings = u[sensor_indices]
        mu = jnp.array([L_domain, dt])
        obs = jnp.concatenate([sensor_readings, mu])
        
        # Pure JAX trace-safe expansion!
        poly_state = get_poly_features_jax(obs)
        
        # Guard clause: Flax serialization sometimes strips the 'params' root key
        apply_params = p if 'params' in p else {'params': p}
        
        return l0_model.apply(apply_params, poly_state)
    
    bench_registry['L0_Poly'] = {'apply': l0_apply, 'params': l0_p, 'color': 'cyan'}
# ---------------------------------------------
    
# Uncontrolled
bench_registry['Uncontrolled'] = {
    'apply': lambda p, u, t, xi: jnp.zeros((n_agents,)), 
    'params': None, 'color': 'red'
}

# --- 4. Simulation ---
print(f"Generating Data & Running Simulations for {list(bench_registry.keys())}...")
key = jax.random.PRNGKey(1234)
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
    final_e = jnp.mean(bench_registry[name]['data'][:, -1]**2, axis=1)
    mean_val, std_val = jnp.mean(final_e), jnp.std(final_e)
    print(f"{name:<15} | {mean_val:.6f}      | ±{2*std_val:.6f}")

if 'L0_Poly' in bench_registry:
    l0_params = bench_registry['L0_Poly']['params']['params']['sparse_layer']['log_alpha']
    gamma, zeta = -0.1, 1.1
    s_stretched = jax.nn.sigmoid(l0_params) * (zeta - gamma) + gamma
    gate = jnp.clip(s_stretched, 0.0, 1.0)
    active_count = jnp.sum(gate > 0)
    print("-" * 70)
    print(f"L0 Model Active Parameters: {active_count} / {gate.size}")

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

# 1. Boxplot 
plt.subplot(1, 2, 1)
data_boxplot = [jnp.mean(bench_registry[n]['data'][:, -1]**2, axis=1) for n in bench_registry]
plt.boxplot(data_boxplot, labels=list(bench_registry.keys()))
plt.yscale('log')
plt.title('Final System Energy (Log Scale)')
plt.ylabel('L2 Energy')
plt.grid(True, alpha=0.3)

# 2. Energy Evolution 
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