"""
Conference-Quality Visualization for 2D Turbulence (Navier-Stokes) Stabilization.
Adapted for Decentralized Control (64 Agents, Viscous Decay).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import flax.serialization
from pathlib import Path

# Force CPU for visualization
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics2D
from models.policy_turb import DecentralizedTurbulenceNet 
from data_utils import get_batch_initial_conditions

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'N_grid': 128,          
    'L_domain': 1.0,        
    
    # Physics Match
    'dt': 0.0005,           
    'viscosity': 5e-5,      
    'substeps': 20,         
    
    'n_agents': 64,         
    'grid_shape': (8, 8),   
    
    # Visualization Timeline 
    'T_chaos_steps': 50,    # 0.5s of chaos
    'T_control_steps': 200, # 2.0s of control
    
    # Snapshots (Physical Time)
    'snapshot_times': [-0.25, 0.0, 0.25, 0.75, 1.5], 
    
    'params_file': 'turbulence_params.msgpack'
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_actuator_grid():
    """Reconstructs the 8x8 grid automatically."""
    grid_dim = int(np.sqrt(CONFIG['n_agents']))
    x_lin = np.linspace(0, CONFIG['L_domain'], grid_dim, endpoint=False) + (CONFIG['L_domain']/grid_dim)/2
    xv, yv = np.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)

def load_params(model, filepath):
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Parameter file {filepath} not found.")
        
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    
    key = jax.random.PRNGKey(42)
    xi_fixed = get_actuator_grid()
    dummy_obs = jnp.zeros((1, CONFIG['N_grid'], CONFIG['N_grid']))
    
    # Init model to get structure
    init_params = model.init(key, xi_fixed, dummy_obs)
    
    # Overwrite with saved weights
    return flax.serialization.from_bytes(init_params, serialized_bytes)

# --- FIXED ZERO POLICY ---
def get_zero_policy(n_agents):
    """
    Returns a policy function compatible with PDEDynamics2D.
    Args:
        n_agents: Number of agents to shape the output zero vector correctly.
    """
    def zero_policy_fn(params, xi, obs):
        # Arguments match PDEDynamics call: self.policy_apply_fn(params, xi_fixed, obs)
        # Returns shape (n_agents,) to ensure u_cmd is not scalar
        return jnp.zeros((n_agents,))
    
    return zero_policy_fn

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def generate_transition_data(key, model, params):
    """Simulates: Chaos -> Transition -> Stabilization."""
    
    # 1. Setup Dynamics
    dyn_chaos = PDEDynamics2D(policy_apply_fn=get_zero_policy(CONFIG['n_agents']))
    dyn_control = PDEDynamics2D(policy_apply_fn=model.apply)
    
    # 2. Get Initial Chaotic State (Spectral)
    # The data loader returns Spectral states (Complex128)
    w0_hat = get_batch_initial_conditions(key, 1, CONFIG['N_grid'], CONFIG['L_domain'])[0]
    
    xi_fixed = get_actuator_grid()
    
    # 3. Phase 1: Run Uncontrolled
    print(f"  [Sim] Running Chaos Phase ({CONFIG['T_chaos_steps']*CONFIG['substeps']*CONFIG['dt']:.2f}s)...")
    
    # Input: Spectral w0_hat
    # Output: Physical Trajectory (w_traj_chaos)
    w_traj_chaos, _ = dyn_chaos.unroll_controlled(
        w0_hat, xi_fixed, params,
        t_steps=CONFIG['T_chaos_steps'],
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], 
        L=CONFIG['L_domain'], 
        dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'],
        actuator_grid_shape=CONFIG['grid_shape']
    )
    
    # 4. Phase 2: Run Controlled
    w_handoff_phys = w_traj_chaos[-1]
    
    # --- FIX: Convert Physical -> Spectral for Solver ---
    w_handoff_hat = jnp.fft.fft2(w_handoff_phys)
    
    print(f"  [Sim] Running Control Phase ({CONFIG['T_control_steps']*CONFIG['substeps']*CONFIG['dt']:.2f}s)...")
    
    w_traj_ctrl, u_force_ctrl = dyn_control.unroll_controlled(
        w_handoff_hat, xi_fixed, params,
        t_steps=CONFIG['T_control_steps'],
        substeps=CONFIG['substeps'],
        N_grid=CONFIG['N_grid'], 
        L=CONFIG['L_domain'], 
        dt=CONFIG['dt'],
        viscosity=CONFIG['viscosity'],
        actuator_grid_shape=CONFIG['grid_shape']
    )
    
    # 5. Stitch (Both are now Physical)
    w_full = jnp.concatenate([w_traj_chaos, w_traj_ctrl], axis=0)
    
    dt_effective = CONFIG['substeps'] * CONFIG['dt']
    n_chaos = CONFIG['T_chaos_steps']
    n_ctrl = CONFIG['T_control_steps']
    
    t_chaos = (jnp.arange(n_chaos) - n_chaos) * dt_effective
    t_ctrl  = jnp.arange(n_ctrl) * dt_effective
    t_full = jnp.concatenate([t_chaos, t_ctrl])
    
    return t_full, w_full, u_force_ctrl

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
    })

def plot_2d_transition(t_full, w_full, u_force_ctrl, example_id=1, save_name="turb_transition.png"):
    setup_academic_style()
    
    # Enstrophy
    enstrophy = jnp.mean(w_full**2, axis=(1,2))
    
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.5], hspace=0.35)
    
    # --- Row 1: Snapshots ---
    target_times = np.array(CONFIG['snapshot_times'])
    snap_indices = []
    
    for t_req in target_times:
        idx = (np.abs(t_full - t_req)).argmin()
        snap_indices.append(idx)
        
    gs_snaps = gridspec.GridSpecFromSubplotSpec(1, len(snap_indices), subplot_spec=gs[0], wspace=0.1)
    
    max_val = jnp.max(jnp.abs(w_full[0])) * 0.8
    vmin, vmax = -max_val, max_val
    
    for i, idx in enumerate(snap_indices):
        ax = fig.add_subplot(gs_snaps[i])
        w_snap = w_full[idx]
        t_snap = t_full[idx]
        
        im = ax.imshow(w_snap, extent=[0, CONFIG['L_domain'], 0, CONFIG['L_domain']], 
                       origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        if i == 0:
            ax.set_ylabel(r"$y$")
            ax.set_yticks([0, CONFIG['L_domain']])
        else:
            ax.set_yticks([])
            
        ax.set_xticks([0, CONFIG['L_domain']])
        ax.set_xlabel(r"$x$")
        
        if t_snap < -1e-4:
            status = "Chaos"
            color = 'firebrick'
        elif t_snap > 1e-4:
            status = "Control ON"
            color = 'navy'
        else:
            status = "Switching"
            color = 'black'
            
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
            
        ax.set_title(f"t = {t_snap:.2f}s\n{status}", color=color, fontweight='bold')
        
        # Plot Actuators
        xi = get_actuator_grid()
        ax.scatter(xi[:,0], xi[:,1], c='k', s=10, alpha=0.3, marker='x') 

    cax = fig.add_axes([0.92, 0.55, 0.015, 0.3])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"Vorticity $\omega(x,y)$")

    # --- Row 2: Enstrophy Trace ---
    ax_ts = fig.add_subplot(gs[1])
    
    mask_chaos = t_full <= 0
    mask_ctrl = t_full >= 0
    
    ax_ts.plot(t_full[mask_chaos], enstrophy[mask_chaos], color='firebrick', lw=2, label='Uncontrolled')
    ax_ts.plot(t_full[mask_ctrl], enstrophy[mask_ctrl], color='navy', lw=2, label='Controlled')
    
    ax_ts.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax_ts.set_yscale('log')
    ax_ts.set_xlim(t_full[0], t_full[-1])
    ax_ts.set_ylabel(r"Enstrophy $\langle \omega^2 \rangle$")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.legend(loc='upper right')
    ax_ts.grid(True, which='both', linestyle='--', alpha=0.3)
    
    final_e = enstrophy[-1]
    ax_ts.text(t_full[-1], final_e, f" Final: {final_e:.2e}", va='bottom', ha='right', fontweight='bold')

    ax_ts.set_title(f"(b) Turbulence Suppression - Example {example_id}", loc='left', fontweight='bold')
    plt.suptitle(f"2D Turbulence Stabilization (Example {example_id} | {CONFIG['n_agents']} Agents)", y=0.98, fontsize=18)
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_name}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("--- 2D Turbulence Visualization Script (Generating 3 Examples) ---")
    
    model = DecentralizedTurbulenceNet(
        features=(32, 64), 
        domain_size=(CONFIG['L_domain'], CONFIG['L_domain']),
        u_max=75.0
    )
    
    try:
        print(f"Loading params from {CONFIG['params_file']}...")
        params = load_params(model, CONFIG['params_file'])
    except Exception as e:
        print(f"Error: {e}")
        print("Run training first to generate parameters.")
        sys.exit(1)
        
    base_key = jax.random.PRNGKey(55) 
    num_examples = 3
    
    for i in range(num_examples):
        print(f"\n═══ Generating Example {i+1} / {num_examples} ═══")
        rng_run, base_key = jax.random.split(base_key)
        t_full, w_full, u_force = generate_transition_data(rng_run, model, params)
        
        filename = f"turb_transition_ex{i+1}.png"
        plot_2d_transition(t_full, w_full, u_force, example_id=i+1, save_name=filename)
        
    print("\nAll examples generated successfully.")