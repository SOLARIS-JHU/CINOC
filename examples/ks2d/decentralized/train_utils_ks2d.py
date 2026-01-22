"""
Modular Training Utilities for KS2D Ablation Studies

Provides reusable training logic for:
1. Noise Robustness Study
2. Sensor Dimension Ablation
3. Agent Count Transfer Study
"""

import jax
import jax.numpy as jnp
import optax
import time
import pickle
import flax.serialization
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from tqdm import trange

jax.config.update("jax_enable_x64", True)

from dynamics_dual import PDEDynamics2D
from models.policy_ks2d import DecentralizedKS2DControlNet
from data_utils import get_batch_initial_conditions


def get_agent_grid(n_agents, L_domain):
    """
    Creates uniform 2D grid of agents. n_agents must be perfect square.

    Args:
        n_agents: Number of agents (must be perfect square)
        L_domain: Domain size

    Returns:
        xi_fixed: (n_agents, 2) array of agent positions
    """
    grid_dim = int(jnp.sqrt(n_agents))
    assert grid_dim**2 == n_agents, f"n_agents={n_agents} must be perfect square"

    spacing = L_domain / grid_dim
    x_lin = jnp.linspace(0, L_domain, grid_dim, endpoint=False) + spacing/2
    xv, yv = jnp.meshgrid(x_lin, x_lin)
    return jnp.stack([xv.flatten(), yv.flatten()], axis=-1)


def get_or_create_data(N_grid, L_domain, pool_size, ic_filename='ks2d_chaotic_ics.pkl'):
    """
    Manages loading/generating chaotic Initial Conditions.

    Args:
        N_grid: Grid resolution
        L_domain: Domain size
        pool_size: Number of ICs to generate
        ic_filename: Filename for cached data

    Returns:
        u_pool: (pool_size, N_grid, N_grid) array of initial conditions
    """
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / ic_filename

    if file_path.exists():
        print(f"[Data] Found existing ICs at: {file_path}")
        with open(file_path, 'rb') as f:
            u_pool = pickle.load(f)
        if u_pool.shape[1] == N_grid:
            return jnp.array(u_pool)
        print(f"[Data] Resolution mismatch. Regenerating...")

    print(f"[Data] Generating {pool_size} chaotic states...")
    key = jax.random.PRNGKey(42)
    u_pool = get_batch_initial_conditions(key, pool_size, N_grid, L_domain)

    import numpy as np
    with open(file_path, 'wb') as f:
        pickle.dump(np.array(u_pool), f)

    return u_pool


def train_ks2d(
    # Physics params
    N_grid=64,
    L_domain=32.0,
    dt=0.005,
    substeps=20,
    sigma=1.2,

    # Training params
    n_agents=100,
    batch_size=4,
    T_steps=50,
    epochs=500,
    pool_size=500,

    # Noise params
    noise_u=0.0,
    noise_z=0.0,

    # Model config
    model=None,
    patch_size=12,

    # Optimizer config
    learning_rate_init=5e-4,
    learning_rate_peak=1e-3,
    learning_rate_end=1e-5,
    warmup_steps=50,

    # Loss weights
    weight_track=100.0,
    weight_effort=1e-4,

    # I/O
    save_dir="./",
    params_file="params.msgpack",
    plot_file="training.png",
    plot_metrics=True,
    verbose=True
):
    """
    Modular training function for KS2D controllers.

    Args:
        N_grid: Spatial resolution
        L_domain: Domain size
        dt: Physics timestep
        substeps: Physics steps per control step
        sigma: Actuator width
        n_agents: Number of agents (must be perfect square)
        batch_size: Training batch size
        T_steps: Control steps per trajectory
        epochs: Training epochs
        pool_size: Size of initial condition pool
        noise_u: Actuator noise magnitude
        noise_z: Sensor noise magnitude
        model: Pre-initialized model (if None, creates default)
        patch_size: Local patch size for decentralized sensing
        learning_rate_*: Learning rate schedule parameters
        weight_*: Loss function weights
        save_dir: Directory to save outputs
        params_file: Filename for trained parameters
        plot_file: Filename for training plot
        plot_metrics: Whether to plot training metrics
        verbose: Whether to print progress

    Returns:
        params: Trained model parameters
        metrics: Training metrics array
    """
    if verbose:
        horizon = T_steps * substeps * dt
        print(f"--- KS2D Training (N={N_grid}, L={L_domain}, Horizon={horizon:.2f}s) ---")
        print(f"    Noise: u={noise_u:.3f}, z={noise_z:.3f}")
        print(f"    Agents: {n_agents}, Patch: {patch_size}px")

    key = jax.random.PRNGKey(42)

    # --- 1. Load Data ---
    u_init_pool = get_or_create_data(N_grid, L_domain, pool_size)
    u_target_pool = jnp.zeros_like(u_init_pool)

    # --- 2. Setup Actuators ---
    xi_fixed_single = get_agent_grid(n_agents, L_domain)
    xi_fixed_batch = jnp.tile(xi_fixed_single, (batch_size, 1, 1))

    # --- 3. Initialize Model ---
    if model is None:
        model = DecentralizedKS2DControlNet(
            features=(64, 128),
            domain_size=(L_domain, L_domain),
            u_max=5.0,
            patch_size=patch_size
        )

    key, init_key = jax.random.split(key)
    dummy_u = jnp.zeros((N_grid, N_grid))
    dummy_xi = jnp.zeros((n_agents, 2))
    params = model.init(init_key, dummy_u, dummy_u, dummy_xi)

    # --- 4. Optimizer ---
    # Ensure decay_steps is positive (must be > warmup_steps)
    effective_decay_steps = max(epochs, warmup_steps + 1)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate_init,
        peak_value=learning_rate_peak,
        warmup_steps=min(warmup_steps, epochs // 2),  # Adjust warmup if epochs is small
        decay_steps=effective_decay_steps,
        end_value=learning_rate_end
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    opt_state = optimizer.init(params)

    # --- 5. Dynamics Wrapper ---
    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)

    # --- 6. Loss Function ---
    def loss_fn(params, u_init, xi_fixed, u_target, rng_key, dynamics):
        u_traj, _, u_ctrl_traj, _ = dynamics.unroll_controlled(
            u_init,
            xi_fixed,
            u_target,
            params,
            t_steps=T_steps,
            substeps=substeps,
            N_grid=N_grid,
            L=L_domain,
            dt=dt,
            sigma=sigma,
            key=rng_key,
            noise_u=noise_u,
            noise_z=noise_z
        )

        # Tracking Loss
        traj_error = u_traj - u_target[None, :, :]
        l_track = jnp.mean(traj_error ** 2)

        # Control Effort Loss
        l_effort = jnp.mean(u_ctrl_traj ** 2)

        # Total Loss
        total_loss = weight_track * l_track + weight_effort * l_effort

        return total_loss, (l_track, l_effort)

    @partial(jax.jit, static_argnames='dynamics')
    def train_step(params, opt_state, u_init_batch, xi_fixed_batch, u_target_batch, rng_key, dynamics):
        # Split keys for batch
        keys = jax.random.split(rng_key, u_init_batch.shape[0])

        batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, None))

        def mean_loss_fn(p):
            losses, auxs = batched_loss_fn(p, u_init_batch, xi_fixed_batch, u_target_batch, keys, dynamics)
            return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)

        (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # --- 7. Training Loop ---
    metrics = []
    start_time = time.time()

    if verbose:
        pbar = trange(epochs, desc="Training")
    else:
        pbar = range(epochs)

    for epoch in pbar:
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, pool_size)
        u_init_b = u_init_pool[idx]
        u_target_b = u_target_pool[idx]

        # Generate fresh key for noise injection
        key, step_key = jax.random.split(key)

        params, opt_state, loss, aux = train_step(
            params, opt_state, u_init_b, xi_fixed_batch, u_target_b, step_key, dynamics
        )

        track_loss, effort_loss = aux
        metrics.append([loss, track_loss, effort_loss])

        if verbose and epoch % 10 == 0:
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Track": f"{track_loss:.4f}"})

    if verbose:
        print(f"Training Complete. Time: {time.time()-start_time:.1f}s")

    # --- 8. Save Parameters ---
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    params_path = save_path / params_file
    with open(params_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))

    if verbose:
        print(f"Parameters saved: {params_path}")

    # --- 9. Plot Metrics ---
    metrics = jnp.array(metrics)

    if plot_metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(metrics[:, 1], label='Tracking MSE', color='blue')
        plt.plot(metrics[:, 2], label='Control Effort', color='orange', alpha=0.7)
        plt.yscale('log')
        plt.title(f'KS2D Training (Noise u={noise_u:.3f}, z={noise_z:.3f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        plot_path = save_path / plot_file
        plt.savefig(plot_path)
        plt.close()

        if verbose:
            print(f"Metrics plotted: {plot_path}")

    return params, metrics


def evaluate_policy(
    params,
    model,
    u_init,
    xi_fixed,
    u_target,
    N_grid=64,
    L_domain=32.0,
    dt=0.005,
    substeps=20,
    T_steps=50,
    sigma=1.2,
    key=None,
    noise_u=0.0,
    noise_z=0.0
):
    """
    Evaluates a trained policy on a single trajectory.

    Args:
        params: Trained model parameters
        model: Policy model
        u_init: Initial state
        xi_fixed: Agent positions
        u_target: Target state
        ... (physics parameters)
        key: Random key for noise
        noise_u: Actuator noise
        noise_z: Sensor noise

    Returns:
        u_traj: State trajectory
        final_mse: Final tracking MSE
        energy_decay: Energy decay rate
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    dynamics = PDEDynamics2D(policy_apply_fn=model.apply)

    u_traj, _, u_ctrl_traj, _ = dynamics.unroll_controlled(
        u_init,
        xi_fixed,
        u_target,
        params,
        t_steps=T_steps,
        substeps=substeps,
        N_grid=N_grid,
        L=L_domain,
        dt=dt,
        sigma=sigma,
        key=key,
        noise_u=noise_u,
        noise_z=noise_z
    )

    # Compute metrics
    final_mse = jnp.mean((u_traj[-1] - u_target)**2)

    # Energy decay rate
    energy = jnp.mean(u_traj**2, axis=(1, 2))
    energy_decay = (energy[0] - energy[-1]) / energy[0]

    return u_traj, final_mse, energy_decay
