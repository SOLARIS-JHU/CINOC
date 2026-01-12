"""
Dataset Generation for Nodal-Lefty Pattern Control

Creates training pairs based on the pattern control problem from:
Ouchdiri et al., "An optimal-control framework for reaction diffusion 
systems with application to synthetic developmental biology," 2025.

The task: Transform patterns from one (αn, αl) configuration to another.
E.g., Striped → Spotted, or vice versa.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
import argparse

# Add path
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

from tesseracts.RDP2d.nodal_lefty_solver import (
    NodalLeftyConfig,
    solve_nodal_lefty,
    generate_initial_condition,
    get_pattern_config,
    build_imex_operators_neumann,
    imex_step_nodal_lefty,
    hill_function
)


# =============================================================================
# Pattern Generation
# =============================================================================

def evolve_to_pattern(
    key: jax.Array,
    alpha_n: float,
    alpha_l: float,
    t_hours: float = 200.0,
    config: NodalLeftyConfig = None
) -> tuple:
    """
    Evolve random IC to stable pattern for given (αn, αl).
    
    Returns:
        yn_init, yl_init: Initial random condition
        yn_final, yl_final: Stable pattern
    """
    if config is None:
        config = NodalLeftyConfig()
    
    # Generate random IC
    key, subkey = jax.random.split(key)
    yn_init, yl_init = generate_initial_condition(subkey, config, noise_level=0.2)
    
    # Dummy agent positions (not used for pattern generation)
    xi_init = jnp.zeros((1, 2)) + config.L / 2
    
    # Evolve
    t_steps = int(t_hours / config.dt)
    
    (yn_hist, yl_hist, _) = solve_nodal_lefty(
        yn_init, yl_init, xi_init,
        t_steps=t_steps,
        N_grid=config.N,
        L=config.L,
        dt=config.dt,
        D_n=config.D_n,
        D_l=config.D_l,
        gamma_n=config.gamma_n,
        gamma_l=config.gamma_l,
        n_n=config.n_n,
        n_l=config.n_l,
        k_n=config.k_n,
        k_l=config.k_l,
        alpha_n=alpha_n,
        alpha_l=alpha_l,
        sigma=config.sigma
    )
    
    yn_final = yn_hist[-1]
    yl_final = yl_hist[-1]
    
    return yn_init, yl_init, yn_final, yl_final


def generate_pattern_atlas(
    n_patterns: int,
    pattern_types: list = None,
    seed: int = 42,
    t_hours: float = 200.0,
    L: float = 800.0,
    N: int = 80
    ) -> dict:
    """
    Generate an atlas of stable patterns for different (αn, αl) configurations.
    
    Returns dict with keys: 'striped', 'spotted', 'modified_spotted', etc.
    Each value is a dict with 'yn', 'yl', 'alpha_n', 'alpha_l'
    """
    if pattern_types is None:
        pattern_types = ['striped', 'spotted', 'modified_spotted', 'modified_striped']
    
    config = NodalLeftyConfig(L=L, N=N)
    atlas = {}
    
    key = jax.random.PRNGKey(seed)
    
    for pattern_type in pattern_types:
        print(f"\nGenerating {n_patterns} '{pattern_type}' patterns...")
        alpha_n, alpha_l, desc = get_pattern_config(pattern_type)
        
        yn_list = []
        yl_list = []
        
        for i in tqdm(range(n_patterns)):
            key, subkey = jax.random.split(key)
            _, _, yn_final, yl_final = evolve_to_pattern(
                subkey, alpha_n, alpha_l, t_hours=t_hours, config=config
            )
            yn_list.append(yn_final)
            yl_list.append(yl_final)
        
        atlas[pattern_type] = {
            'yn': jnp.stack(yn_list, axis=0),
            'yl': jnp.stack(yl_list, axis=0),
            'alpha_n': alpha_n,
            'alpha_l': alpha_l,
            'description': desc
        }
    
    return atlas


# =============================================================================
# Dataset Creation
# =============================================================================

def create_control_dataset(
    atlas: dict,
    n_train: int = 100,
    n_val: int = 20,
    n_test: int = 20,
    initial_pattern: str = 'striped',
    target_pattern: str = 'spotted',
    L: float = 800.0,
    N: int = 80,
    config: NodalLeftyConfig = None,
    seed: int = 42
) -> dict:
    """
    Create training dataset for pattern control.
    
    The task: Given initial state from 'initial_pattern' configuration,
    control to reach target state from 'target_pattern' configuration.
    
    Returns dict with train/val/test splits, each containing:
        yn_init, yl_init: Initial fields
        yn_target, yl_target: Target pattern
        alpha_n_init, alpha_l_init, alpha_n_target, alpha_l_target: Parameters
    """
    if config is None:
        config = NodalLeftyConfig(L=L, N=N)
    else:
        # Override if config is passed but make sure it matches desired L/N if they were somehow different
        pass 
    
    key = jax.random.PRNGKey(seed)
    
    # Get pattern pools
    init_pool = atlas[initial_pattern]
    target_pool = atlas[target_pattern]
    
    n_init_patterns = init_pool['yn'].shape[0]
    n_target_patterns = target_pool['yn'].shape[0]
    
    def create_split(n_samples: int) -> dict:
        nonlocal key
        
        key, k1, k2 = jax.random.split(key, 3)
        
        # Random indices
        init_idx = jax.random.randint(k1, (n_samples,), 0, n_init_patterns)
        target_idx = jax.random.randint(k2, (n_samples,), 0, n_target_patterns)
        
        # For initial condition, use the pattern as starting point
        # (already converged, so we start from the natural attractor)
        return {
            'yn_init': init_pool['yn'][init_idx],
            'yl_init': init_pool['yl'][init_idx],
            'yn_target': target_pool['yn'][target_idx],
            'yl_target': target_pool['yl'][target_idx],
            'alpha_n_init': init_pool['alpha_n'],
            'alpha_l_init': init_pool['alpha_l'],
            'alpha_n_target': target_pool['alpha_n'],
            'alpha_l_target': target_pool['alpha_l'],
        }
    
    return {
        'train': create_split(n_train),
        'val': create_split(n_val),
        'test': create_split(n_test),
        'config': {
            'N': config.N,
            'L': config.L,
            'D_n': config.D_n,
            'D_l': config.D_l,
            'gamma_n': config.gamma_n,
            'gamma_l': config.gamma_l,
            'n_n': config.n_n,
            'n_l': config.n_l,
            'k_n': config.k_n,
            'k_l': config.k_l,
            'beta_n': config.beta_n,
            'beta_l': config.beta_l,
            'dt': config.dt,
        }
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_atlas(atlas: dict, save_path: Path):
    """Visualize pattern atlas."""
    n_types = len(atlas)
    
    fig, axes = plt.subplots(2, n_types, figsize=(4*n_types, 8))
    
    for i, (pattern_type, data) in enumerate(atlas.items()):
        # Show first pattern
        yn = data['yn'][0]
        yl = data['yl'][0]
        
        im1 = axes[0, i].imshow(yn.T, origin='lower', cmap='jet')
        axes[0, i].set_title(f"{pattern_type}\nαn={data['alpha_n']}, αl={data['alpha_l']}")
        if i == 0:
            axes[0, i].set_ylabel('Nodal')
        plt.colorbar(im1, ax=axes[0, i])
        
        im2 = axes[1, i].imshow(yl.T, origin='lower', cmap='jet')
        if i == 0:
            axes[1, i].set_ylabel('Lefty')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle('Nodal-Lefty Pattern Atlas', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


def visualize_dataset(dataset: dict, n_samples: int = 4, save_path: Path = None):
    """Visualize training samples."""
    train = dataset['train']
    
    fig, axes = plt.subplots(4, n_samples, figsize=(4*n_samples, 12))
    
    for i in range(min(n_samples, len(train['yn_init']))):
        # Initial Nodal
        axes[0, i].imshow(train['yn_init'][i].T, origin='lower', cmap='jet')
        axes[0, i].set_title(f"Sample {i}")
        if i == 0:
            axes[0, i].set_ylabel('Initial yn')
        
        # Initial Lefty
        axes[1, i].imshow(train['yl_init'][i].T, origin='lower', cmap='jet')
        if i == 0:
            axes[1, i].set_ylabel('Initial yl')
        
        # Target Nodal
        axes[2, i].imshow(train['yn_target'][i].T, origin='lower', cmap='jet')
        if i == 0:
            axes[2, i].set_ylabel('Target yn')
        
        # Target Lefty
        axes[3, i].imshow(train['yl_target'][i].T, origin='lower', cmap='jet')
        if i == 0:
            axes[3, i].set_ylabel('Target yl')
    
    plt.suptitle(f"Training Samples: αn={train['alpha_n_init']}→{train['alpha_n_target']}, "
                 f"αl={train['alpha_l_init']}→{train['alpha_l_target']}", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Nodal-Lefty control dataset')
    parser.add_argument('--n_patterns', type=int, default=20, 
                        help='Number of patterns per type')
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=20)
    parser.add_argument('--n_test', type=int, default=20)
    parser.add_argument('--initial', type=str, default='striped',
                        choices=['striped', 'spotted', 'modified_spotted', 'modified_striped'])
    parser.add_argument('--target', type=str, default='spotted',
                        choices=['striped', 'spotted', 'modified_spotted', 'modified_striped'])
    parser.add_argument('--t_hours', type=float, default=200.0,
                        help='Evolution time for pattern formation')
    parser.add_argument('--L', type=float, default=800.0, help='Domain size (micron)')
    parser.add_argument('--N', type=int, default=80, help='Grid resolution')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    print("="*60)
    print("Nodal-Lefty Pattern Control Dataset Generation")
    print("="*60)
    
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Generate atlas
    print("\n--- Generating Pattern Atlas ---")
    atlas = generate_pattern_atlas(
        n_patterns=args.n_patterns,
        pattern_types=[args.initial, args.target],
        t_hours=args.t_hours,
        L=args.L,
        N=args.N
    )
    
    # Save atlas
    atlas_np = {}
    for ptype, data in atlas.items():
        for k, v in data.items():
            if isinstance(v, jnp.ndarray):
                atlas_np[f"{ptype}_{k}"] = np.array(v)
            else:
                atlas_np[f"{ptype}_{k}"] = v
    np.savez(data_dir / 'pattern_atlas.npz', **atlas_np)
    print(f"\nSaved atlas to {data_dir / 'pattern_atlas.npz'}")
    
    # Create dataset
    print(f"\n--- Creating Control Dataset: {args.initial} → {args.target} ---")
    dataset = create_control_dataset(
        atlas,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        initial_pattern=args.initial,
        target_pattern=args.target,
        L=args.L,
        N=args.N
    )
    
    # Save splits
    for split in ['train', 'val', 'test']:
        split_np = {k: np.array(v) if isinstance(v, jnp.ndarray) else v 
                    for k, v in dataset[split].items()}
        np.savez(data_dir / f'{split}_data.npz', **split_np)
        print(f"Saved {split}: {len(dataset[split]['yn_init'])} samples")
    
    # Save config
    np.savez(data_dir / 'config.npz', **dataset['config'])
    
    # Visualize
    if args.visualize:
        print("\n--- Generating Visualizations ---")
        visualize_atlas(atlas, data_dir / 'atlas_preview.png')
        visualize_dataset(dataset, n_samples=4, save_path=data_dir / 'training_samples.png')
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print(f"Files saved to: {data_dir}")
    print("="*60)
