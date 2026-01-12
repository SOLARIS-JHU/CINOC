"""
Policy Network for 2D Schnakenberg Pattern Control.

This module implements a DeepONet-style controller for the Schnakenberg system:
- Dual-species control (activator + inhibitor injection)
- Mobile agents with 2D velocities
- CNN branch for 2D error field processing
- Fourier trunk for periodic position encoding
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


def circular_pad_2d(x, pad_width):
    """Periodic padding for 2D fields with channels. x shape: (H, W, C)"""
    return jnp.pad(x, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='wrap')


class SchnakenbergControlNet(nn.Module):
    """
    Centralized Controller for 2D Schnakenberg Pattern Control.
    
    Architecture:
    - Branch: CNN processes 2D error field (u_error, v_error, gradients)
    - Trunk: Fourier encoding of 2D actuator positions (periodic)
    - Fusion: DeepONet-style broadcast + concatenation
    - Heads: 
        - Activator injection rate (u_ctrl): scalar per agent
        - Inhibitor injection rate (v_ctrl): scalar per agent  
        - Velocity (vel): 2D vector per agent
    """
    features: Sequence[int] = (32, 64, 64)  # CNN channels
    u_max: float = 1.0       # Max activator injection
    v_max: float = 1.0       # Max inhibitor injection
    vel_max: float = 1.0     # Max velocity
    L_domain: float = 10.0   # Domain size
    
    def setup(self):
        # Periodic Fourier frequencies (2π ensures continuity at boundaries)
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0]) * 2.0 * jnp.pi / self.L_domain
    
    def branch_net(self, u_error, v_error, u_grad_x, u_grad_y, v_grad_x, v_grad_y):
        """
        CNN branch for 2D error field processing.
        
        Input: 6 channels (u_error, v_error, 4 gradient components)
        Output: Global context vector
        """
        # Stack into (N, N, 6)
        x = jnp.stack([u_error, v_error, u_grad_x, u_grad_y, v_grad_x, v_grad_y], axis=-1)
        
        for feat in self.features:
            # Periodic padding before convolution
            x = circular_pad_2d(x, pad_width=1)
            x = nn.Conv(feat, kernel_size=(3, 3), padding='VALID')(x)
            x = nn.relu(x)
        
        # Global pooling + compression
        x = jnp.mean(x, axis=(0, 1))  # Global average pool to (C,)
        x = nn.Dense(128)(x)
        x = nn.tanh(x)
        
        return x
    
    def trunk_net(self, xi):
        """
        Fourier encoding for 2D actuator positions.
        
        Args:
            xi: (M, 2) actuator positions in domain coordinates
            
        Returns:
            (M, trunk_dim) position encodings
        """
        # Fourier features for periodic domain
        angle_x = xi[:, 0, None] * self.frequencies
        angle_y = xi[:, 1, None] * self.frequencies
        
        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)  # (M, 32)
        
        # MLP
        for feat in [64, 64]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        
        return encoded
    
    @nn.compact
    def __call__(self, u_curr, v_curr, u_target, v_target, xi_curr):
        """
        Forward pass.
        
        Args:
            u_curr: (N, N) current activator field
            v_curr: (N, N) current inhibitor field
            u_target: (N, N) target activator field
            v_target: (N, N) target inhibitor field
            xi_curr: (M, 2) current actuator positions
            
        Returns:
            u_ctrl: (M,) activator injection rates
            v_ctrl: (M,) inhibitor injection rates
            vel: (M, 2) actuator velocities
        """
        # Clip positions to normalized domain [0, 1] (inputs are normalized)
        xi_curr = jnp.clip(xi_curr, 0.0, 1.0)
        
        # Error fields
        u_error = u_curr - u_target
        v_error = v_curr - v_target
        
        # Periodic gradients (central difference with roll)
        dx = self.L_domain / u_curr.shape[0]
        
        u_grad_x = (jnp.roll(u_error, -1, axis=1) - jnp.roll(u_error, 1, axis=1)) / (2 * dx)
        u_grad_y = (jnp.roll(u_error, -1, axis=0) - jnp.roll(u_error, 1, axis=0)) / (2 * dx)
        v_grad_x = (jnp.roll(v_error, -1, axis=1) - jnp.roll(v_error, 1, axis=1)) / (2 * dx)
        v_grad_y = (jnp.roll(v_error, -1, axis=0) - jnp.roll(v_error, 1, axis=0)) / (2 * dx)
        
        # Branch: global context from error field
        branch_out = self.branch_net(u_error, v_error, u_grad_x, u_grad_y, v_grad_x, v_grad_y)
        
        # Trunk: position encoding
        trunk_out = self.trunk_net(xi_curr)
        
        # Fusion: broadcast branch to all agents
        branch_repeated = jnp.tile(branch_out, (xi_curr.shape[0], 1))
        combined = jnp.concatenate([branch_repeated, trunk_out], axis=-1)
        
        # Shared hidden layers
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)
        
        # Output heads
        u_ctrl_raw = nn.Dense(1)(h).squeeze(-1)  # (M,)
        v_ctrl_raw = nn.Dense(1)(h).squeeze(-1)  # (M,)
        vel_raw = nn.Dense(2)(h)                  # (M, 2)
        
        # Scale to physical limits
        u_ctrl = self.u_max * jnp.tanh(u_ctrl_raw)
        v_ctrl = self.v_max * jnp.tanh(v_ctrl_raw)
        vel = self.vel_max * jnp.tanh(vel_raw)
        
        return u_ctrl, v_ctrl, vel


class DecentralizedSchnakenbergControlNet(nn.Module):
    """
    Decentralized Controller - each agent only sees local patch.
    """
    features: Sequence[int] = (16, 32)
    patch_size: int = 16
    u_max: float = 1.0
    v_max: float = 1.0
    vel_max: float = 1.0
    L_domain: float = 10.0
    n_grid: int = 64
    
    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0]) * 2.0 * jnp.pi / self.L_domain
    
    def extract_periodic_patch(self, field_6ch, xi):
        """Extract local patch with periodic boundary handling."""
        # Convert domain coordinates to grid indices
        center_i = ((xi[1] / self.L_domain) * self.n_grid).astype(int)
        center_j = ((xi[0] / self.L_domain) * self.n_grid).astype(int)
        
        # Roll field so agent is at center
        shift_i = (self.n_grid // 2) - center_i
        shift_j = (self.n_grid // 2) - center_j
        rolled = jnp.roll(field_6ch, (shift_i, shift_j), axis=(0, 1))
        
        # Slice from center
        start_i = (self.n_grid // 2) - (self.patch_size // 2)
        start_j = (self.n_grid // 2) - (self.patch_size // 2)
        
        return jax.lax.dynamic_slice(
            rolled,
            (start_i, start_j, 0),
            (self.patch_size, self.patch_size, 6)
        )
    
    def branch_net(self, patch):
        """Process local patch."""
        x = patch
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = x / (jnp.linalg.norm(x) + 1.0)
            x = nn.tanh(x)
        
        x = x.reshape(-1)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        return x
    
    def trunk_net(self, xi):
        """Fourier position encoding."""
        angle_x = xi[:, 0, None] * self.frequencies
        angle_y = xi[:, 1, None] * self.frequencies
        
        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)
        
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded
    
    @nn.compact
    def __call__(self, u_curr, v_curr, u_target, v_target, xi_curr):
        # Errors and gradients
        u_error = u_curr - u_target
        v_error = v_curr - v_target
        
        u_grad_x = jnp.roll(u_error, -1, axis=1) - jnp.roll(u_error, 1, axis=1)
        u_grad_y = jnp.roll(u_error, -1, axis=0) - jnp.roll(u_error, 1, axis=0)
        v_grad_x = jnp.roll(v_error, -1, axis=1) - jnp.roll(v_error, 1, axis=1)
        v_grad_y = jnp.roll(v_error, -1, axis=0) - jnp.roll(v_error, 1, axis=0)
        
        # Stack (N, N, 6)
        full_field = jnp.stack([u_error, v_error, u_grad_x, u_grad_y, v_grad_x, v_grad_y], axis=-1)
        
        # Extract patches for each agent
        patches = jax.vmap(self.extract_periodic_patch, in_axes=(None, 0))(full_field, xi_curr)
        branch_outs = jax.vmap(self.branch_net)(patches)
        
        # Trunk
        trunk_outs = self.trunk_net(xi_curr)
        
        # Fusion
        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        
        # Output heads
        u_ctrl = self.u_max * jnp.tanh(nn.Dense(1)(h).squeeze(-1))
        v_ctrl = self.v_max * jnp.tanh(nn.Dense(1)(h).squeeze(-1))
        vel = self.vel_max * jnp.tanh(nn.Dense(2)(h))
        
        return u_ctrl, v_ctrl, vel
