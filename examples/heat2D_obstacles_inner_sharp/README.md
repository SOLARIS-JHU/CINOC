# 2D Heat Equation Control with Inner Obstacles (Sharp Penalty)

This experiment demonstrates Differentiable Predictive Control (DPC) for the 2D heat equation with obstacles placed **between actuator initial positions**, using a **sharp collision penalty** that only activates when agents are inside obstacles.

## Key Differences from `heat2D_obstacles`

### 1. Obstacle Placement
- **Original (`heat2D_obstacles`)**: Obstacles placed outside agent grid (left, right, bottom)
- **This experiment**: Obstacles placed BETWEEN agents in their initial 4×4 grid
  - (0.34, 0.34) - Bottom-left quadrant
  - (0.50, 0.50) - Center (between four middle agents)
  - (0.66, 0.66) - Top-right quadrant

### 2. Safe Distance
- **Original**: R_safe = 0.08
- **This experiment**: R_safe = 0.05 (reduced minimum safe distance, only used for agent-agent collisions)

### 3. Collision Penalty (KEY DIFFERENCE)
- **Sharp penalty**: Agents are ONLY penalized when their center is INSIDE the obstacle
  ```python
  # No safety margin - only penalize when distance < obstacle_radius
  l_coll_obstacles = mean(max(0, obstacle_radii - dists_obstacles)^2)
  ```
- Penalty activates **abruptly** at obstacle boundary
- No gradual repulsion - agents can get very close without penalty
- More challenging for learning due to sparse gradient signal

## Agent Initialization

Agents are initialized in a 4×4 grid at positions:
- x, y ∈ {0.26, 0.42, 0.58, 0.74}

Obstacles are strategically placed in the gaps between these positions, forcing agents to navigate around them during control.

## Usage

### Quick Test (Centralized)
```bash
cd centralized
python train.py --test
```

### Quick Test (Decentralized)
```bash
cd decentralized
python train.py --test
```

### Full Training (Centralized)
```bash
cd centralized
python train.py
```

### Full Training (Decentralized)
```bash
cd decentralized
python train.py
```

## Expected Behavior

Agents must learn to:
1. Navigate around obstacles placed within their formation
2. Avoid penetrating obstacles (no safety margin)
3. Drive temperature field toward target configuration
4. Avoid inter-agent collisions (R_safe = 0.05 maintained)

The sharp penalty provides:
- **Sparse gradients**: Only when colliding
- **Harder learning**: Less guidance for avoidance
- **Potential benefit**: Agents can get closer to obstacles when needed

## Comparison with `heat2D_obstacles_inner`

| Feature | `heat2D_obstacles_inner` | `heat2D_obstacles_inner_sharp` (this) |
|---------|--------------------------|---------------------------------------|
| **Obstacle penalty** | Continuous (proximity-based) | Sharp (only when inside) |
| **Penalty activation** | distance < R_safe + obstacle_radius | distance < obstacle_radius |
| **Gradient signal** | Smooth, informative | Sparse, abrupt |
| **Learning difficulty** | Easier (smooth gradients) | Harder (sparse gradients) |
| **Agent behavior** | Maintain safety margin | Can get very close |

The sharp penalty is more challenging but may allow agents to utilize space more efficiently by getting closer to obstacles when needed for control.

## Implementation Details

The key difference is in the collision loss calculation:

**Continuous penalty** (`heat2D_obstacles_inner`):
```python
safety_distances = R_safe + obstacle_radii[None, None, :]
l_coll_obstacles = jnp.mean(jnp.maximum(0, safety_distances - dists_obstacles) ** 2)
```

**Sharp penalty** (this experiment):
```python
# Only penalize when agent center is INSIDE obstacle
l_coll_obstacles = jnp.mean(jnp.maximum(0, obstacle_radii[None, None, :] - dists_obstacles) ** 2)
```

This removes the R_safe safety margin for obstacles, creating a discontinuous penalty function.
