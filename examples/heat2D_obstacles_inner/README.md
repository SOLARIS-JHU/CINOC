# 2D Heat Equation Control with Inner Obstacles (Continuous Penalty)

This experiment demonstrates Differentiable Predictive Control (DPC) for the 2D heat equation with obstacles placed **between actuator initial positions**, using a **continuous distance penalty** for collision avoidance.

## Key Differences from `heat2D_obstacles`

### 1. Obstacle Placement
- **Original (`heat2D_obstacles`)**: Obstacles placed outside agent grid (left, right, bottom)
- **This experiment**: Obstacles placed BETWEEN agents in their initial 4×4 grid
  - (0.34, 0.34) - Bottom-left quadrant
  - (0.50, 0.50) - Center (between four middle agents)
  - (0.66, 0.66) - Top-right quadrant

### 2. Safe Distance
- **Original**: R_safe = 0.08
- **This experiment**: R_safe = 0.05 (reduced minimum safe distance)

### 3. Collision Penalty
- **Continuous distance penalty**: Agents are penalized based on proximity to obstacles
  ```python
  safety_distances = R_safe + obstacle_radii
  l_coll_obstacles = mean(max(0, safety_distances - dists_obstacles)^2)
  ```
- Penalty increases **smoothly** as agents approach obstacles
- Agents maintain safety margin of R_safe = 0.05 from obstacle surface

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
2. Maintain minimum distance of R_safe = 0.05 from obstacle surfaces
3. Drive temperature field toward target configuration
4. Avoid inter-agent collisions

The continuous penalty provides smooth gradients for learning obstacle avoidance.

## Comparison with `heat2D_obstacles_inner_sharp`

- This experiment: **Continuous penalty** (penalize based on proximity)
- Sharp version: **Sharp penalty** (only penalize when inside obstacle)

See `../heat2D_obstacles_inner_sharp/README.md` for details on the sharp penalty approach.
