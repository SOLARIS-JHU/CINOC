# CINOC: Cardinality-Invariant Neural Operator Policies for Scalable PDE Control

> "You speak for the whole planet, do you? For the common consciousness of every dewdrop, of every pebble, of even the liquid central core of the planet?"
> 
> "I do, and so can any portion of the planet in which the intensity of the common consciousness is great enough."
>
> — **Isaac Asimov**, *Foundation and Earth*

<!-- [TODO]: ricontrolla nome e ricontrolla link per paper -->
Implementation of [Cardinality-Invariant Neural Operator Policies for Scalable PDE Control]() by [Pietro Zanotta](https://github.com/PietroZanotta)<sup>1, 2</sup>, [Dibakar Roy Sarkar](https://github.com/RoyDibs)<sup>1, 2</sup>, [Honghui Zheng](https://github.com/Honghui-Zheng)<sup>1, 2</sup>, [Somdatta Goswami](https://github.com/somdattagoswami)<sup>2</sup>, [Ján Drgoňa](https://github.com/drgona)<sup>2</sup>.

<sup>1</sup>: Equal contribution
<sup>2</sup>: Department of Civil and System Engineering, Johns Hopkins University, Baltimore, USA

Correspondence to: [pzanott1@jh.edu](mailto:pzanott1@jh.edu).

![ks1d](/examples/ks1d/decentralized/figures/images/gif/ks1d_multiscale_comparison.gif)
---

## Overview

Controlling complex physical systems governed by partial differential equations (PDEs) with learning-based policies is traditionally bottlenecked by fixed-dimensional representations. If the number of sensors, actuators, or agents in a swarm changes, the policy architecture typically fails and must be retrained from scratch.

**CINOC** addresses this fundamental limitation by reformulating multi-agent PDE control as an operator learning problem. By mapping local state field observations to continuous control functions and training end-to-end via differentiable physics solvers, our framework yields policies that naturally adapt to varying sensor and actuator configurations. 

### Key Features & Discoveries

* **Cardinality Invariance (Zero-Shot Scalability):** Policies trained on small swarms can be deployed zero-shot onto significantly larger populations without requiring any fine-tuning.
* **Stigmergic Coordination:** Decentralized agents share a common policy and coordinate implicitly through their shared physical environment. This drives an emergent self-normalization effect, where individual agents automatically scale back their control efforts as the swarm size increases.
* **Theoretical Guarantees:** Grounded in mean-field theory, we proove that policy gradients computed from finite-agent systems reliably converge to the exact gradients of the continuous control limit.
* **Versatile PDE Benchmarks:** The framework is empirically validated on diverse tasks—including trajectory tracking, chaotic stabilization, and density transport—across linear, nonlinear, chaotic, and turbulent PDE regimes.
* **High Robustness:** Learned operators exhibit graceful degradation and robustness to partial agent failures, sensor noise, changes in grid resolution, and minor parametric shifts in the underlying physics.
 
---

## Getting Started

### Prerequisites
0. **Clone the repository:**
```bash
git clone https://github.com/SOLARIS-JHU/CINOC
cd CINOC
```

1. **Set up Python virtual environment:**
```bash
python -m venv .venv
```

Activate the virtual environment:
- **Linux/MacOS:**
```bash
source .venv/bin/activate
```
- **Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Experiments:**
```python
cd examples heat2D/decentralized # or any other repository
python3 train.py
python3 visualize_paper.py
```

---
## Tutorials

Looking to learn more? We have a collection of step-by-step guides and examples to help you get started. 

Check out the **[Tutorials directory](./tutorials)** for more information.

---
## Repository Structure

The structure of this repository is the following:
```text
CINOC/
├── assets/                         # Assets for README
├── examples/                       # High-level scripts for specific PDE problems
│   ├── fkpp1d/                     # Fisher-KPP 1D reaction-diffusion trajectory tracking example
│   │    └── decentralized/          
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │ 
│   ├── heat1d/                     # 1D Heat Equation example
│   │    └── decentralized/          
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │
│   │── heat2D/                     # 2D Heat Equation trajectory tracking example   
│   │   └── decentralized/
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │ 
│   │── heat2D_obstacles/           # 2D Heat Equation trajectory tracking with geometric obstacles example   
│   │   └── decentralized/
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │
│   │── ks1d/                       # ks1d stabilization example
│   │    └── decentralized/
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │
│   │── ks2d/                       # ks2d stabilization example   
│   │    └── decentralized/
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │
│   │── turbulence2d/               # Turbulence stabilization example   
│   │    └── decentralized/
│   │       ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │
│   │── density/                    # Density control example   
│   │   └── centralized/
│   │        ├── bench/              # RL and DPC benchmarking code
│   │       ├── figures/            # Figures for ablations
│   │       └── models/             # Pretrained models
│   │
├── models/                         # Core neural network architectures
│   │── policy.py                   # DPC policies for Heat1d, Heat2d, Heat2d Obs, FKPP1d
│   ├── policy_ks1d.py              # DPC policy for KS1D 
│   ├── policy_ks2d.py              # DPC policy for KS2D 
│   ├── policy_ns2d.py              # DPC policy for Density Control problem
│   └── policy_turb.py              # DPC policy for Turbulence stabilization problem 
│ 
├── tesseracts/                     # Numerical simulators logic
│   ├── solverFKPP_.../             # Solver specifically for FKPP problems
│   ├── solverHeat_.../             # Solvers specifically for Heat problems
│   │   ├── solver.py               # The underlying physics engine logic
│   └── ...
│
├── tutorials/                      # Pedagogical Jupiter Notebooks         
│   ├── fkpp1d/             
│   ├── tutorial.ipynb   
│   ├── heat2d/             
│   └── tutorial.ipynb            
│   
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---


## Tech Stack

- **Processor:** Intel Core Ultra 9 275HX (24 cores, up to 5.4 GHz)
- **GPU:** NVIDIA GeForce RTX 5090 Laptop GPU (24GB GDDR7 VRAM)
- **Operating System:** Ubuntu 22.04 running under Windows Subsystem for Linux (WSL2)
- **Main Frameworks:** JAX (v0.8.1) for numerical computing; Tesseract-JAX (v0.2.2) for differentiable PDE solvers
- **Hardware Acceleration:** CUDA backend with NVIDIA driver v581.57

---
