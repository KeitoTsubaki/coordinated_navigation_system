# Coordinated Formation Control of Multi-robot using WHILL

## Overview
This project aims to develop a multi-robot navigation system for WHILL mobility devices. <br>
Each robots navigate to the goal keeping formation optimized in their surrounding environment. Based on their surrounding objects, they should optimize their formation (e.g., line formation in the narrow passage, or moving side by side in an afordable situation).<br>
Considering this system is developed for WHILL mobilities that is ridden by human, the adaptation of formation is kind to humans, such as not too high speed, not too close to obstacles.  

## Features
- **Coordinated Navigation**: Multiple WHILL devices moving in unison. (At first, I only use two WHILLs)
- **Formation Control**: Maintaining specific geometric formations (e.g., line, wedge)
- **Reinforcement Learning**
- **ROS2**

## System Architecture


## Repository Structure
```bash
coordinated_navigation_system\
├── .gitignore
├── .gitmodules
├── GEMINI.md
├── README.md
├── requirements.txt
├── .vscode/
│   └── settings.json
│
├── rl_training/                      # <-- "The Brain": Your RL algorithms live here.
│   ├── train_ddpg.py
│   ├── your_new_rl_agent.py          # Your new agent that learns from the simulator.
│   ├── requirements.txt              # Python dependencies for RL (PyTorch, etc.).
│   ├── ... (other existing files)
│
└── ros2_ws/                          # <-- "The Body": The robot environment lives here.
    ├── src/
    │   ├── ... (your other ros2 packages)
    │   └── matplotlib_2d_simulator/  # <-- NEW: The ROS2 2D Simulator Package.
    │       ├── matplotlib_2d_simulator/  # Python source code for the node.
    │       │   ├── __init__.py
    │       │   ├── simulator_node.py     # Main executable ROS2 node.
    │       │   ├── environment.py        # Defines the 2D world.
    │       │   └── robot.py              # Simulates robot physics & LiDAR.
    │       ├── config/
    │       │   └── simulator_config.yaml # All simulation parameters.
    │       ├── launch/
    │       │   └── simulator.launch.py   # A convenient way to run the node.
    │       ├── resource/
    │       │   └── matplotlib_2d_simulator # Required for package discovery.
    │       ├── setup.py                  # Build instructions for colcon.
    │       └── package.xml               # ROS2 package manifest.
    │
    ├── build/   (Will be created by `colcon build`)
    ├── install/ (Will be created by `colcon build`)
    └── log/     (Will be created by `colcon build`)
```

## Getting Started
- PyTorchのインストール
- 必要パッケージのインストール
```
uv init

uv pip install torch --torch-backend=auto (NVIDIA DriverさえあればOK)

uv sync

cd simulation_core
uv pip install .
# editable mode を使うなら
uv pip install -e .
```
### Prerequisistes
- Ubuntu 22.04
- ROS2 Humble
- Python 3.13
### Installation


## Usage


## Modules

### Navigation Module

### Formation Module

### Velocity Adjustment Module

The renderer applies a formation-aware velocity adjustment helper
(`multi_robot/renderer/speed_adjustment.py`). The helper implements the
following attenuation scheme:

* `l_theta = ((-cos(theta_f) + 1) / 2) ^ eta_theta` reflects the formation
  heading error.
* `l_d = min(((1 / d_f - 1 / T_df) ^ eta_d) / k_d, 1)` when `d_f < T_df`,
  otherwise zero, captures how far the robot lags behind the formation target.
* `l_t = max(l_theta - l_d, 0)` represents the lateral deviation gate.

The executable commands are blended as `v_a = K[v_e * (1 - l_t)]` and
`omega_a = K[omega_e + min(|d_o|, 1) * (omega_f - omega_e)]`. `K[·]` denotes
saturation to the physical limits `[0, V_MAX]` and `[-OMEGA_MAX, OMEGA_MAX]`,
while `omega_f` keeps a fixed magnitude of `0.4π rad/s` (its sign follows the
heading error) so the robot turns at a consistent rate in the direction of the
formation target.

The tuning constants live in `simulation_core/config/simulator_config.yaml`:

```yaml
speed_adjustment:
  eta_theta: 1.0  # Exponent for heading attenuation
  eta_d: 1.0      # Exponent for distance attenuation
  k_d: 0.5        # Distance attenuation gain
  omega_fixed: 1.2566370614 # Fixed magnitude (0.4 * pi rad/s)
  T_df: 1.0       # Distance threshold (m)
```

The renderer (`PolicyController`) applies this helper whenever a viable
formation navigation target is available. During reinforcement learning
training, formation objectives are ignored and the raw policy commands are
executed after applying only the acceleration limits, keeping the training
signal focused on goal-seeking behaviour.

### Formation defaults

The simulator reads default formation settings from
`simulation_core/config/simulator_config.yaml`. The new `formation`
section lets you control the initial robot count, preferred pattern, and
robot spacing:

```yaml
formation:
  robot_count: 3          # Number of robots spawned when no override is provided
  default_type: "triangle"  # Formation label passed to the planner
  spacing_m: 1.5          # Desired spacing in metres
```

`spacing_m` is specified in metres and automatically converted into pixels
using the current map scale (`PIXELS_PER_METER`). The resolved value is
exposed in `MultiRobotSimulator.default_formation_spacing` (pixels) and
`MultiRobotSimulator.default_formation_spacing_m` (metres). Passing
`None` for the `num_robots`, `formation_type`, or `spacing` arguments
when constructing the simulator or optimising a formation causes these
defaults to be used.

To visualise the formation-centre cost landscape during optimisation, set
`formation_debug_plot_enabled` to `true` in the simulator configuration or
toggle the attribute on a `MultiRobotSimulator` instance. When enabled, the
renderer collects every candidate centre and displays a scatter plot coloured by
cost using Matplotlib; this requires the optional `matplotlib` dependency to be
installed.

## Roadmap

