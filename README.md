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

### Run the pygame renderer demo
1. Install the dependencies (at minimum `pygame` is required for the demo). If
   you are using the provided requirements file, run:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the interactive viewer from the project root:

   ```bash
   python -m multi_robot.renderer
   ```

   The window starts with two robots and an example map. You can experiment
   with different behaviours by editing `multi_robot/renderer/core.py` or by
   instantiating `MultiRobotSimulator` from your own scripts.

3. Use the following keyboard shortcuts while the window is focused:

   | Key | Action |
   | --- | ------ |
   | `W` | Move the selected robot forward |
   | `Q` / `E` | Rotate the selected robot left / right |
   | `TAB` | Cycle through robots |
   | `1`&ndash;`9` (`0` for the 10th) | Jump to a specific robot |
   | `C` | Load the next map layout |
   | `SPACE` | Reset the current map |
   | `L` | Toggle the LiDAR beams for the selected robot |
   | `ESC` | Quit the simulator |

   The simulator records each robot's path and displays HUD information in the
   side panel to help with debugging formations and sensor data.

## Modules

### Navigation Module

### Formation Module

### Velocity Adjustment Module


## Roadmap