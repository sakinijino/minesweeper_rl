# Minesweeper RL Agent

A reinforcement learning agent that plays Minesweeper using MaskablePPO and a custom CNN architecture.

## Quick Start

### Installation

```bash
# Clone and setup
git clone <your-repository-url>
cd minesweeper_rl

# Create a Virtual Environment (Recommended):**
python -m venv .venv
# Activate the environment
source .venv/bin/activate # on macos/linux
# .venv\Scripts\activate # on windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python train.py

# Advanced training with custom parameters
python train.py \
  --total_timesteps 1_000_000 \
  --n_envs 4 \
  --learning_rate 0.0001 \
  --width 5 \
  --height 5 \
  --n_mines 3 \
  --device cuda
```

### Playing

```bash
# Watch AI play (with visualization)
python play.py --mode agent --training_run_dir ./training_runs/your_run_directory/

# AI batch evaluation (no visualization)
python play.py --mode batch --num_episodes 100 --training_run_dir ./training_runs/your_run_directory/

# Human play
python play.py --mode human --width 5 --height 5 --n_mines 3
```

## Key Features

- **Custom Gymnasium Environment**: Full Minesweeper game logic with action masking
- **MaskablePPO**: Prevents invalid moves (clicking revealed cells)
- **Custom CNN**: Optimized for grid-based input processing
- **Multiple Play Modes**: AI demonstration, batch evaluation, human play
- **TensorBoard Integration**: Monitor training progress
- **Checkpoint System**: Resume training from any checkpoint

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--total_timesteps` | Total training steps | 1,000,000 |
| `--n_envs` | Parallel environments | 4 |
| `--learning_rate` | Learning rate | 0.0001 |
| `--width/height` | Grid dimensions | 5x5 |
| `--n_mines` | Number of mines | 3 |
| `--device` | Training device | auto |

## Play Modes

| Mode | Description | Usage |
|------|-------------|-------|
| `agent` | AI plays with visualization | `--mode agent` |
| `batch` | AI evaluation without graphics | `--mode batch` |
| `human` | Human player with mouse input | `--mode human` |

## Monitoring

```bash
# View training progress
tensorboard --logdir ./training_runs/
```

## Continue Training

```bash
# Resume from latest checkpoint
python train.py --continue_from ./training_runs/your_run_directory/

# Resume from specific step
python train.py --continue_from ./training_runs/your_run_directory/ --continue_steps 100000
```

## Project Structure

```
minesweeper_rl/
├── src/
│   ├── env/           # Environment and CNN implementations
│   ├── utils/         # Utilities and config
│   └── factories/     # Model and environment factories
├── train.py           # Training script
├── play.py            # Playing/evaluation script
└── README.md          # This file
```