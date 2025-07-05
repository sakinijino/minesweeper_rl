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

# Advanced training with custom parameters (showcasing available options)
python train.py \
  --total_timesteps 2_000_000 \
  --n_envs 8 \
  --learning_rate 3e-4 \
  --batch_size 256 \
  --n_epochs 10 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_range 0.2 \
  --ent_coef 0.01 \
  --vf_coef 0.5 \
  --features_dim 256 \
  --pi_layers 128 64 \
  --vf_layers 512 256 128 \
  --width 16 \
  --height 16 \
  --n_mines 40 \
  --reward_win 10.0 \
  --reward_lose -10.0 \
  --reward_reveal 1.0 \
  --reward_invalid -1.0 \
  --device cuda \
  --seed 42 \
  --checkpoint_freq 50000 \
  --vec_env_type subproc
```

### Playing

```bash
# Watch AI play (automatically loads saved config from training run)
python play.py --mode agent --training_run_dir ./training_runs/your_run_directory/

# AI batch evaluation (no visualization)  
python play.py --mode batch --num_episodes 100 --training_run_dir ./training_runs/your_run_directory/

# Override environment settings while using saved model
python play.py --mode agent --training_run_dir ./training_runs/your_run_directory/ \
  --width 10 --height 10 --n_mines 15 --delay 0.5

# Use specific checkpoint step
python play.py --mode batch --training_run_dir ./training_runs/your_run_directory/ \
  --checkpoint_steps 500000 --num_episodes 50

# Human play with custom environment
python play.py --mode human --width 8 --height 8 --n_mines 12

# Load from config file instead of training run
python play.py --mode agent --config ./configs/my_config.json
```

## Key Features

- **Custom Gymnasium Environment**: Full Minesweeper game logic with action masking
- **MaskablePPO**: Prevents invalid moves (clicking revealed cells)
- **Custom CNN**: Optimized for grid-based input processing
- **Advanced Configuration System**: JSON-based config with automatic saving/loading
- **Multiple Play Modes**: AI demonstration, batch evaluation, human play
- **TensorBoard Integration**: Monitor training progress
- **Checkpoint System**: Resume training from any checkpoint
- **Factory Pattern**: Modular environment and model creation

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

## Configuration System

The new configuration system automatically saves all training parameters and allows easy reuse and modification:

```bash
# Training automatically saves config.json to the training run directory
python train.py --width 16 --height 16 --n_mines 40 --learning_rate 3e-4

# Later, resume using saved config (loads all original parameters)
python train.py --continue_from ./training_runs/latest_run/

# Override specific parameters while keeping the rest
python train.py --continue_from ./training_runs/latest_run/ --learning_rate 1e-5

# Save and load custom config files
python train.py --save_config ./my_configs/large_grid.json --width 20 --height 20
python train.py --config ./my_configs/large_grid.json
```

Configuration files are saved in JSON format and include:
- Model hyperparameters (learning rate, batch size, etc.)
- Network architecture (CNN features, layer sizes)  
- Environment settings (grid size, mines, rewards)
- Training execution parameters (timesteps, checkpoints, etc.)

## Continue Training

```bash
# Resume from latest checkpoint (automatically loads saved config)
python train.py --continue_from ./training_runs/your_run_directory/

# Resume from specific step
python train.py --continue_from ./training_runs/your_run_directory/ --continue_steps 100000

# Resume training but override specific hyperparameters
python train.py --continue_from ./training_runs/your_run_directory/ \
  --learning_rate 1e-5 --total_timesteps 3_000_000

# Resume with different environment settings
python train.py --continue_from ./training_runs/your_run_directory/ \
  --width 20 --height 20 --n_mines 60 --n_envs 12
```

## Project Structure

```
minesweeper_rl/
├── src/
│   ├── env/               # Environment and CNN implementations
│   │   ├── minesweeper_env.py  # Custom Gymnasium environment
│   │   └── custom_cnn.py       # CNN feature extractor
│   ├── config/            # Configuration system
│   │   ├── config_manager.py   # Configuration management
│   │   └── config_schemas.py   # Configuration data schemas
│   ├── factories/         # Model and environment factories
│   │   ├── environment_factory.py  # Environment creation
│   │   └── model_factory.py        # Model creation
│   └── utils/             # Utilities and legacy config
│       ├── checkpoint_utils.py     # Checkpoint management
│       └── config.py              # Legacy config (for backward compatibility)
├── tests/                 # Unit tests
│   ├── test_config_manager.py     # Configuration system tests
│   ├── test_config_schemas.py     # Schema validation tests
│   ├── test_environment_factory.py # Environment factory tests
│   └── test_model_factory.py      # Model factory tests
├── train.py              # Training script (new config system)
├── play.py               # Playing/evaluation script (new config system)
├── train_legacy.py       # Legacy training script
├── play_legacy.py        # Legacy playing script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```