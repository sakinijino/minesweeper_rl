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
# Quick start with local configuration (for testing)
python train.py --config configs/local_config.yaml

# Quick start with Colab configuration (for long training)
python train.py --config configs/colab_config.yaml

# Override specific parameters
python train.py --config configs/local_config.yaml \
  --total_timesteps 50000 \
  --learning_rate 0.0005

# Advanced training with custom parameters (showcasing available options)
python train.py --config configs/colab_config.yaml \
  --total_timesteps 2_000_000 \
  --n_envs 8 \
  --learning_rate 0.0003 \
  --batch_size 256 \
  --n_epochs 10 \
  --device cuda \
  --seed 42
```

### Playing

```bash
# Watch AI play with specific model directory
python play.py --mode agent --model_dir ./training_runs/ppo_run_5x5x3_seed42_20250705121812/

# Watch AI play with latest experiment from training_runs/
python play.py --mode agent --training_run_dir ./training_runs/

# AI batch evaluation (no visualization)  
python play.py --mode batch --num_episodes 100 --model_dir ./training_runs/ppo_run_5x5x3_seed42_20250705121812/

# Override environment settings while using saved model
python play.py --mode agent --model_dir ./training_runs/ppo_run_5x5x3_seed42_20250705121812/ \
  --width 10 --height 10 --n_mines 15 --delay 0.5

# Use specific checkpoint step
python play.py --mode batch --model_dir ./training_runs/ppo_run_5x5x3_seed42_20250705121812/ \
  --checkpoint_steps 500000 --num_episodes 50

# Human play with custom environment
python play.py --mode human --width 8 --height 8 --n_mines 12

# Load from config file instead of training run
python play.py --mode agent --config ./configs/my_config.json
```

### Continue Training

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

### Monitoring

```bash
# View training progress
tensorboard --logdir ./training_runs/
```

## Key Features

- **Custom Gymnasium Environment**: Full Minesweeper game logic with action masking
- **MaskablePPO**: Prevents invalid moves (clicking revealed cells)
- **Custom CNN**: Optimized for grid-based input processing
- **Advanced Configuration System**: YAML/JSON-based config with parameter priority system
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

## Model Loading Options

| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--model_dir` | Load from specific model directory | `--model_dir ./training_runs/ppo_run_5x5x3_seed42_20250705121812/` |
| `--training_run_dir` | Load latest model from experiment directory | `--training_run_dir ./training_runs/` |
| `--config` | Load from configuration file | `--config ./configs/my_config.json` |

## Configuration System

The new configuration system uses YAML/JSON files with a parameter priority system:

**Priority (highest to lowest):**
1. Command-line arguments
2. Configuration file parameters  
3. Continue training parameters

```bash
# Use predefined configurations
python train.py --config configs/local_config.yaml    # For local testing
python train.py --config configs/colab_config.yaml    # For Colab training

# Override specific parameters (command-line has highest priority)
python train.py --config configs/local_config.yaml --learning_rate 0.0005

# Continue training (loads config from original run)
python train.py --continue_from ./training_runs/your_run/ --config configs/local_config.yaml
```

**Configuration files support YAML and JSON formats:**
- `configs/local_config.yaml` - Optimized for local testing (small env, fast training)
- `configs/colab_config.yaml` - Optimized for Colab training (GPU, longer training)

**Configuration includes:**
- Model hyperparameters (learning rate, batch size, etc.)
- Network architecture (CNN features, layer sizes)  
- Environment settings (grid size, mines, rewards)
- Training execution parameters (timesteps, checkpoints, device, etc.)
- Paths configuration (experiment directory, model prefix)

## Project Structure

```
minesweeper_rl/
├── configs/               # Configuration files
│   ├── local_config.yaml      # Local testing configuration
│   └── colab_config.yaml      # Colab training configuration
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