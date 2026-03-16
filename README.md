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

### Experiment Workflow (Modal Cloud Training)

```bash
make train CONFIG=experiments/configs/exp_NNN_xxx.yaml   # launch cloud training
make pull [RUN=run_name]                                 # download results
make analyze [RUN=run_name] [EXP_ID=exp_NNN]            # diagnose training curves
make eval [RUN=run_name]                                 # clean win rate (100 episodes)
make play [RUN=run_name]                                 # watch agent (optional)
make compare                                             # cross-run comparison (optional)
make tensorboard                                         # view curves in browser
make list                                                # list runs in Modal Volume
make test                                                # run unit tests
```

After `make eval`, record the win rate manually in `experiments/log.md`.

### Playing

```bash
# Batch evaluation — clean win rate (no visualization)
make eval RUN=mw_ppo_5x5x3_seed42_xxx

# Watch AI play with visualization
make play RUN=mw_ppo_5x5x3_seed42_xxx

# Compare all runs in training_runs/
make compare

# Human play with custom environment
python play.py --mode human --width 8 --height 8 --n_mines 12

# Advanced: use specific checkpoint or override env settings
python play.py --mode batch --model_dir ./training_runs/run_name/ \
  --checkpoint_steps 500000 --num_episodes 50
python play.py --mode compare --model_dirs ./training_runs/m1/ ./training_runs/m2/ --num_episodes 50
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

### Monitoring & Analysis

```bash
make tensorboard                                         # view curves in browser
make analyze [RUN=run_name] [EXP_ID=exp_NNN]            # print metrics summary + save JSON
make test                                                # run unit tests
```

## Key Features

- **Custom Gymnasium Environment**: Full Minesweeper game logic with action masking
- **MaskablePPO**: Prevents invalid moves (clicking revealed cells)
- **Custom CNN**: Optimized for grid-based input processing
- **Advanced Configuration System**: YAML/JSON-based config with parameter priority system
- **Multiple Play Modes**: AI demonstration, batch evaluation, human play, model comparison
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
| `compare` | Compare multiple models performance | `--mode compare` |

## Model Loading Options

| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--model_dir` | Load from specific model directory | `--model_dir ./training_runs/ppo_run_5x5x3_seed42_20250705121812/` |
| `--training_run_dir` | Load latest model from experiment directory | `--training_run_dir ./training_runs/` |
| `--model_dirs` | Multiple model directories (compare mode only) | `--model_dirs ./model1/ ./model2/ ./model3/` |
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
├── configs/               # Base config templates (local/colab, not experiment-specific)
│   ├── local_config.yaml
│   └── colab_config.yaml
├── experiments/           # Experiment tracking
│   ├── configs/               # Per-experiment YAML (exp_NNN_description.yaml)
│   ├── results/               # Auto-generated metrics JSON (git-tracked)
│   ├── log.md                 # Hand-written experiment analysis & conclusions
│   └── ideas.md               # Optimization backlog
├── scripts/               # Workflow tooling
│   ├── analyze.py             # Parse TensorBoard logs → metrics summary + JSON
│   └── pull_run.sh            # Download run from Modal Volume
├── src/
│   ├── env/               # Environment and CNN implementations
│   │   ├── minesweeper_env.py
│   │   └── custom_cnn.py
│   ├── config/            # Configuration system
│   │   ├── config_manager.py
│   │   └── config_schemas.py
│   ├── factories/         # Model and environment factories
│   │   ├── environment_factory.py
│   │   └── model_factory.py
│   └── utils/
│       └── checkpoint_utils.py
├── tests/                 # Unit tests
├── training_runs/         # Local run cache (gitignored, download with make pull)
├── Makefile               # Unified command interface (make train/pull/analyze/eval/…)
├── train.py              # Training entry point
├── train_modal.py        # Modal cloud training wrapper
├── play.py               # Evaluation / play entry point
└── requirements.txt
```