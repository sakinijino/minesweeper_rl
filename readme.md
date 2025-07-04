# Minesweeper RL Agent using MaskablePPO

This project trains a Reinforcement Learning agent to play the classic game of Minesweeper using a custom Gymnasium environment and Stable Baselines3 with `sb3_contrib`. The agent uses MaskablePPO, which handles invalid action masking (preventing clicks on already revealed cells), and a custom Convolutional Neural Network (CNN) to process the game board state.

## Features

* Custom Gymnasium environment for Minesweeper (`minesweeper_env.py`).
* Custom CNN feature extractor tailored for grid-based input (`custom_cnn.py`).
* Training script (`train.py`) using MaskablePPO from `sb3_contrib`.
* Hyperparameter configuration via command-line arguments (with defaults from `config.py`).
* Evaluation and playing script (`play.py`) with multiple modes:
    * Watch the trained agent play.
    * Play the game yourself.
    * Run the agent in batch mode for performance statistics (win rate).
* Integration with TensorBoard for monitoring training progress.
* Designed for easy use on platforms like Google Colab for GPU-accelerated training.

## File Structure

* `minesweeper_env.py`: Defines the `MinesweeperEnv` class, compliant with Gymnasium API. Handles game logic, state representation, rewards, and rendering.
* `custom_cnn.py`: Defines the `CustomCNN` class, inheriting from Stable Baselines3's `BaseFeaturesExtractor`.
* `train.py`: Script for training the MaskablePPO agent. Handles environment creation, model setup, training loop, callbacks, and saving models/statistics. Accepts command-line arguments for configuration.
* `play.py`: Script for interacting with the environment or evaluating a trained agent. Supports human play, agent play (interactive), and agent batch evaluation. Accepts command-line arguments.
* `config.py`: Contains default configuration values (e.g., default environment size, default save directories, default model prefix). These defaults can be overridden by command-line arguments in `train.py` and `play.py`.
* `README.md`: This file.

## Setup and Installation

1.  **Prerequisites:**
    * Python 3.8+

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    # source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install gymnasium pygame numpy torch "stable-baselines3[extra]" sb3_contrib
    ```

## Configuration

Default settings (environment size, save paths, model prefix) are defined in `config.py`.

However, the **recommended way to configure training and playing runs** is via **command-line arguments** provided to `train.py` and `play.py`. This makes experiments more reproducible and manageable, especially when using platforms like Google Colab. See the usage examples below.

## Training the Agent (`train.py`)

The `train.py` script trains the MaskablePPO agent.

**Basic Training (using defaults):**
```bash
python train.py
```

**Continue Training from Checkpoint:**
```bash
# Continue from the latest checkpoint in a training run directory
python train.py --continue_from ./training_runs/ppo_run_5x5x3_20241215-143000/

# Continue from a specific step checkpoint
python train.py --continue_from ./training_runs/ppo_run_5x5x3_20241215-143000/ --continue_steps 100000

# Continue training with different hyperparameters (will override original config)
python train.py --continue_from ./training_runs/ppo_run_5x5x3_20241215-143000/ --learning_rate 0.0005 --total_timesteps 2000000
```

**Advanced Training:**

Make sure you are in the project directory
```bash
python train.py \
  --total_timesteps 1_000_000 \
  --n_envs 4 \
  --n_steps 1024 \
  --batch_size 128 \
  --n_epochs 8 \
  --learning_rate 0.0001 \
  --ent_coef 0.01 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_range 0.2 \
  --vf_coef 0.5 \
  --features_dim 128 \
  --pi_layers "64,64" \
  --vf_layers "64,64" \
  --checkpoint_freq 50000 \
  --experiment_base_dir "./training_runs" \
  --model_prefix "ppo_run" \
  --width 5 \
  --height 5 \
  --n_mines 3 \
  --reward_win 0.2 \
  --reward_lose -0.05 \
  --reward_reveal 0.1 \
  --reward_invalid -0.1 \
  --seed 42 \
  --device "cuda" \
  --vec_env_type "subproc"

# --total_timesteps: Total steps for training (Default: 1000000)
# --n_envs: Number of parallel environments (Default: 4)
# --n_steps: Steps per env per update (Default: 1024)
# --batch_size: Minibatch size (Default: 128)
# --n_epochs: Optimization epochs per update (Default: 10)
# --learning_rate / --lr: Learning rate (Default: 0.0001)
# --ent_coef: Entropy coefficient (Default: 0.01)
# --gamma: Discount factor (Default: 0.99)
# --gae_lambda: GAE lambda factor (Default: 0.90)
# --clip_range: PPO clipping parameter (Default: 0.2)
# --vf_coef: Value function loss coefficient (Default: 1.0)
# --features_dim: CNN output features dimension (Default: 128)
# --pi_layers: Policy head layers, comma-separated (Default: "64,64")
# --vf_layers: Value head layers, comma-separated (Default: "256,256")
# --checkpoint_freq: Total steps between checkpoints (Default: 50000)
# --continue_from: Directory path containing checkpoints to continue training from (Default: None)
# --continue_steps: Specific step checkpoint to continue from - uses latest if not specified (Default: None)
# --experiment_base_dir: Base directory for all training run outputs (Default from config.py)
# --model_prefix: Prefix for saved files (Default from config.py) - IMPORTANT for identifying runs
# --width: Environment grid width (Default from config.py)
# --height: Environment grid height (Default from config.py)
# --n_mines: Number of mines (Default from config.py)
# --reward_win: Reward for winning the game (Default from config.py)
# --reward_lose: Penalty for hitting a mine (Default from config.py)
# --reward_reveal: Reward for revealing a safe cell (Default from config.py)
# --reward_invalid: Penalty for clicking revealed cells (Default from config.py)
# --max_reward_per_step: Maximum reward in one step (Default from config.py)
# --seed: Random seed for reproducibility (Default: None)
# --device: Training device ('auto', 'cpu', 'cuda') (Default: 'auto') - Use 'cuda' on Colab GPU
# --vec_env_type: VecEnv type ('subproc', 'dummy') (Default: 'subproc')
```

**Monitoring with TensorBoard:**
During or after training, you can monitor progress using TensorBoard:

```bash
# Point to the directory specified by --log_dir (or the default ./logs/)
tensorboard --logdir ./training_runs/

# for Colab
# %load_ext tensorboard
# %tensorboard --logdir ./training_runs/
```

## Playing and Evaluating (play.py)

The play.py script allows you to interact with the environment or test a trained agent.

**Watch Trained Agent Play (Interactive)**
(Note: Requires a graphical display. May require workarounds on headless systems like standard Colab.)

Specify the parameters matching the training run of the model you want to load.
```bash
# Use latest checkpoint (default behavior)
python play.py \
  --training_run_dir ./training_runs/ppo_run_5x5x3_seed42_20250515-085710/ \
  --width 5 \
  --height 5 \
  --n_mines 3 \
  --delay 0.2 \
  --seed 42 \
  --device "cuda"

# Use specific checkpoint step
python play.py \
  --training_run_dir ./training_runs/ppo_run_5x5x3_seed42_20250515-085710/ \
  --checkpoint_steps 100000 \
  --width 5 \
  --height 5 \
  --n_mines 3 \
  --delay 0.2 \
  --seed 42 \
  --device "cuda"

# --training_run_dir: Directory to training dir of save models and stats
# --checkpoint_steps: Specific checkpoint step to load (optional, uses latest if not specified)
```

**Agent Batch Evaluation (No Graphics)**
Run the agent for a specified number of episodes and calculate the win rate. Ideal for headless environments like Colab.
```bash
# Evaluate latest checkpoint
python play.py \
  --batch \
  --num_episodes 200 \
  --training_run_dir ./training_runs/ppo_run_5x5x3_seed42_20250515-085710/ \
  --width 5 \
  --height 5 \
  --n_mines 3 \
  --seed 42 \
  --device "cuda"

# Evaluate specific checkpoint step
python play.py \
  --batch \
  --num_episodes 200 \
  --training_run_dir ./training_runs/ppo_run_5x5x3_seed42_20250515-085710/ \
  --checkpoint_steps 150000 \
  --width 5 \
  --height 5 \
  --n_mines 3 \
  --seed 42 \
  --device "cuda"

# --training_run_dir: Directory to training dir of save models and stats
# --checkpoint_steps: Specific checkpoint step to load (optional, uses latest if not specified)
```

**Play Manually (Interactive)**
(Note: Requires a graphical display.)
```bash
python play.py \
  --human \
  --width 5 \
  --height 5 \
  --n_mines 3
```