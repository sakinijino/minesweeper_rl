# train.py
import os, datetime, json
import argparse
import re
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# 导入你的环境类和特征提取器
from minesweeper_env import MinesweeperEnv
from custom_cnn import CustomCNN
# 导入 config 来获取默认路径和环境参数
import config
# 导入 checkpoint 相关工具函数
from checkpoint_utils import find_best_checkpoint, load_training_config, find_vecnormalize_stats
# 导入模型工厂
from model_factory import create_model

def parse_int_list(string_list):
    """Helper function to parse comma-separated integers."""
    if not string_list: # Handle empty string case
        return []
    try:
        return [int(x.strip()) for x in string_list.split(',') if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer list format: {string_list}. Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for Minesweeper.")

    # --- PPO Hyperparameters ---
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps per environment per update")
    parser.add_argument("--batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of optimization epochs per update")
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-4, dest='learning_rate', help="Learning rate")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.90, help="Factor for Generalized Advantage Estimation")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--vf_coef", type=float, default=1.0, help="Value function coefficient in the loss calculation")

    # --- Policy Network Architecture (方案 A) ---
    parser.add_argument("--features_dim", type=int, default=128, help="Output dimension of the CNN features extractor")
    parser.add_argument("--pi_layers", type=str, default="64,64",
                        help="Comma-separated layer sizes for the policy network head (e.g., '128,64')")
    parser.add_argument("--vf_layers", type=str, default="256,256",
                        help="Comma-separated layer sizes for the value network head (e.g., '512,256')")
    parser.add_argument("--checkpoint_freq", type=int, default=50000, help="Total steps between checkpoints")

    # --- Paths and Naming (Defaults from config.py) ---
    parser.add_argument("--experiment_base_dir", type=str, default=config.EXPERIMENT_BASE_DIR, help="Base directory for all training run outputs")
    parser.add_argument("--model_prefix", type=str, default=config.MODEL_PREFIX, help="Prefix for saved model files and VecNormalize stats")

    # --- Environment Parameters (Defaults from config.py) ---
    parser.add_argument("--width", type=int, default=config.WIDTH, help="Width of the Minesweeper grid")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="Height of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=config.N_MINES, help="Number of mines in the grid")
    parser.add_argument("--reward_win", type=float, default=config.REWARD_WIN, help="Reward for winning the game")
    parser.add_argument("--reward_lose", type=float, default=config.REWARD_LOSE, help="Penalty for hitting a mine")
    parser.add_argument("--reward_reveal", type=float, default=config.REWARD_REVEAL, help="Reward for revealing a safe cell")
    parser.add_argument("--reward_invalid", type=float, default=config.REWARD_INVALID, help="Penalty for clicking revealed cells")
    parser.add_argument("--max_reward_per_step", type=float, default=config.MAX_REWARD_PER_STEP, help="Maximum reward in one step")

    # --- Continue Training ---
    parser.add_argument("--continue_from", type=str, default=None, help="Directory path containing checkpoints to continue training from")
    parser.add_argument("--continue_steps", type=int, default=None, help="Specific step checkpoint to continue from (optional, uses latest if not specified)")

    # --- Other Settings ---
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for training (auto, cpu, cuda)")
    parser.add_argument("--vec_env_type", type=str, default="subproc", choices=["subproc", "dummy"], help="Type of VecEnv (subproc for parallel, dummy for sequential/debug)")

    args = parser.parse_args()

    # --- Continue Training Logic ---
    continue_training = False
    checkpoint_path = None
    loaded_config = None
    original_run_dir = None
    
    if args.continue_from:
        continue_training = True
        checkpoint_dir = args.continue_from
        
        # Check if continue_from is a run directory (contains models/ subdirectory)
        models_dir = os.path.join(checkpoint_dir, "models")
        if os.path.exists(models_dir):
            original_run_dir = checkpoint_dir
            checkpoint_dir = models_dir
        
        # Find the best checkpoint to continue from
        try:
            checkpoint_path = find_best_checkpoint(checkpoint_dir, args.continue_steps)
            print(f"Found checkpoint to continue from: {checkpoint_path}")
            
            # Try to load the original training configuration
            if original_run_dir:
                config_path = os.path.join(original_run_dir, "training_config.json")
                loaded_config = load_training_config(config_path)
                if loaded_config:
                    print(f"Loaded original training configuration from: {config_path}")
                    # Update args with loaded config, but allow command line args to override
                    for key, value in loaded_config.items():
                        if key in vars(args) and getattr(args, key) == parser.get_default(key):
                            # Only update if current value is the default (not explicitly set)
                            setattr(args, key, value)
                    print("Updated training configuration with loaded values (command line args take precedence)")
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)

    print("--- Training Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-----------------------------")

    # --- 构建本次运行的专属目录 ---
    if continue_training and original_run_dir:
        # For continue training, create a new timestamped directory based on original
        # Use normpath to handle trailing slashes correctly
        original_run_name = os.path.basename(os.path.normpath(original_run_dir))
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{original_run_name}_continue_{timestamp}"
        run_dir = os.path.join(args.experiment_base_dir, run_name)
        print(f"Continuing training in new directory: {run_dir}")
        print(f"Original run directory: {original_run_dir}")
        print(f"Extracted original run name: {original_run_name}")
    else:
        # Normal training - create new directory
        run_name_parts = [
            args.model_prefix,
            f"{args.width}x{args.height}x{args.n_mines}",
        ]

        if args.seed is not None:
            run_name_parts.append(f"seed{args.seed}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name_parts.append(timestamp)

        run_name = "_".join(run_name_parts)
        run_dir = os.path.join(args.experiment_base_dir, run_name)

    specific_log_dir = os.path.join(run_dir, "logs")
    specific_model_dir = os.path.join(run_dir, "models")
    config_save_path = os.path.join(run_dir, "training_config.json")

    # --- Ensure directories exist ---
    os.makedirs(specific_log_dir, exist_ok=True)
    os.makedirs(specific_model_dir, exist_ok=True)

    # --- Construct paths ---
    final_model_path = os.path.join(specific_model_dir, "final_model.zip")
    stats_path = os.path.join(specific_model_dir, "final_stats_vecnormalize.pkl")

    # --- 保存配置 ---
    config_to_save = {}
    # 将 argparse 的 Namespace 对象转换为字典
    for key, value in vars(args).items():
        # argparse 通常存储的是可序列化的基本类型
        config_to_save[key] = value
    
    try:
        with open(config_save_path, 'w') as f:
            json.dump(config_to_save, f, indent=4, sort_keys=True)
        print(f"Training configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

    # --- Parse Network Architecture ---
    try:
        pi_layer_sizes = parse_int_list(args.pi_layers)
        vf_layer_sizes = parse_int_list(args.vf_layers)
        print(f"Parsed policy layers: {pi_layer_sizes}")
        print(f"Parsed value layers: {vf_layer_sizes}")
    except argparse.ArgumentTypeError as e:
        print(f"Error parsing network layers: {e}")
        exit(1)

    # --- Create Environment Function ---
    def create_env():
        """Creates an instance of the Minesweeper environment."""
        env = MinesweeperEnv(
            width=args.width,
            height=args.height,
            n_mines=args.n_mines,
            reward_win=args.reward_win,
            reward_lose=args.reward_lose,
            reward_reveal=args.reward_reveal,
            reward_invalid=args.reward_invalid,
            max_reward_per_step=args.max_reward_per_step,
            render_mode=None # No rendering during training
        )
        return env

    # --- Create Vectorized Environment ---
    vec_env_cls = SubprocVecEnv if args.vec_env_type == "subproc" and args.n_envs > 1 else DummyVecEnv
    print(f"Using VecEnv type: {vec_env_cls.__name__}")

    train_env = make_vec_env(
        create_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls
    )
    # Normalize rewards, but not observations (as observations are already normalized in env)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_obs=10., gamma=args.gamma)
    print(f"Environment VecNormalize stats will be saved to: {stats_path}")


    # --- Checkpoint Callback ---
    # Save a checkpoint every N steps, where N is roughly checkpoint_freq total steps
    save_freq_per_env = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=specific_model_dir,
        save_replay_buffer=True, # Save replay buffer for off-policy algos (doesn't hurt for PPO)
        save_vecnormalize=True # Save VecNormalize statistics automatically
    )
    print(f"Checkpoints will be saved every {save_freq_per_env * args.n_envs} total steps.")


    # --- Create or Load MaskablePPO Model ---
    # Prepare model configuration
    model_config = {
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'learning_rate': args.learning_rate,
        'ent_coef': args.ent_coef,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'vf_coef': args.vf_coef,
        'device': args.device,
        'seed': args.seed
    }
    
    # Determine checkpoint and stats paths for continue training
    checkpoint_to_load = checkpoint_path if continue_training else None
    vecnormalize_stats_path = None
    
    if continue_training and checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        vecnormalize_stats_path = find_vecnormalize_stats(checkpoint_path)
        if vecnormalize_stats_path:
            print(f"Found VecNormalize stats: {vecnormalize_stats_path}")
        else:
            print("Warning: Could not find corresponding VecNormalize stats file")
    else:
        print("Creating new model...")
    
    # Create model using factory
    try:
        model, train_env = create_model(
            env=train_env,
            checkpoint_path=checkpoint_to_load,
            vecnormalize_stats_path=vecnormalize_stats_path,
            tensorboard_log=specific_log_dir,
            features_dim=args.features_dim,
            pi_layers=pi_layer_sizes,
            vf_layers=vf_layer_sizes,
            **model_config
        )
        print(f"Model created/loaded successfully on device: {model.device}")
        if continue_training:
            print(f"TensorBoard logging set to: {specific_log_dir}")
    except Exception as e:
        print(f"Error creating model: {e}")
        exit(1)

    # --- Training ---
    if continue_training:
        print("Continuing training from checkpoint...")
        reset_timesteps = False  # Don't reset timesteps when continuing
    else:
        print("Starting training from scratch...")
        reset_timesteps = True   # Reset timesteps for new training
        
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # --- Save Final Model and Environment Stats ---
        model.save(final_model_path)
        print(f"Training finished or interrupted. Final model saved to: {final_model_path}")

        # VecNormalize stats are saved by the callback, but save again explicitly is fine
        # to ensure the latest stats associated with the final model are saved.
        train_env.save(stats_path)
        print(f"Final environment VecNormalize stats saved to: {stats_path}")

        # --- Close Environment ---
        train_env.close()
        print("Environment closed.")