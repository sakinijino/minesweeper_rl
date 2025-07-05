# train_new.py - Refactored training script with new configuration system
import os, datetime, json
import argparse
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Import environment and model factories
from src.env.minesweeper_env import MinesweeperEnv
from src.env.custom_cnn import CustomCNN
from src.factories.model_factory import create_model
from src.factories.environment_factory import create_training_environment

# Legacy config import removed - now using new configuration system
from src.utils.checkpoint_utils import find_best_checkpoint, load_training_config, find_vecnormalize_stats

# Import new configuration system
from src.config.config_manager import ConfigManager
from src.config.config_schemas import TrainingConfig


def setup_argument_parser():
    """
    Setup and return configured argument parser with new configuration system.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for Minesweeper with new config system.")

    # Core training arguments - simplified for most common overrides
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to configuration file (JSON format)")
    parser.add_argument("--total_timesteps", type=int, default=None, 
                        help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=None, 
                        help="Learning rate")
    parser.add_argument("--n_envs", type=int, default=None, 
                        help="Number of parallel environments")

    # Environment parameters - commonly overridden
    parser.add_argument("--width", type=int, default=None, 
                        help="Width of the Minesweeper grid")
    parser.add_argument("--height", type=int, default=None, 
                        help="Height of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=None, 
                        help="Number of mines in the grid")

    # Training execution
    parser.add_argument("--device", type=str, default=None, 
                        choices=["auto", "cpu", "cuda"], 
                        help="Device to use for training")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility")

    # Path and output configuration
    parser.add_argument("--experiment_base_dir", type=str, default=None, 
                        help="Base directory for all training run outputs")
    parser.add_argument("--model_prefix", type=str, default=None, 
                        help="Prefix for saved model files")

    # Continue training
    parser.add_argument("--continue_from", type=str, default=None, 
                        help="Directory path containing checkpoints to continue training from")
    parser.add_argument("--continue_steps", type=int, default=None, 
                        help="Specific step checkpoint to continue from")

    # Advanced parameters (less commonly overridden)
    parser.add_argument("--n_steps", type=int, default=None, 
                        help="Number of steps per environment per update")
    parser.add_argument("--batch_size", type=int, default=None, 
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=None, 
                        help="Number of optimization epochs per update")
    parser.add_argument("--ent_coef", type=float, default=None, 
                        help="Entropy coefficient")
    parser.add_argument("--gamma", type=float, default=None, 
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=None, 
                        help="Factor for Generalized Advantage Estimation")
    parser.add_argument("--clip_range", type=float, default=None, 
                        help="Clipping parameter for PPO")
    parser.add_argument("--vf_coef", type=float, default=None, 
                        help="Value function coefficient")

    # Network architecture
    parser.add_argument("--features_dim", type=int, default=None, 
                        help="Output dimension of the CNN features extractor")
    parser.add_argument("--pi_layers", type=str, default=None,
                        help="Comma-separated layer sizes for policy network (e.g., '128,64')")
    parser.add_argument("--vf_layers", type=str, default=None,
                        help="Comma-separated layer sizes for value network (e.g., '512,256')")

    # Rewards (less commonly overridden)
    parser.add_argument("--reward_win", type=float, default=None, 
                        help="Reward for winning the game")
    parser.add_argument("--reward_lose", type=float, default=None, 
                        help="Penalty for hitting a mine")
    parser.add_argument("--reward_reveal", type=float, default=None, 
                        help="Reward for revealing a safe cell")
    parser.add_argument("--reward_invalid", type=float, default=None, 
                        help="Penalty for clicking revealed cells")
    parser.add_argument("--max_reward_per_step", type=float, default=None, 
                        help="Maximum reward in one step")

    # Other settings
    parser.add_argument("--checkpoint_freq", type=int, default=None, 
                        help="Steps between checkpoints")
    parser.add_argument("--vec_env_type", type=str, default=None, 
                        choices=["subproc", "dummy"], 
                        help="Type of VecEnv")

    return parser


def load_and_setup_config(args):
    """
    Load and setup configuration based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple[ConfigManager, bool, str, dict, str]: 
        (config_manager, continue_training, checkpoint_path, loaded_config, original_run_dir)
    """
    config_manager = ConfigManager()
    
    # Handle continue training first
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
            
            # Load configuration from training run
            try:
                config_manager.load_from_training_run(original_run_dir)
                print(f"Loaded configuration from training run: {original_run_dir}")
            except FileNotFoundError:
                print(f"Warning: No training config found in {original_run_dir}, using defaults")
        
        # Find the best checkpoint to continue from
        try:
            checkpoint_path = find_best_checkpoint(checkpoint_dir, args.continue_steps)
            print(f"Found checkpoint to continue from: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)
    
    # Load configuration file if specified
    elif args.config:
        try:
            config_manager.load_from_file(args.config)
            print(f"Loaded configuration from file: {args.config}")
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {args.config}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            exit(1)
    
    # Update configuration with command line arguments
    config_manager.update_from_args(args)
    
    # Handle pi_layers and vf_layers parsing
    if args.pi_layers:
        try:
            pi_layers = [int(x.strip()) for x in args.pi_layers.split(',') if x.strip()]
            config_manager.config.network_architecture.pi_layers = pi_layers
        except ValueError as e:
            print(f"Error parsing pi_layers: {e}")
            exit(1)
    
    if args.vf_layers:
        try:
            vf_layers = [int(x.strip()) for x in args.vf_layers.split(',') if x.strip()]
            config_manager.config.network_architecture.vf_layers = vf_layers
        except ValueError as e:
            print(f"Error parsing vf_layers: {e}")
            exit(1)
    
    # Validate configuration
    if not config_manager.validate_config():
        print("Error: Invalid configuration parameters")
        exit(1)
    
    return config_manager, continue_training, checkpoint_path, loaded_config, original_run_dir


def create_training_directories(config_manager, continue_training, original_run_dir):
    """
    Create training directories based on configuration.
    
    Args:
        config_manager: Configuration manager instance
        continue_training: Whether this is continue training
        original_run_dir: Original run directory for continue training
        
    Returns:
        Tuple[str, str, str, str, str, str]: Directory paths
    """
    config = config_manager.config
    
    if continue_training and original_run_dir:
        # For continue training, create a new timestamped directory
        original_run_name = os.path.basename(os.path.normpath(original_run_dir))
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_name = f"{original_run_name}_continue_{timestamp}"
        run_dir = os.path.join(config.paths_config.experiment_base_dir, run_name)
        print(f"Continuing training in new directory: {run_dir}")
    else:
        # Normal training - create new directory
        run_name_parts = [
            config.paths_config.model_prefix,
            f"{config.environment_config.width}x{config.environment_config.height}x{config.environment_config.n_mines}",
        ]

        if config.training_execution.seed is not None:
            run_name_parts.append(f"seed{config.training_execution.seed}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_name_parts.append(timestamp)

        run_name = "_".join(run_name_parts)
        run_dir = os.path.join(config.paths_config.experiment_base_dir, run_name)

    # Create subdirectories
    specific_log_dir = os.path.join(run_dir, "logs")
    specific_model_dir = os.path.join(run_dir, "models")
    config_save_path = os.path.join(run_dir, "training_config.json")
    
    # Ensure directories exist
    os.makedirs(specific_log_dir, exist_ok=True)
    os.makedirs(specific_model_dir, exist_ok=True)
    
    # Construct output file paths
    final_model_path = os.path.join(specific_model_dir, "final_model.zip")
    stats_path = os.path.join(specific_model_dir, "final_stats_vecnormalize.pkl")
    
    return run_dir, specific_log_dir, specific_model_dir, config_save_path, final_model_path, stats_path


def save_training_config(config_manager, config_save_path):
    """
    Save training configuration to JSON file.
    
    Args:
        config_manager: Configuration manager instance
        config_save_path: Path to save configuration
    """
    try:
        config_manager.save_to_file(config_save_path)
        print(f"Training configuration saved to: {config_save_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def create_training_environment_from_config(config_manager):
    """
    Create training environment from configuration.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Training environment
    """
    # Use environment factory with ConfigManager directly (no need for args conversion)
    return create_training_environment(config_manager=config_manager)


def create_model_from_config(config_manager, train_env, continue_training, checkpoint_path, specific_log_dir):
    """
    Create model from configuration.
    
    Args:
        config_manager: Configuration manager instance
        train_env: Training environment
        continue_training: Whether continuing training
        checkpoint_path: Path to checkpoint (if continuing)
        specific_log_dir: Tensorboard log directory
        
    Returns:
        Tuple[model, updated_env]
    """
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
    
    # Create model using factory with ConfigManager
    try:
        model, train_env = create_model(
            env=train_env,
            checkpoint_path=checkpoint_to_load,
            vecnormalize_stats_path=vecnormalize_stats_path,
            tensorboard_log=specific_log_dir,
            config_manager=config_manager
        )
        print(f"Model created/loaded successfully on device: {model.device}")
        return model, train_env
    except Exception as e:
        print(f"Error creating model: {e}")
        exit(1)


def setup_checkpoint_callback(config_manager, specific_model_dir):
    """
    Setup checkpoint callback from configuration.
    
    Args:
        config_manager: Configuration manager instance
        specific_model_dir: Model save directory
        
    Returns:
        CheckpointCallback
    """
    config = config_manager.config
    
    # Save a checkpoint every N steps
    save_freq_per_env = max(config.training_execution.checkpoint_freq // config.training_execution.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=specific_model_dir,
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    print(f"Checkpoints will be saved every {save_freq_per_env * config.training_execution.n_envs} total steps.")
    return checkpoint_callback


def run_training_loop(model, config_manager, checkpoint_callback, continue_training, final_model_path, stats_path, train_env):
    """
    Execute training loop.
    
    Args:
        model: Training model
        config_manager: Configuration manager instance
        checkpoint_callback: Checkpoint callback
        continue_training: Whether this is continue training
        final_model_path: Final model save path
        stats_path: Stats save path
        train_env: Training environment
    """
    config = config_manager.config
    
    if continue_training:
        print("Continuing training from checkpoint...")
        reset_timesteps = False
    else:
        print("Starting training from scratch...")
        reset_timesteps = True
        
    try:
        model.learn(
            total_timesteps=config.training_execution.total_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save Final Model and Environment Stats
        model.save(final_model_path)
        print(f"Training finished or interrupted. Final model saved to: {final_model_path}")

        # Save VecNormalize stats
        train_env.save(stats_path)
        print(f"Final environment VecNormalize stats saved to: {stats_path}")

        # Close Environment
        train_env.close()
        print("Environment closed.")


def print_training_configuration(config_manager):
    """Print training configuration information."""
    print("--- Training Configuration ---")
    config_dict = config_manager.get_training_config().__dict__
    for section, section_config in config_dict.items():
        print(f"{section}:")
        if hasattr(section_config, '__dict__'):
            for key, value in section_config.__dict__.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {section_config}")
    print("-----------------------------")


def main():
    """Main training function with new configuration system."""
    # 1. Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 2. Load and setup configuration
    config_manager, continue_training, checkpoint_path, loaded_config, original_run_dir = load_and_setup_config(args)
    
    # 3. Print configuration information
    print_training_configuration(config_manager)
    
    # 4. Create training directories
    run_dir, specific_log_dir, specific_model_dir, config_save_path, final_model_path, stats_path = create_training_directories(
        config_manager, continue_training, original_run_dir
    )
    
    # 5. Save training configuration
    save_training_config(config_manager, config_save_path)
    
    # 6. Create training environment
    train_env = create_training_environment_from_config(config_manager)
    print(f"Environment VecNormalize stats will be saved to: {stats_path}")
    
    # 7. Setup checkpoint callback
    checkpoint_callback = setup_checkpoint_callback(config_manager, specific_model_dir)
    
    # 8. Create training model
    model, train_env = create_model_from_config(
        config_manager, train_env, continue_training, checkpoint_path, specific_log_dir
    )
    
    # 9. Execute training loop
    run_training_loop(
        model, config_manager, checkpoint_callback, continue_training, 
        final_model_path, stats_path, train_env
    )


if __name__ == '__main__':
    main()