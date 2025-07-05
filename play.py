# play_new.py - Refactored play script with new configuration system
import os
import gymnasium as gym
import pygame
import time
import argparse
import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# Import environment classes
from src.env.minesweeper_env import MinesweeperEnv

# Legacy config import removed - now using new configuration system
from src.utils.checkpoint_utils import find_best_checkpoint, load_training_config, find_vecnormalize_stats

# Import model and environment factories
from src.factories.model_factory import create_inference_model
from src.factories.environment_factory import create_inference_environment

# Import new configuration system
from src.config.config_manager import ConfigManager
from src.config.config_schemas import PlayConfig, EnvironmentConfig


def setup_argument_parser():
    """
    Setup and return configured argument parser with new configuration system.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Play Minesweeper in different modes with new config system.")

    # Core play arguments
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["agent", "batch", "human"], 
                        help="Play mode: agent (AI with visualization), batch (AI without visualization), human (human player)")
    
    # Model and configuration loading
    parser.add_argument("--model_dir", type=str, default=None, 
                        help="Specific model directory (complete path with timestamp)")
    parser.add_argument("--training_run_dir", type=str, default=None, 
                        help="Experiment directory to find latest training run")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to configuration file (JSON format)")
    parser.add_argument("--checkpoint_steps", type=int, default=None, 
                        help="Specific checkpoint step to load (uses latest if not specified)")

    # Play execution parameters
    parser.add_argument("--num_episodes", type=int, default=100, 
                        help="Number of episodes to run in batch mode (default: 100)")
    parser.add_argument("--delay", type=float, default=0.1, 
                        help="Delay (in seconds) between agent moves in interactive mode (default: 0.1)")
    parser.add_argument("--device", type=str, default=None, 
                        choices=["auto", "cpu", "cuda"], 
                        help="Device to use for loading the model")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for environment reset")

    # Environment parameters (can override loaded config)
    parser.add_argument("--width", type=int, default=None, 
                        help="Width of the Minesweeper grid")
    parser.add_argument("--height", type=int, default=None, 
                        help="Height of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=None, 
                        help="Number of mines in the grid")
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

    return parser


def load_and_setup_play_config(args):
    """
    Load and setup play configuration based on arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple[ConfigManager, PlayConfig, str, str]: (config_manager, play_config, model_path, stats_path)
    """
    config_manager = None
    model_path = None
    stats_path = None
    
    # Validate mutually exclusive arguments
    if args.model_dir and args.training_run_dir:
        print("Error: Cannot specify both --model_dir and --training_run_dir")
        exit(1)
    
    # Load configuration from specific model directory
    if args.model_dir:
        try:
            # Load training configuration
            config_manager = ConfigManager()
            config_manager.load_from_training_run(args.model_dir)
            print(f"Loaded configuration from model directory: {args.model_dir}")
            
            # Find model and stats paths
            model_path, stats_path = resolve_model_paths_from_run_dir(args.model_dir, args.checkpoint_steps)
            
        except FileNotFoundError:
            print(f"Warning: No training config found in {args.model_dir}")
            # Still try to find model paths
            try:
                model_path, stats_path = resolve_model_paths_from_run_dir(args.model_dir, args.checkpoint_steps)
            except Exception as e:
                print(f"Error: Could not find model files: {e}")
                exit(1)
    
    # Load configuration from experiment directory (find latest)
    elif args.training_run_dir:
        try:
            # Find latest experiment directory
            latest_experiment_dir = find_latest_experiment_dir(args.training_run_dir)
            print(f"Found latest experiment: {latest_experiment_dir}")
            
            # Load training configuration
            config_manager = ConfigManager()
            config_manager.load_from_training_run(latest_experiment_dir)
            print(f"Loaded configuration from latest experiment: {latest_experiment_dir}")
            
            # Find model and stats paths
            model_path, stats_path = resolve_model_paths_from_run_dir(latest_experiment_dir, args.checkpoint_steps)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)
        except Exception as e:
            print(f"Error: Could not find model files: {e}")
            exit(1)
    
    # Load configuration file if specified
    elif args.config:
        try:
            config_manager = ConfigManager(args.config)
            print(f"Loaded configuration from file: {args.config}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            exit(1)
    
    # Check if we have a configuration manager
    if config_manager is None:
        print("Error: No configuration source provided. Use --config, --model_dir, or --training_run_dir")
        exit(1)
    
    # Load command line arguments if provided
    if args:
        config_manager.load_from_args(args)
    
    # Build configuration
    try:
        config = config_manager.build_config()
    except Exception as e:
        print(f"Error building configuration: {e}")
        exit(1)
    
    # Create play configuration directly from args and training config
    # PlayConfig is no longer part of TrainingConfig
    play_config = PlayConfig(
        mode=args.mode,
        num_episodes=args.num_episodes,
        delay=args.delay,
        checkpoint_steps=args.checkpoint_steps,
        environment_config=config.environment_config
    )
    
    return config_manager, play_config, model_path, stats_path


def find_latest_experiment_dir(experiment_base_dir):
    """
    Find the latest experiment directory based on timestamp.
    
    Args:
        experiment_base_dir: Base directory containing experiment runs
        
    Returns:
        str: Path to the latest experiment directory
        
    Raises:
        FileNotFoundError: If no experiment directories found
    """
    if not os.path.exists(experiment_base_dir):
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_base_dir}")
    
    # Get all directories that look like experiment runs
    experiment_dirs = []
    for item in os.listdir(experiment_base_dir):
        item_path = os.path.join(experiment_base_dir, item)
        if os.path.isdir(item_path) and '_' in item:
            # Check if it contains a timestamp pattern (8 digits + 6 digits)
            if any(part.isdigit() and len(part) == 14 for part in item.split('_')):
                experiment_dirs.append(item)
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in {experiment_base_dir}")
    
    def get_latest_timestamp(dir_name):
        """Extract the latest timestamp from directory name."""
        parts = dir_name.split('_')
        timestamps = [part for part in parts if part.isdigit() and len(part) == 14]
        return max(timestamps) if timestamps else '0'
    
    # Sort by latest timestamp (latest first)
    experiment_dirs.sort(key=get_latest_timestamp, reverse=True)
    
    latest_dir = os.path.join(experiment_base_dir, experiment_dirs[0])
    return latest_dir


def resolve_model_paths_from_run_dir(run_dir, checkpoint_steps=None):
    """
    Resolve model and stats paths from training run directory.
    
    Args:
        run_dir: Training run directory
        checkpoint_steps: Specific checkpoint steps (optional)
        
    Returns:
        Tuple[str, str]: (model_path, stats_path)
    """
    try:
        # Check if run_dir is a training run directory (contains models/ subdirectory)
        models_dir = os.path.join(run_dir, "models")
        if os.path.exists(models_dir):
            checkpoint_dir = models_dir
        else:
            # Assume run_dir is already the models directory
            checkpoint_dir = run_dir
        
        # Find the best checkpoint
        best_checkpoint = find_best_checkpoint(checkpoint_dir, checkpoint_steps)
        print(f"Selected checkpoint: {os.path.basename(best_checkpoint)}")
        
        # Find corresponding VecNormalize stats file
        stats_path = find_vecnormalize_stats(best_checkpoint)
        if stats_path:
            print(f"Found VecNormalize stats: {os.path.basename(stats_path)}")
        else:
            print("Warning: No VecNormalize stats file found, using unnormalized environment")
        
        return best_checkpoint, stats_path
        
    except Exception as e:
        raise Exception(f"Could not resolve model paths from {run_dir}: {e}")




def load_model_and_environment(config_manager, env, model_path, stats_path):
    """
    Load model and environment for inference.
    
    Args:
        config_manager: Configuration manager instance
        env: Environment instance
        model_path: Path to model file
        stats_path: Path to VecNormalize stats file
        
    Returns:
        Tuple[model, updated_env] or (None, env) if no model needed
    """
    if not model_path or not os.path.exists(model_path):
        if model_path:
            print(f"Error: Model path not found: {model_path}")
        return None, env

    print(f"Loading model: {model_path}")
    
    # Get device from play config or training config
    device = "cpu"  # Default device for play
    config = config_manager.get_config()
    if hasattr(config, 'training_execution') and config.training_execution.device:
        device = config.training_execution.device
    
    try:
        model, env = create_inference_model(
            env=env,
            config_manager=config_manager,
            checkpoint_path=model_path,
            vecnormalize_stats_path=stats_path
        )
        print(f"Model loaded on device: {model.device}")
        
        # Set VecNormalize to evaluation mode
        if hasattr(env, 'training'):
            env.training = False
            env.norm_reward = False
            print("VecNormalize set to evaluation mode")
            
        return model, env
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, env


def setup_random_seed(config_manager, env, play_config=None):
    """Setup random seed from configuration."""
    seed = None
    config = config_manager.get_config()
    if play_config and hasattr(play_config, 'seed'):
        seed = play_config.seed
    elif config.training_execution.seed:
        seed = config.training_execution.seed
    
    if seed is not None:
        set_random_seed(seed)
        env.seed(seed)
        print(f"Using random seed: {seed}")


def print_final_statistics(total_games, wins, player_type):
    """Print final statistics."""
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    print(f"\n--- Final {player_type} Mode Statistics ---")
    print(f"Total Games Played: {total_games}")
    print(f"{player_type} Wins: {wins}")
    print(f"{player_type} Win Rate: {win_rate:.2f}%")


def print_episode_result(episode, total_episodes, episode_steps, episode_reward, won_episode):
    """Print single episode result."""
    status = "Win" if won_episode else "Lose"
    print(f"Episode {episode + 1}/{total_episodes} finished - "
          f"Steps: {episode_steps}, Reward: {episode_reward:.2f}, Result: {status}")


def run_batch_mode(config_manager, play_config, model_path, stats_path):
    """
    Batch mode: Agent runs multiple games without visualization.
    
    Args:
        config_manager: Configuration manager instance
        play_config: Play configuration
        model_path: Path to model file
        stats_path: Path to VecNormalize stats file
    """
    print(f"--- Running Batch Mode ({play_config.num_episodes} episodes) ---")

    # Create environment (no rendering)
    env, _ = create_inference_environment(
        config_manager=config_manager,
        mode='batch',
        vecnormalize_stats_path=stats_path
    )
    setup_random_seed(config_manager, env, play_config)
    
    # Load model
    model, env = load_model_and_environment(config_manager, env, model_path, stats_path)
    if model is None:
        env.close()
        return

    # Batch game loop
    total_games = 0
    wins = 0

    for episode in range(play_config.num_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        won_episode = False

        while not terminated and not truncated:
            # Get action masks and predict action
            action_masks = env.env_method("action_masks")[0]
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

            # Execute action
            obs, reward, terminated_arr, info_arr = env.step(action)
            terminated = terminated_arr[0]
            actual_info = info_arr[0]
            won_episode = actual_info.get('is_success', False)
            truncated = actual_info.get('TimeLimit.truncated', False)
            reward = reward[0]

            episode_reward += reward
            episode_steps += 1

        # Statistics
        total_games += 1
        if won_episode:
            wins += 1

        print_episode_result(episode, play_config.num_episodes, episode_steps, episode_reward, won_episode)

    # Print final statistics
    print_final_statistics(total_games, wins, "Agent")
    env.close()


def run_human_mode(config_manager, play_config, stats_path):
    """
    Human mode: Human player plays through mouse clicks.
    
    Args:
        config_manager: Configuration manager instance
        play_config: Play configuration
        stats_path: Path to VecNormalize stats file
    """
    print("--- Running Human Mode ---")

    # Create environment (with rendering)
    env, env_instance = create_inference_environment(
        config_manager=config_manager,
        mode='human',
        vecnormalize_stats_path=stats_path
    )
    setup_random_seed(config_manager, env, play_config)
    
    # Human mode doesn't need a model, but VecNormalize stats are handled in environment factory

    # Game statistics
    total_games = 0
    player_wins = 0

    try:
        obs = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        current_game_won = False

        running = True
        while running:
            action = None

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Human click input
                    x, y = event.pos
                    col = x // env_instance.cell_size
                    row = y // env_instance.cell_size
                    
                    if (0 <= row < env_instance.height and 
                        0 <= col < env_instance.width and 
                        not env_instance.revealed[row, col]):
                        action = [row * env_instance.width + col]
                    else:
                        print("Cell already revealed or out of bounds.")

            if not running:
                break

            # Execute action (if any)
            if action is not None:
                obs, reward, terminated_arr, info_arr = env.step(action)
                terminated = terminated_arr[0]
                actual_info = info_arr[0]
                current_game_won = actual_info.get('is_success', False)
                truncated = actual_info.get('TimeLimit.truncated', False)
                reward = reward[0]

                total_reward += reward
                step_count += 1
                
                print(f"Step: {step_count}, Action: {action[0]}, Reward: {reward:.2f}, Done: {terminated}")

                # Check game end
                if terminated or truncated:
                    total_games += 1
                    if current_game_won:
                        player_wins += 1
                        print("Game Over - YOU WIN!")
                    else:
                        if terminated:
                            print("Game Over - YOU LOSE!")
                        else:
                            print("Game Over!")

                    print(f"Final Reward: {total_reward:.2f}, Steps: {step_count}")
                    current_win_rate = (player_wins / total_games * 100) if total_games > 0 else 0
                    print(f"--- Stats so far --- Games: {total_games}, Wins: {player_wins}, Win Rate: {current_win_rate:.2f}% ---")

                    time.sleep(2)  # Pause to display results
                    print("Resetting environment...")
                    obs = env.reset()
                    total_reward = 0
                    step_count = 0
                    terminated = False
                    truncated = False
                    current_game_won = False
            else:
                # No action, still need to render to keep window updated
                env_instance._render_frame()

    except KeyboardInterrupt:
        print("\nHuman mode interrupted by user.")
    finally:
        print_final_statistics(total_games, player_wins, "Human")
        env.close()
        print("Environment closed. Game exited.")


def run_agent_mode(config_manager, play_config, model_path, stats_path):
    """
    Agent demonstration mode: Agent plays with visualization.
    
    Args:
        config_manager: Configuration manager instance
        play_config: Play configuration
        model_path: Path to model file
        stats_path: Path to VecNormalize stats file
    """
    print("--- Running Agent Mode ---")

    # Create environment (with rendering)
    env, env_instance = create_inference_environment(
        config_manager=config_manager,
        mode='agent',
        vecnormalize_stats_path=stats_path
    )
    setup_random_seed(config_manager, env, play_config)
    
    # Load model
    model, env = load_model_and_environment(config_manager, env, model_path, stats_path)
    if model is None:
        env.close()
        return

    # Game statistics
    total_games = 0
    agent_wins = 0

    try:
        obs = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        current_game_won = False

        running = True
        while running:
            # Handle Pygame events (mainly quit events)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("Quitting agent mode...")
                        running = False
                        break

            if not running:
                break

            # Agent decision
            action_masks = env.env_method("action_masks")[0]
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            print(f"Agent action: {action}")
            time.sleep(play_config.delay)  # Delay for observation

            # Execute action
            obs, reward, terminated_arr, info_arr = env.step(action)
            terminated = terminated_arr[0]
            actual_info = info_arr[0]
            current_game_won = actual_info.get('is_success', False)
            truncated = actual_info.get('TimeLimit.truncated', False)
            reward = reward[0]

            total_reward += reward
            step_count += 1
            
            print(f"Step: {step_count}, Action: {action[0]}, Reward: {reward:.2f}, Done: {terminated}")

            # Check game end
            if terminated or truncated:
                total_games += 1
                if current_game_won:
                    agent_wins += 1
                    print("Game Over - AGENT WINS!")
                else:
                    if terminated:
                        print("Game Over - AGENT LOSES!")
                    else:
                        print("Game Over!")

                print(f"Final Reward: {total_reward:.2f}, Steps: {step_count}")
                current_win_rate = (agent_wins / total_games * 100) if total_games > 0 else 0
                print(f"--- Stats so far --- Games: {total_games}, Wins: {agent_wins}, Win Rate: {current_win_rate:.2f}% ---")

                time.sleep(2)  # Pause to display results
                print("Resetting environment...")
                obs = env.reset()
                total_reward = 0
                step_count = 0
                terminated = False
                truncated = False
                current_game_won = False

    except KeyboardInterrupt:
        print("\nAgent mode interrupted by user.")
    finally:
        print_final_statistics(total_games, agent_wins, "Agent")
        env.close()
        print("Environment closed. Game exited.")


def print_play_configuration(play_config):
    """Print play configuration information."""
    print("--- Play Configuration ---")
    env_config = play_config.environment_config
    
    print(f"Mode: {play_config.mode}")
    print(f"Episodes: {play_config.num_episodes}")
    print(f"Delay: {play_config.delay}")
    print(f"Environment: {env_config.width}x{env_config.height} with {env_config.n_mines} mines")
    print("--------------------------")


def run_selected_mode(config_manager, play_config, model_path, stats_path):
    """
    Run the selected play mode.
    
    Args:
        config_manager: Configuration manager instance
        play_config: Play configuration
        model_path: Path to model file
        stats_path: Path to VecNormalize stats file
    """
    if play_config.mode == "batch":
        run_batch_mode(config_manager, play_config, model_path, stats_path)
    elif play_config.mode == "human":
        run_human_mode(config_manager, play_config, stats_path)
    elif play_config.mode == "agent":
        run_agent_mode(config_manager, play_config, model_path, stats_path)
    else:
        print(f"Error: Unknown mode '{play_config.mode}'")
        exit(1)


def main():
    """Main play function with new configuration system."""
    # 1. Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 2. Load and setup configuration
    config_manager, play_config, model_path, stats_path = load_and_setup_play_config(args)
    
    # 3. Print configuration information
    print_play_configuration(play_config)
    
    # 4. Run selected mode
    run_selected_mode(config_manager, play_config, model_path, stats_path)


if __name__ == "__main__":
    main()