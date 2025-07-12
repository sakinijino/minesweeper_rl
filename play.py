import os
import argparse
from stable_baselines3.common.utils import set_random_seed

from src.config.config_manager import ConfigManager
from src.config.config_schemas import PlayConfig

from src.factories.model_factory import create_inference_model
from src.factories.environment_factory import create_inference_environment、

from src.utils.checkpoint_utils import (
    find_best_checkpoint, 
    find_all_experiment_dirs,
    find_latest_experiment_dir,
    resolve_model_paths_from_run_dir
)

def setup_argument_parser():
    """
    Setup and return configured argument parser with new configuration system.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Play Minesweeper in different modes with new config system.")

    # Core play arguments
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["agent", "batch", "human", "compare"], 
                        help="Play mode: agent (AI with visualization), batch (AI without visualization), human (human player), compare (compare multiple models)")
    
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
    
    # Compare mode specific arguments
    parser.add_argument("--model_dirs", type=str, nargs='+', default=None,
                        help="List of model directories to compare (for compare mode)")

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


def print_episode_result(episode, total_episodes, episode_steps, episode_reward, won_episode):
    """Print single episode result."""
    status = "Win" if won_episode else "Lose"
    print(f"Episode {episode + 1}/{total_episodes} finished - "
          f"Steps: {episode_steps}, Reward: {episode_reward:.2f}, Result: {status}")


def run_interactive_game_loop(env, env_instance, action_provider, play_config, mode_name="Interactive"):
    """
    Universal interactive game loop for both human and agent modes.
    
    Args:
        env: Vectorized environment instance  
        env_instance: Raw environment instance for direct access
        action_provider: Function that takes (obs, env_instance) and returns (action, should_quit)
        play_config: Play configuration containing delay settings
        mode_name: Name of the mode for logging
        
    Returns:
        dict: Statistics including total_games, wins
    """
    # Game statistics
    total_games = 0
    player_wins = 0

    try:
        obs = env.reset()
        env_instance.render()  # Render initial state
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        current_game_won = False

        running = True
        while running:
            # Get action from provider (human input, AI model, etc.)
            action, delay, should_quit = action_provider(obs, env_instance)
            
            if should_quit:
                running = False
                break
                
            # Execute action if provided
            if action is not None:
                obs, reward, terminated_arr, info_arr = env.step(action)
                terminated = terminated_arr[0]
                actual_info = info_arr[0]
                current_game_won = actual_info.get('is_success', False)
                truncated = actual_info.get('TimeLimit.truncated', False)
                reward = reward[0]

                total_reward += reward
                step_count += 1
                
                print(f"Step: {step_count}, Action: {action[0] if isinstance(action, list) else action}, Reward: {reward:.2f}, Done: {terminated}")
                
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
                    
                    # Show terminal state during delay
                    env_instance.wait_seconds(play_config.delay)
                    while env_instance.is_waiting():
                        if env_instance.check_quit_key():
                            running = False
                            break
                        # Render terminal state instead of current state
                        env_instance.render(use_last_state=True)
                    
                    if running:
                        print("Resetting environment...")
                        obs = env.reset()
                        env_instance.render()  # Render after reset
                        total_reward = 0
                        step_count = 0
                        terminated = False
                        truncated = False
                        current_game_won = False

                # Render after action
                env_instance.render()
                # Non-blocking delay for human observation
                env_instance.wait_seconds(delay)
                while env_instance.is_waiting():
                    if env_instance.check_quit_key():
                        return None, True
                    env_instance.render()
            else:
                # No action, still render to keep display updated
                env_instance.render()

    except KeyboardInterrupt:
        print(f"\n{mode_name} mode interrupted by user.")
    finally:
        print_final_statistics(total_games, player_wins, mode_name)
        env.close()
        print("Environment closed. Game exited.")
        
    return {'total_games': total_games, 'wins': player_wins}


def run_batch_episodes(env, model, num_episodes, verbose=True):
    """
    Run multiple episodes in batch mode (no rendering, for evaluation).
    
    Args:
        env: Environment instance
        model: Model instance
        num_episodes: Number of episodes to play
        verbose: Whether to print progress for each episode
        
    Returns:
        dict: Statistics including total_games, wins, win_rate, avg_steps, avg_reward
    """
    total_games = 0
    wins = 0
    total_steps = 0
    total_reward = 0
    
    for episode in range(num_episodes):
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
        
        # Update statistics
        total_games += 1
        if won_episode:
            wins += 1
        total_steps += episode_steps
        total_reward += episode_reward
        
        if verbose:
            print_episode_result(episode, num_episodes, episode_steps, episode_reward, won_episode)
    
    # Calculate averages
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    avg_steps = total_steps / total_games if total_games > 0 else 0
    avg_reward = total_reward / total_games if total_games > 0 else 0
    
    return {
        'total_games': total_games,
        'wins': wins,
        'win_rate': win_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward
    }


def print_final_statistics(total_games, wins, player_type):
    """Print final statistics."""
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    print(f"\n--- Final {player_type} Mode Statistics ---")
    print(f"Total Games Played: {total_games}")
    print(f"{player_type} Wins: {wins}")
    print(f"{player_type} Win Rate: {win_rate:.2f}%")


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
    
    # Load model
    model, env = load_model_and_environment(config_manager, env, model_path, stats_path)
    if model is None:
        env.close()
        return
    
    setup_random_seed(config_manager, env, play_config)

    # Play games using the common function
    stats = run_batch_episodes(env, model, play_config.num_episodes, verbose=True)

    # Print final statistics
    print_final_statistics(stats['total_games'], stats['wins'], "Agent")
    env.close()


def run_human_mode(config_manager, play_config, stats_path):
    """
    Human mode: Human player plays through mouse clicks.
    
    Args:
        config_manager: Configuration manager instance
        play_config: Play configuration
        stats_path: Path to VecNormalize stats file
    """
    # Create environment (with rendering)
    env, env_instance = create_inference_environment(
        config_manager=config_manager,
        mode='human',
        vecnormalize_stats_path=stats_path
    )
    
    setup_random_seed(config_manager, env, play_config)
    
    # Define inline action provider for human input
    def get_human_action(obs, env_instance, pause_mode=False):
        """Get action from human mouse clicks."""
        if pause_mode:
            # During pause, only check for quit
            _, quit_flag = env_instance.get_user_action()
            return None, quit_flag
        
        # Normal gameplay - get user action
        user_action, quit_flag = env_instance.get_user_action()
        action = [user_action] if user_action is not None else None
        return action, 0, quit_flag
    
    # Use universal interactive game loop
    return run_interactive_game_loop(env, env_instance, get_human_action, play_config, "Human")


def run_agent_mode(config_manager, play_config, model_path, stats_path):
    """
    Agent demonstration mode: Agent plays with visualization.
    
    Args:
        config_manager: Configuration manager instance
        play_config: Play configuration
        model_path: Path to model file
        stats_path: Path to VecNormalize stats file
    """
    # Create environment (with rendering)
    env, env_instance = create_inference_environment(
        config_manager=config_manager,
        mode='agent',
        vecnormalize_stats_path=stats_path
    )
    
    # Load model
    model, env = load_model_and_environment(config_manager, env, model_path, stats_path)
    if model is None:
        env.close()
        return
    
    setup_random_seed(config_manager, env, play_config)
    
    # Define inline action provider for AI agent (closure has access to env and model)
    def get_agent_action(obs, env_instance, pause_mode=False):
        """Get action from AI agent with visualization delay."""
        if pause_mode:
            # During pause, only check for quit
            return None, env_instance.check_quit_key()
            
        # Check for quit events first
        if env_instance.check_quit_key():
            print("Quitting agent mode...")
            return None, True

        # Agent decision (env and model accessible via closure)
        action_masks = env.env_method("action_masks")[0]
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        print(f"Agent action: {action}")
        
        return action, play_config.delay, False
    
    # Use universal interactive game loop
    return run_interactive_game_loop(env, env_instance, get_agent_action, play_config, "Agent")


def print_play_configuration(play_config):
    """Print play configuration information."""
    print("--- Play Configuration ---")
    env_config = play_config.environment_config
    
    print(f"Mode: {play_config.mode}")
    print(f"Episodes: {play_config.num_episodes}")
    print(f"Delay: {play_config.delay}")
    print(f"Environment: {env_config.width}x{env_config.height} with {env_config.n_mines} mines")
    print("--------------------------")


def run_selected_mode(args):
    """
    Run the selected play mode.
    
    Args:
        args: Command line arguments
    """
    # Load and setup configuration for other modes
    config_manager, play_config, model_path, stats_path = load_and_setup_play_config(args)
    
    # Print configuration information
    print_play_configuration(play_config)
    
    if play_config.mode == "batch":
        run_batch_mode(config_manager, play_config, model_path, stats_path)
    elif play_config.mode == "human":
        run_human_mode(config_manager, play_config, stats_path)
    elif play_config.mode == "agent":
        run_agent_mode(config_manager, play_config, model_path, stats_path)
    elif play_config.mode == "compare":
        # Compare mode will be handled separately in main()
        pass
    else:
        print(f"Error: Unknown mode '{play_config.mode}'")
        exit(1)


def print_comparison_results(results):
    """
    Print formatted comparison results table.
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        print("\nNo results to compare.")
        return
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Sort by win rate (descending)
    results.sort(key=lambda x: x['stats']['win_rate'], reverse=True)
    
    # Print header
    print(f"{'Rank':<6} {'Model':<30} {'Games':<8} {'Wins':<8} {'Win Rate':<12} {'Avg Steps':<12} {'Avg Reward':<12}")
    print("-"*80)
    
    # Print each result
    for i, result in enumerate(results):
        stats = result['stats']
        model_name = result['model_name']
        if len(model_name) > 30:
            model_name = model_name[:27] + "..."
        
        print(f"{i+1:<6} {model_name:<30} {stats['total_games']:<8} {stats['wins']:<8} "
              f"{stats['win_rate']:<12.2f} {stats['avg_steps']:<12.2f} {stats['avg_reward']:<12.2f}")
    
    print("="*80)
    
    # Print best model
    if results:
        best = results[0]
        print(f"\nBest Model: {best['model_name']} with {best['stats']['win_rate']:.2f}% win rate")


def run_compare_mode(args):
    """
    Compare mode: Run multiple models and compare their performance.
    
    Args:
        args: Command line arguments
    """
    print("--- Running Compare Mode ---")
    
    # Collect models to compare
    models_to_compare = []
    
    # Option 1: Scan training directory for all experiments
    if args.training_run_dir:
        print(f"Scanning training directory: {args.training_run_dir}")
        experiment_dirs = find_all_experiment_dirs(args.training_run_dir)
        if not experiment_dirs:
            print(f"Error: No experiment directories found in {args.training_run_dir}")
            exit(1)
        print(f"Found {len(experiment_dirs)} experiments to compare")
        
        # For each experiment, find the best checkpoint
        for exp_dir in experiment_dirs:
            try:
                exp_name = os.path.basename(exp_dir)
                models_dir = os.path.join(exp_dir, "models")
                model_path = find_best_checkpoint(models_dir)
                models_to_compare.append((model_path, exp_name))
            except Exception as e:
                print(f"Warning: Could not find model in {exp_dir}: {e}")
                continue
    
    # Option 2: Use specified model directories
    elif args.model_dirs:
        print(f"Using specified model directories: {len(args.model_dirs)} models")
        for model_dir in args.model_dirs:
            # Find best checkpoint in each directory
            try:
                model_path, _ = resolve_model_paths_from_run_dir(model_dir, args.checkpoint_steps)
                dir_name = os.path.basename(model_dir)
                models_to_compare.append((model_path, dir_name))
            except Exception as e:
                print(f"Warning: Could not find model in {model_dir}: {e}")
                continue
    
    else:
        print("Error: Compare mode requires either --training_run_dir or --model_dirs")
        exit(1)
    
    if not models_to_compare:
        print("Error: No valid models found to compare")
        exit(1)
    
    # Results storage
    results = []
    
    # Run each model
    for i, (model_path, model_name) in enumerate(models_to_compare):
        print(f"\n[{i+1}/{len(models_to_compare)}] Testing model: {model_name}")
        
        try:
            # Temporarily set model_dir for load_and_setup_play_config
            original_model_dir = getattr(args, 'model_dir', None)
            original_training_run_dir = getattr(args, 'training_run_dir', None)
            
            # Determine model directory
            model_dir = os.path.dirname(model_path)
            if "models" in model_dir:
                # Go up one directory to find training config
                training_dir = os.path.dirname(model_dir)
            else:
                training_dir = model_dir
            
            # Set args for load_and_setup_play_config
            args.model_dir = training_dir
            args.training_run_dir = None
            
            # Use existing configuration loading logic
            config_manager, _, loaded_model_path, stats_path = load_and_setup_play_config(args)
            
            # Restore original args
            args.model_dir = original_model_dir
            args.training_run_dir = original_training_run_dir
            
            # Use the model_path we found, not the one from load_and_setup_play_config
            # because we want the specific checkpoint, not necessarily the best one
            
            # Create environment
            env, _ = create_inference_environment(
                config_manager=config_manager,
                mode='batch',
                vecnormalize_stats_path=stats_path
            )
            
            # Load model
            model, env = load_model_and_environment(config_manager, env, model_path, stats_path)

            # Set seed if provided
            if args.seed:
                set_random_seed(args.seed)
                env.seed(args.seed)
            
            if model is None:
                print(f"Error: Could not load model {model_name}")
                env.close()
                continue
            
            # Play games
            print(f"Running {args.num_episodes} episodes...")
            stats = run_batch_episodes(env, model, args.num_episodes, verbose=False)
            
            # Store results
            results.append({
                'model_name': model_name,
                'model_path': model_path,
                'stats': stats
            })
            
            # Print individual results
            print(f"  Win Rate: {stats['win_rate']:.2f}%")
            print(f"  Average Steps: {stats['avg_steps']:.2f}")
            print(f"  Average Reward: {stats['avg_reward']:.2f}")
            
            env.close()
            
        except Exception as e:
            import traceback
            print(f"Error testing model {model_name}: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            continue
    
    # Print comparison table
    print_comparison_results(results)


def main():
    """Main play function with new configuration system."""
    # 1. Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if args.mode == "compare":
        run_compare_mode(args)
    else:
        run_selected_mode(args)


if __name__ == "__main__":
    main()