# play.py
import os
import gymnasium as gym
import pygame
import time
import argparse
import numpy as np # Import numpy for stats

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# 导入你的环境类
from minesweeper_env import MinesweeperEnv
# 导入 config 来获取默认值
import config

def run_batch_mode(args):
    """
    Runs the agent in batch mode without rendering for a specified number of episodes.
    Uses configuration from the 'args' object.
    """
    print(f"--- Running Batch Mode ({args.num_episodes} episodes) ---")

    # --- Environment Creation (No Rendering) ---
    def create_env_instance():
        # Set render_mode to None or 'rgb_array' to disable pygame window
        return MinesweeperEnv(
            width=args.width,       # Use arg value
            height=args.height,     # Use arg value
            n_mines=args.n_mines,    # Use arg value
            reward_win=args.reward_win,
            reward_lose=args.reward_lose,
            reward_reveal=args.reward_reveal,
            reward_invalid=args.reward_invalid,
            max_reward_per_step=args.max_reward_per_step,
            render_mode=None,        # No rendering needed for batch mode
        )

    # Use DummyVecEnv for batch mode as well (simpler for single env evaluation)
    env = DummyVecEnv([lambda: create_env_instance()])

    if args.seed is not None:
        set_random_seed(args.seed)
        env.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # --- Load VecNormalize stats IF path is provided and exists ---
    stats_path = args.stats_path # Use arg value
    if stats_path and os.path.exists(stats_path):
         print(f"Loading VecNormalize stats from: {stats_path}")
         env = VecNormalize.load(stats_path, env)
         env.training = False # Set to evaluation mode
         env.norm_reward = False # Don't normalize reward during evaluation
         print("VecNormalize stats loaded.")
    else:
         print(f"VecNormalize stats path not found or not specified: {stats_path}. Using unnormalized environment.")


    # --- Model Loading ---
    model_path = args.model_path # Use arg value
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        env.close()
        return

    print(f"Loading model: {model_path}")
    # Use device specified in args
    model = MaskablePPO.load(model_path, env=env, device=args.device)
    print(f"Model loaded on device: {model.device}")

    # --- Batch Game Loop ---
    total_games = 0
    wins = 0

    for episode in range(args.num_episodes): # Use arg value
        obs = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        won_episode = False

        while not terminated and not truncated:
            # Get action mask
            action_masks = env.env_method("action_masks")[0]

            # Predict action
            action, _states = model.predict(obs,
                                            action_masks=action_masks,
                                            deterministic=True)

            # Step environment
            obs, reward, terminated_arr, info_arr = env.step(action)
            terminated = terminated_arr[0]
            actual_info = info_arr[0] # Access the info dict from the underlying env
            won_episode = actual_info.get('is_win', False) # Check if 'is_win' exists in info
            truncated = actual_info.get('TimeLimit.truncated', False)
            reward = reward[0] # Get reward from the single env
            # obs is already updated

            episode_reward += reward
            episode_steps += 1

            # No rendering or delay in batch mode

        # End of episode
        total_games += 1
        if won_episode:
            wins += 1

        print(f"Episode {episode + 1}/{args.num_episodes} finished - Steps: {episode_steps}, Reward: {episode_reward:.2f}, Win: {won_episode}") # Use arg value

    # --- Statistics ---
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    print("\n--- Batch Mode Statistics ---")
    print(f"Total Games Played: {total_games}")
    print(f"Model Wins: {wins}")
    print(f"Model Win Rate: {win_rate:.2f}%")

    env.close()


def run_interactive_mode(args):
    """
    Runs the game in interactive mode (human or agent) with rendering and statistics.
    Uses configuration from the 'args' object.
    """
    print("--- Running Interactive Mode ---")
    player_type = "Human" if args.human else "Agent"
    print(f"Player: {player_type}")

     # Statistics tracking
    total_games = 0
    player_wins = 0

    # --- Environment Creation (with Rendering) ---
    def create_env_instance():
        # Create with 'human' render mode for interactive play
        return MinesweeperEnv(
            width=args.width,      # Use arg value
            height=args.height,    # Use arg value
            n_mines=args.n_mines,   # Use arg value
            reward_win=args.reward_win,
            reward_lose=args.reward_lose,
            reward_reveal=args.reward_reveal,
            reward_invalid=args.reward_invalid,
            max_reward_per_step=args.max_reward_per_step,
            render_mode='human'
        )

    # Need the raw instance for direct access (e.g., cell_size, mouse clicks)
    env_instance = create_env_instance()
    # Wrap in DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([lambda: env_instance])

    if args.seed is not None:
        set_random_seed(args.seed)
        env.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # --- Load VecNormalize stats IF path is provided and exists ---
    stats_path = args.stats_path # Use arg value
    if stats_path and os.path.exists(stats_path):
         print(f"Loading VecNormalize stats from: {stats_path}")
         env = VecNormalize.load(stats_path, env)
         env.training = False # Set to evaluation mode
         env.norm_reward = False # Don't normalize reward during evaluation
         print("VecNormalize stats loaded.")
    else:
         print(f"VecNormalize stats path not found or not specified: {stats_path}. Using unnormalized environment.")


    # --- Model Loading (only if agent is playing) ---
    model = None
    if not args.human: # If agent is playing
        model_path = args.model_path # Use arg value
        if not model_path or not os.path.exists(model_path):
            print(f"Error: Model path not found: {model_path}")
            env.close()
            return

        print(f"Loading model: {model_path}")
        # Use device specified in args
        model = MaskablePPO.load(model_path, env=env, device=args.device)
        print(f"Model loaded on device: {model.device}")

    # --- Game Loop ---
    try:
        obs = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        current_game_won = False

        running = True
        while running:
            action = None # Action to be taken in this step

            # --- Handle Pygame Events (Needed for window closing, human input, agent quit key) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if args.human and event.type == pygame.MOUSEBUTTONDOWN: # Human input
                    x, y = event.pos
                    col = x // env_instance.cell_size # Use attributes from raw env
                    row = y // env_instance.cell_size
                    if 0 <= row < env_instance.height and 0 <= col < env_instance.width:
                        # Check if cell is already revealed using the raw env's state
                        if not env_instance.revealed[row, col]:
                            action = [row * env_instance.width + col] # Action is index
                        else:
                            print("Cell already revealed.")
                elif not args.human and event.type == pygame.KEYDOWN: # Agent quit key
                     if event.key == pygame.K_q:
                          print("Quitting interactive agent play...")
                          running = False
                          break

            if not running: break

            # --- Determine Action ---
            if not args.human and action is None: # Agent's turn (and no quit event)
                action_masks = env.env_method("action_masks")[0]
                action, _states = model.predict(obs,
                                                action_masks=action_masks,
                                                deterministic=True)
                # Optional: Print agent action
                print(f"Agent action: {action}")
                time.sleep(args.delay) # Use delay arg

            # --- Perform Action (if any determined) ---
            if action is not None:
                 obs, reward, terminated_arr, info_arr = env.step(action) # Step the VecEnv
                 terminated = terminated_arr[0]
                 actual_info = info_arr[0] # Get info from the single underlying env
                 current_game_won = actual_info.get('is_win', False)
                 truncated = actual_info.get('TimeLimit.truncated', False)
                 reward = reward[0] # Get reward from the single env
                 # obs is already updated

                 total_reward += reward
                 step_count += 1
                 # Optional: Print step info
                 print(f"Step: {step_count}, Action: {action[0]}, Reward: {reward:.2f}, Done: {terminated}")


                 # --- Check Game End for Statistics ---
                 if terminated or truncated:
                     total_games += 1
                     if current_game_won:
                         player_wins += 1
                         print("Game Over - YOU WIN!")
                     else:
                          if terminated: # Only print lose if actually terminated (not just truncated)
                               print("Game Over - YOU LOSE!")
                          else:
                               print("Game Over!")

                     print(f"Final Reward: {total_reward:.2f}, Steps: {step_count}")
                     current_win_rate = (player_wins / total_games * 100) if total_games > 0 else 0
                     print(f"--- Stats so far --- Games: {total_games}, Wins: {player_wins}, Win Rate: {current_win_rate:.2f}% ---")

                     time.sleep(2) # Pause before reset
                     print("Resetting environment...")
                     obs = env.reset() # Reset the environment
                     total_reward = 0
                     step_count = 0
                     terminated = False # Reset flags for new game
                     truncated = False
                     current_game_won = False

            else:
                 # If no action was taken (e.g., human didn't click),
                 # we still need to render to keep the window updated.
                 # The environment's step usually handles rendering, but only if called.
                 # Call the underlying env's render method directly.
                 env_instance._render_frame()


    except KeyboardInterrupt:
        print("\nInteractive mode interrupted by user.")
    finally:
        # --- Final Statistics After Exiting ---
        print("\n--- Final Interactive Mode Statistics ---")
        win_rate = (player_wins / total_games * 100) if total_games > 0 else 0
        print(f"Total Games Played: {total_games}")
        print(f"{player_type} Wins: {player_wins}")
        print(f"{player_type} Win Rate: {win_rate:.2f}%")

        env.close()
        print("Environment closed. Game exited.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Minesweeper interactively or run batch agent tests.")

    # --- Mode Selection ---
    parser.add_argument("--human", action="store_true", help="Enable human play mode (interactive).")
    parser.add_argument("--batch", action="store_true", help="Enable batch mode (agent plays without rendering).")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to run in batch mode.")

    # --- Paths and Naming (Defaults from config.py) ---
    parser.add_argument("--model_dir", type=str, default=config.MODEL_DIR, help="Directory to save models and stats")
    parser.add_argument("--model_prefix", type=str, default=config.MODEL_PREFIX, help="Prefix for saved model files and VecNormalize stats")

    # --- Environment Parameters ---
    parser.add_argument("--width", type=int, default=config.WIDTH, help="Width of the Minesweeper grid")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="Height of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=config.N_MINES, help="Number of mines in the grid")
    parser.add_argument("--reward_win", type=float, default=config.REWARD_WIN, help="Reward for winning the game")
    parser.add_argument("--reward_lose", type=float, default=config.REWARD_LOSE, help="Penalty for hitting a mine")
    parser.add_argument("--reward_reveal", type=float, default=config.REWARD_REVEAL, help="Reward for revealing a safe cell")
    parser.add_argument("--reward_invalid", type=float, default=config.REWARD_INVALID, help="Penalty for clicking revealed cells")
    parser.add_argument("--max_reward_per_step", type=float, default=config.MAX_REWARD_PER_STEP, help="Maximum reward in one step")

    # --- Execution Parameters ---
    parser.add_argument("--delay", type=float, default=0.1, help="Delay (in seconds) between agent moves in interactive mode.")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device to use for loading the model (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment reset (optional)")


    args = parser.parse_args()

    print("--- Play Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("--------------------------")

    args.model_path = os.path.join(args.model_dir, f"{args.model_prefix}_{args.width}x{args.height}x{args.n_mines}_final.zip")
    args.stats_path = os.path.join(args.model_dir, f"{args.model_prefix}_{args.width}x{args.height}x{args.n_mines}_vecnormalize.pkl")


    # --- Run Selected Mode ---
    if args.batch:
        if args.human:
            print("Error: Cannot use --human and --batch simultaneously.")
        else:
            run_batch_mode(args) # Pass all parsed args
    else:
        # Default to interactive mode (human or agent)
        run_interactive_mode(args) # Pass all parsed args