# train.py
import os
import argparse
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
    parser.add_argument("--log_dir", type=str, default=config.LOG_DIR, help="Directory for TensorBoard logs")
    parser.add_argument("--model_dir", type=str, default=config.MODEL_DIR, help="Directory to save models and stats")
    parser.add_argument("--model_prefix", type=str, default=config.MODEL_PREFIX, help="Prefix for saved model files and VecNormalize stats")
    parser.add_argument("--tb_log_name", type=str, default=config.TB_LOG_NAME, help="TensorBoard log name")

    # --- Environment Parameters (Defaults from config.py) ---
    parser.add_argument("--width", type=int, default=config.WIDTH, help="Width of the Minesweeper grid")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="Height of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=config.N_MINES, help="Number of mines in the grid")
    parser.add_argument("--reward-win", type=float, default=config.REWARD_WIN, help="Reward for winning the game")
    parser.add_argument("--reward-lose", type=float, default=config.REWARD_LOSE, help="Penalty for hitting a mine")
    parser.add_argument("--reward-reveal", type=float, default=config.REWARD_REVEAL, help="Reward for revealing a safe cell")
    parser.add_argument("--reward-invalid", type=float, default=config.REWARD_INVALID, help="Penalty for clicking revealed cells")
    parser.add_argument("--max-reward-per-step", type=float, default=config.MAX_REWARD_PER_STEP, help="Maximum reward in one step")

    # --- Other Settings ---
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for training (auto, cpu, cuda)")
    parser.add_argument("--vec_env_type", type=str, default="subproc", choices=["subproc", "dummy"], help="Type of VecEnv (subproc for parallel, dummy for sequential/debug)")

    args = parser.parse_args()

    print("--- Training Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-----------------------------")


    # --- Ensure directories exist ---
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # --- Construct paths ---
    final_model_path = os.path.join(args.model_dir, f"{args.model_prefix}_{args.width}x{args.height}x{args.n_mines}_final.zip")
    stats_path = os.path.join(args.model_dir, f"{args.model_prefix}_{args.width}x{args.height}x{args.n_mines}_vecnormalize.pkl")

    # --- Parse Network Architecture ---
    try:
        pi_layer_sizes = parse_int_list(args.pi_layers)
        vf_layer_sizes = parse_int_list(args.vf_layers)
        # Assuming Stable Baselines3 expects a list containing one dict for MLP arch after features
        NET_ARCH_CONFIG = dict(pi=pi_layer_sizes, vf=vf_layer_sizes)
        print(f"Parsed net_arch: {NET_ARCH_CONFIG}")
    except argparse.ArgumentTypeError as e:
        print(f"Error parsing network layers: {e}")
        exit(1)

    # --- Define Policy Kwargs ---
    POLICY_KWARGS = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=NET_ARCH_CONFIG
    )

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
        save_path=args.model_dir,
        name_prefix=f"{args.model_prefix}_{args.width}x{args.height}x{args.n_mines}",
        save_replay_buffer=True, # Save replay buffer for off-policy algos (doesn't hurt for PPO)
        save_vecnormalize=True # Save VecNormalize statistics automatically
    )
    print(f"Checkpoints will be saved every {save_freq_per_env * args.n_envs} total steps.")


    # --- Create MaskablePPO Model ---
    model = MaskablePPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=args.log_dir,
        device=args.device,
        seed=args.seed
    )
    print(f"Model created on device: {model.device}")

    # --- Training ---
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=f"{args.tb_log_name}_{args.width}x{args.height}x{args.n_mines}",
            reset_num_timesteps=True # Start timesteps from 0
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