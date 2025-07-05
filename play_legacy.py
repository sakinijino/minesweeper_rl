# play.py
import os
import gymnasium as gym
import pygame
import time
import argparse
import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

# 导入你的环境类
from src.env.minesweeper_env import MinesweeperEnv
# 导入 config 来获取默认值
from src.utils import config
# 导入 checkpoint 相关工具函数
from src.utils.checkpoint_utils import find_best_checkpoint, load_training_config, find_vecnormalize_stats
# 导入模型工厂
from src.factories.model_factory import create_inference_model
# 导入环境工厂
from src.factories.environment_factory import create_inference_environment


# ===== 游戏模式相关辅助函数 =====

def create_environment(args, render_mode=None, vecnormalize_stats_path=None):
    """
    创建 Minesweeper 环境实例（使用环境工厂）。
    
    Args:
        args: 命令行参数
        render_mode: 渲染模式 ('human', None, 'rgb_array')
        vecnormalize_stats_path: VecNormalize统计文件路径
        
    Returns:
        配置好的环境实例
    """
    # 确定模式
    if render_mode == 'human':
        mode = 'interactive'
    else:
        mode = 'batch'
    
    # 使用环境工厂创建推理环境
    env, raw_env = create_inference_environment(
        args, 
        mode=mode,
        vecnormalize_stats_path=vecnormalize_stats_path
    )
    
    return env, raw_env


def setup_random_seed(args, env):
    """设置随机种子"""
    if args.seed is not None:
        set_random_seed(args.seed)
        env.seed(args.seed)
        print(f"Using random seed: {args.seed}")


def load_model_and_environment(args, env):
    """
    加载模型和环境统计数据。
    
    Args:
        args: 命令行参数
        env: 环境实例
        
    Returns:
        Tuple[model, updated_env] 或 (None, env) 如果不需要模型
    """
    model_path = args.model_path
    stats_path = args.stats_path
    
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return None, env

    print(f"Loading model: {model_path}")
    
    try:
        model, env = create_inference_model(
            env=env,
            checkpoint_path=model_path,
            vecnormalize_stats_path=stats_path,
            device=args.device
        )
        print(f"Model loaded on device: {model.device}")
        
        # 设置 VecNormalize 为评估模式
        if hasattr(env, 'training'):
            env.training = False
            env.norm_reward = False
            print("VecNormalize set to evaluation mode")
            
        return model, env
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, env


def print_final_statistics(total_games, wins, player_type):
    """打印最终统计信息"""
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    print(f"\n--- Final {player_type} Mode Statistics ---")
    print(f"Total Games Played: {total_games}")
    print(f"{player_type} Wins: {wins}")
    print(f"{player_type} Win Rate: {win_rate:.2f}%")


def print_episode_result(episode, total_episodes, episode_steps, episode_reward, won_episode):
    """打印单局游戏结果"""
    status = "Win" if won_episode else "Lose"
    print(f"Episode {episode + 1}/{total_episodes} finished - "
          f"Steps: {episode_steps}, Reward: {episode_reward:.2f}, Result: {status}")


# ===== 游戏模式函数 =====

def run_batch_mode(args):
    """
    批量模式：智能体在无界面环境中进行多局游戏测试。
    """
    print(f"--- Running Batch Mode ({args.num_episodes} episodes) ---")

    # 创建环境（无渲染）
    env, _ = create_environment(args, render_mode=None, vecnormalize_stats_path=args.stats_path)
    setup_random_seed(args, env)
    
    # 加载模型
    model, env = load_model_and_environment(args, env)
    if model is None:
        env.close()
        return

    # 批量游戏循环
    total_games = 0
    wins = 0

    for episode in range(args.num_episodes):
        obs = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        won_episode = False

        while not terminated and not truncated:
            # 获取动作掩码并预测动作
            action_masks = env.env_method("action_masks")[0]
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

            # 执行动作
            obs, reward, terminated_arr, info_arr = env.step(action)
            terminated = terminated_arr[0]
            actual_info = info_arr[0]
            won_episode = actual_info.get('is_success', False)
            truncated = actual_info.get('TimeLimit.truncated', False)
            reward = reward[0]

            episode_reward += reward
            episode_steps += 1

        # 统计结果
        total_games += 1
        if won_episode:
            wins += 1

        print_episode_result(episode, args.num_episodes, episode_steps, episode_reward, won_episode)

    # 打印最终统计
    print_final_statistics(total_games, wins, "Agent")
    env.close()


def run_human_mode(args):
    """
    人类模式：人类玩家通过鼠标点击进行游戏。
    """
    print("--- Running Human Mode ---")

    # 创建环境（有渲染）
    env, env_instance = create_environment(args, render_mode='human', vecnormalize_stats_path=args.stats_path)
    setup_random_seed(args, env)
    
    # 人类模式不需要加载模型，但需要处理 VecNormalize
    # VecNormalize 统计已经在环境工厂中处理了（如果提供了 stats_path）

    # 游戏统计
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

            # 处理 Pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 人类点击输入
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

            # 执行动作（如果有）
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

                # 检查游戏结束
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

                    time.sleep(2)  # 暂停显示结果
                    print("Resetting environment...")
                    obs = env.reset()
                    total_reward = 0
                    step_count = 0
                    terminated = False
                    truncated = False
                    current_game_won = False
            else:
                # 没有动作时仍需渲染以保持窗口更新
                env_instance._render_frame()

    except KeyboardInterrupt:
        print("\nHuman mode interrupted by user.")
    finally:
        print_final_statistics(total_games, player_wins, "Human")
        env.close()
        print("Environment closed. Game exited.")


def run_agent_mode(args):
    """
    智能体演示模式：智能体在有界面环境中进行游戏演示。
    """
    print("--- Running Agent Mode ---")

    # 创建环境（有渲染）
    env, env_instance = create_environment(args, render_mode='human', vecnormalize_stats_path=args.stats_path)
    setup_random_seed(args, env)
    
    # 加载模型
    model, env = load_model_and_environment(args, env)
    if model is None:
        env.close()
        return

    # 游戏统计
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
            # 处理 Pygame 事件（主要是退出事件）
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

            # 智能体决策
            action_masks = env.env_method("action_masks")[0]
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            print(f"Agent action: {action}")
            time.sleep(args.delay)  # 延迟以便观察

            # 执行动作
            obs, reward, terminated_arr, info_arr = env.step(action)
            terminated = terminated_arr[0]
            actual_info = info_arr[0]
            current_game_won = actual_info.get('is_success', False)
            truncated = actual_info.get('TimeLimit.truncated', False)
            reward = reward[0]

            total_reward += reward
            step_count += 1
            
            print(f"Step: {step_count}, Action: {action[0]}, Reward: {reward:.2f}, Done: {terminated}")

            # 检查游戏结束
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

                time.sleep(2)  # 暂停显示结果
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


# ===== 辅助函数 =====

def setup_argument_parser():
    """
    设置并返回配置好的参数解析器。
    
    Returns:
        ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description="Play Minesweeper in different modes.")

    # --- 模式选择 (必填参数) ---
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["agent", "batch", "human"], 
                        help="Play mode: agent (AI with visualization), batch (AI without visualization), human (human player)")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run in batch mode.")

    # --- 路径和命名 ---
    parser.add_argument("--training_run_dir", type=str, default=config.EXPERIMENT_BASE_DIR, 
                        help="Directory to training dir of save models and stats")
    parser.add_argument("--checkpoint_steps", type=int, default=None, 
                        help="Specific checkpoint step to load (uses latest if not specified)")

    # --- 环境参数 ---
    parser.add_argument("--width", type=int, default=config.WIDTH, help="Width of the Minesweeper grid")
    parser.add_argument("--height", type=int, default=config.HEIGHT, help="Height of the Minesweeper grid")
    parser.add_argument("--n_mines", type=int, default=config.N_MINES, help="Number of mines in the grid")
    parser.add_argument("--reward_win", type=float, default=config.REWARD_WIN, help="Reward for winning the game")
    parser.add_argument("--reward_lose", type=float, default=config.REWARD_LOSE, help="Penalty for hitting a mine")
    parser.add_argument("--reward_reveal", type=float, default=config.REWARD_REVEAL, help="Reward for revealing a safe cell")
    parser.add_argument("--reward_invalid", type=float, default=config.REWARD_INVALID, help="Penalty for clicking revealed cells")
    parser.add_argument("--max_reward_per_step", type=float, default=config.MAX_REWARD_PER_STEP, help="Maximum reward in one step")

    # --- 执行参数 ---
    parser.add_argument("--delay", type=float, default=0.1, help="Delay (in seconds) between agent moves in interactive mode.")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], 
                        help="Device to use for loading the model (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for environment reset (optional)")

    return parser


def resolve_model_paths(args):
    """
    解析模型路径，查找检查点和统计文件。
    
    Args:
        args: 命令行参数
        
    Returns:
        Tuple[str, str]: (模型路径, 统计文件路径)
    """
    try:
        # 检查 training_run_dir 是否是运行目录（包含 models/ 子目录）
        models_dir = os.path.join(args.training_run_dir, "models")
        if os.path.exists(models_dir):
            checkpoint_dir = models_dir
        else:
            # 假设 training_run_dir 已经是 models 目录
            checkpoint_dir = args.training_run_dir
        
        # 查找最佳检查点
        best_checkpoint = find_best_checkpoint(checkpoint_dir, args.checkpoint_steps)
        print(f"Selected checkpoint: {os.path.basename(best_checkpoint)}")
        
        # 查找对应的 VecNormalize 统计文件
        stats_path = find_vecnormalize_stats(best_checkpoint)
        if stats_path:
            print(f"Found VecNormalize stats: {os.path.basename(stats_path)}")
        else:
            print("Warning: No VecNormalize stats file found, using unnormalized environment")
        
        print(f"Model path: {best_checkpoint}")
        print(f"Stats path: {stats_path}")
        
        return best_checkpoint, stats_path
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the training_run_dir contains valid checkpoint files.")
        exit(1)
    except Exception as e:
        print(f"Error setting up model paths: {e}")
        exit(1)


def print_play_configuration(args):
    """打印游戏配置信息"""
    print("--- Play Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("--------------------------")


def run_selected_mode(args):
    """
    根据选定的模式运行相应的游戏模式。
    
    Args:
        args: 命令行参数
    """
    if args.mode == "batch":
        run_batch_mode(args)
    elif args.mode == "human":
        run_human_mode(args)
    elif args.mode == "agent":
        run_agent_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        exit(1)


def main():
    """主游戏函数"""
    # 1. 设置参数解析
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 2. 打印配置信息
    print_play_configuration(args)
    
    # 3. 解析模型路径（对于需要AI的模式）
    if args.mode in ["agent", "batch"]:
        model_path, stats_path = resolve_model_paths(args)
        args.model_path = model_path
        args.stats_path = stats_path
    else:
        # 人类模式不需要模型路径
        args.model_path = None
        args.stats_path = None
    
    # 4. 运行选定的模式
    run_selected_mode(args)


# ===== 主程序 =====

if __name__ == "__main__":
    main()