# config.py

# Environment Parameters
WIDTH = 5
HEIGHT = 5
N_MINES = 3
# Reward Parameters
REWARD_WIN = 0.2  # 胜利奖励
REWARD_LOSE = -0.05  # 失败惩罚
REWARD_REVEAL = 0.1  # 每揭开一个安全格子的奖励
REWARD_INVALID = -0.1  # 点击已揭开格子的惩罚
MAX_REWARD_PER_STEP = None  # 单步最大奖励，None 表示不限制

# Log Parameters
EXPERIMENT_BASE_DIR = "./training_runs" #Base directory for all training run outputs
MODEL_PREFIX = "ppo_run" # Model Prefix - Define the core name for your training runs here
