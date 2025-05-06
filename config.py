# config.py

# Environment Parameters
WIDTH = 4
HEIGHT = 4
N_MINES = 3
# Reward Parameters
REWARD_WIN = 1.0  # 胜利奖励
REWARD_LOSE = -1.0  # 失败惩罚
REWARD_REVEAL = 0.1  # 每揭开一个安全格子的奖励
REWARD_INVALID = -0.1  # 点击已揭开格子的惩罚
MAX_REWARD_PER_STEP = None  # 单步最大奖励，None 表示不限制

# Directories
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"

# Model Prefix - Define the core name for your training runs here
# This prefix will be used for checkpoint callbacks and the final model name.
MODEL_PREFIX = "ppo_minesweeper" # <-- CHANGE THIS AS NEEDED

# TensorBoard 日志名称
TB_LOG_NAME = "PPO_Minesweeper"
