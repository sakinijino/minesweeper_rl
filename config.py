# config.py

# Environment Parameters (Optional but good to keep consistent)
WIDTH = 4
HEIGHT = 4
N_MINES = 3

# Directories
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"

# Model Prefix - Define the core name for your training runs here
# This prefix will be used for checkpoint callbacks and the final model name.
MODEL_PREFIX = "ppo_minesweeper" # <-- CHANGE THIS AS NEEDED

# TensorBoard 日志名称
TB_LOG_NAME = "PPO_Minesweeper"
