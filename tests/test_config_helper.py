"""
Helper functions for testing with the new configuration system.
"""

import tempfile
import yaml
import os
from src.config.config_manager import ConfigManager


def create_test_config_manager():
    """
    Create a ConfigManager instance for testing with a temporary config file.
    
    Returns:
        ConfigManager: Configured manager instance
    """
    # Create a minimal valid configuration
    test_config = {
        "model_hyperparams": {
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.90,
            "clip_range": 0.2,
            "vf_coef": 1.0,
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 10
        },
        "network_architecture": {
            "features_dim": 128,
            "pi_layers": [64, 64],
            "vf_layers": [256, 256]
        },
        "environment_config": {
            "width": 5,
            "height": 5,
            "n_mines": 3,
            "reward_win": 10.0,
            "reward_lose": -10.0,
            "reward_reveal": 1.0,
            "reward_invalid": -1.0,
            "max_reward_per_step": 10.0
        },
        "training_execution": {
            "total_timesteps": 100000,
            "n_envs": 4,
            "vec_env_type": "subproc",
            "checkpoint_freq": 10000,
            "device": "auto",
            "seed": 42
        },
        "paths_config": {
            "experiment_base_dir": "experiments",
            "model_prefix": "test_model"
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        # Create and build config manager
        config_manager = ConfigManager(temp_config_path)
        config_manager.build_config()
        return config_manager
    finally:
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)