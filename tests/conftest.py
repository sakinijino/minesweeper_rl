import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_checkpoint_dir(temp_dir):
    """Create a mock checkpoint directory with sample files."""
    checkpoint_dir = os.path.join(temp_dir, "models")
    os.makedirs(checkpoint_dir)
    
    # Create sample checkpoint files
    checkpoint_files = [
        "rl_model_50000_steps.zip",
        "rl_model_100000_steps.zip", 
        "rl_model_150000_steps.zip",
        "final_model.zip"
    ]
    
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        # Create empty files for testing
        Path(filepath).touch()
    
    return checkpoint_dir


@pytest.fixture
def mock_training_run_dir(temp_dir):
    """Create a complete mock training run directory structure."""
    run_dir = os.path.join(temp_dir, "test_run_5x5x3_seed42_20250515-120000")
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(models_dir)
    os.makedirs(logs_dir)
    
    # Create checkpoint files
    checkpoint_files = [
        "rl_model_25000_steps.zip",
        "rl_model_50000_steps.zip",
        "rl_model_75000_steps.zip",
        "final_model.zip"
    ]
    
    for filename in checkpoint_files:
        filepath = os.path.join(models_dir, filename)
        Path(filepath).touch()
    
    # Create VecNormalize stats files
    stats_files = [
        "rl_model_vecnormalize_25000_steps.pkl",
        "rl_model_vecnormalize_50000_steps.pkl", 
        "rl_model_vecnormalize_75000_steps.pkl",
        "final_stats_vecnormalize.pkl"
    ]
    
    for filename in stats_files:
        filepath = os.path.join(models_dir, filename)
        Path(filepath).touch()
    
    # Create training config
    config = {
        "model_hyperparams": {
            "learning_rate": 0.0001
        },
        "environment_config": {
            "width": 5,
            "height": 5,
            "n_mines": 3
        },
        "training_execution": {
            "total_timesteps": 100000,
            "n_envs": 4,
            "seed": 42
        }
    }
    
    config_path = os.path.join(run_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    return run_dir


@pytest.fixture
def empty_checkpoint_dir(temp_dir):
    """Create an empty checkpoint directory."""
    checkpoint_dir = os.path.join(temp_dir, "empty_models")
    os.makedirs(checkpoint_dir)
    return checkpoint_dir