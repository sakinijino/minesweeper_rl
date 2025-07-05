import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.config_manager import ConfigManager
from src.config.config_schemas import (
    TrainingConfig,
    ModelHyperparams,
    EnvironmentConfig,
    TrainingExecutionConfig,
    PlayConfig
)


class TestConfigManager:
    def test_init(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert isinstance(manager.config, TrainingConfig)

    def test_load_from_file(self, temp_dir):
        """Test loading configuration from JSON file."""
        # Create test config file
        config_data = {
            "model_hyperparams": {
                "learning_rate": 2e-4,
                "gamma": 0.95
            },
            "environment_config": {
                "width": 8,
                "height": 8,
                "n_mines": 10
            },
            "training_execution": {
                "total_timesteps": 500000,
                "n_envs": 2
            }
        }
        
        config_path = os.path.join(temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigManager()
        manager.load_from_file(config_path)
        
        assert manager.config.model_hyperparams.learning_rate == 2e-4
        assert manager.config.model_hyperparams.gamma == 0.95
        assert manager.config.environment_config.width == 8
        assert manager.config.environment_config.height == 8
        assert manager.config.training_execution.total_timesteps == 500000
        assert manager.config.training_execution.n_envs == 2

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file."""
        manager = ConfigManager()
        with pytest.raises(FileNotFoundError):
            manager.load_from_file("/nonexistent/path.json")

    def test_load_from_invalid_json(self, temp_dir):
        """Test loading from invalid JSON file."""
        config_path = os.path.join(temp_dir, "invalid.json")
        with open(config_path, 'w') as f:
            f.write("invalid json content")
        
        manager = ConfigManager()
        with pytest.raises(json.JSONDecodeError):
            manager.load_from_file(config_path)

    def test_save_to_file(self, temp_dir):
        """Test saving configuration to JSON file."""
        manager = ConfigManager()
        manager.config.model_hyperparams.learning_rate = 2e-4
        manager.config.environment_config.width = 8
        
        config_path = os.path.join(temp_dir, "saved_config.json")
        manager.save_to_file(config_path)
        
        assert os.path.exists(config_path)
        
        # Verify saved content
        with open(config_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["model_hyperparams"]["learning_rate"] == 2e-4
        assert saved_data["environment_config"]["width"] == 8

    def test_update_from_dict(self):
        """Test updating configuration from dictionary."""
        manager = ConfigManager()
        
        updates = {
            "model_hyperparams": {
                "learning_rate": 3e-4,
                "gamma": 0.98
            },
            "environment_config": {
                "width": 12,
                "n_mines": 25
            }
        }
        
        manager.update_from_dict(updates)
        
        assert manager.config.model_hyperparams.learning_rate == 3e-4
        assert manager.config.model_hyperparams.gamma == 0.98
        assert manager.config.environment_config.width == 12
        assert manager.config.environment_config.n_mines == 25
        # Check that other values remain unchanged
        assert manager.config.environment_config.height == 16  # default value

    def test_update_from_args(self):
        """Test updating configuration from argparse namespace."""
        manager = ConfigManager()
        
        # Mock argparse namespace
        class MockArgs:
            def __init__(self):
                self.learning_rate = 5e-4
                self.width = 10
                self.height = 10
                self.total_timesteps = 2_000_000
                self.nonexistent_param = "should_be_ignored"
        
        args = MockArgs()
        manager.update_from_args(args)
        
        assert manager.config.model_hyperparams.learning_rate == 5e-4
        assert manager.config.environment_config.width == 10
        assert manager.config.environment_config.height == 10
        assert manager.config.training_execution.total_timesteps == 2_000_000

    def test_get_training_config(self):
        """Test getting training configuration."""
        manager = ConfigManager()
        training_config = manager.get_training_config()
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config == manager.config

    def test_get_environment_config(self):
        """Test getting environment configuration."""
        manager = ConfigManager()
        env_config = manager.get_environment_config()
        
        assert isinstance(env_config, EnvironmentConfig)
        assert env_config == manager.config.environment_config

    def test_get_play_config(self):
        """Test getting play configuration."""
        manager = ConfigManager()
        
        # Set some play-specific values
        manager.config.play_config = PlayConfig(
            mode="agent",
            num_episodes=50,
            delay=0.5
        )
        
        play_config = manager.get_play_config()
        
        assert isinstance(play_config, PlayConfig)
        assert play_config.mode == "agent"
        assert play_config.num_episodes == 50
        assert play_config.delay == 0.5

    def test_validate_config(self):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Test valid config
        assert manager.validate_config() is True
        
        # Test invalid config (this will be expanded when validation is implemented)
        manager.config.environment_config.width = 0
        assert manager.validate_config() is False

    def test_merge_configs(self):
        """Test merging two configurations."""
        manager1 = ConfigManager()
        manager1.config.model_hyperparams.learning_rate = 1e-4
        manager1.config.environment_config.width = 8
        
        # Create partial config for merging
        partial_config_dict = {
            "model_hyperparams": {
                "learning_rate": 2e-4
            },
            "environment_config": {
                "height": 12
            }
        }
        
        # Update manager1 with partial config
        manager1.update_from_dict(partial_config_dict)
        
        # Check that values are updated correctly
        assert manager1.config.model_hyperparams.learning_rate == 2e-4
        assert manager1.config.environment_config.height == 12
        # Check that other values remain unchanged
        assert manager1.config.environment_config.width == 8

    def test_load_from_training_run(self, mock_training_run_dir):
        """Test loading configuration from training run directory."""
        manager = ConfigManager()
        manager.load_from_training_run(mock_training_run_dir)
        
        # Check that values from the mock training config are loaded
        assert manager.config.training_execution.total_timesteps == 100000
        assert manager.config.training_execution.n_envs == 4
        assert manager.config.model_hyperparams.learning_rate == 0.0001
        assert manager.config.environment_config.width == 5
        assert manager.config.environment_config.height == 5
        assert manager.config.environment_config.n_mines == 3

    def test_load_from_training_run_missing_config(self, temp_dir):
        """Test loading from training run directory without config file."""
        manager = ConfigManager()
        
        # Create directory without config file
        run_dir = os.path.join(temp_dir, "incomplete_run")
        os.makedirs(run_dir)
        
        with pytest.raises(FileNotFoundError):
            manager.load_from_training_run(run_dir)

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = ConfigManager.create_default_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.model_hyperparams.learning_rate == 1e-4
        assert config.environment_config.width == 16
        assert config.training_execution.total_timesteps == 1_000_000

    def test_create_play_config(self):
        """Test creating play configuration from training config."""
        training_config = TrainingConfig()
        training_config.environment_config.width = 8
        training_config.environment_config.height = 8
        training_config.environment_config.n_mines = 10
        
        play_config = ConfigManager.create_play_config(
            training_config, 
            mode="agent", 
            num_episodes=50
        )
        
        assert isinstance(play_config, PlayConfig)
        assert play_config.mode == "agent"
        assert play_config.num_episodes == 50
        # Environment config should be preserved
        assert play_config.environment_config.width == 8
        assert play_config.environment_config.height == 8
        assert play_config.environment_config.n_mines == 10