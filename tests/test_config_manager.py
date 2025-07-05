import pytest
import tempfile
import json
import yaml
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from argparse import Namespace

from src.config.config_manager import ConfigManager, ConfigurationError
from src.config.config_schemas import (
    TrainingConfig,
    ModelHyperparams,
    EnvironmentConfig,
    TrainingExecutionConfig,
    PlayConfig
)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
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
            "width": 16,
            "height": 16,
            "n_mines": 40,
            "reward_win": 10.0,
            "reward_lose": -10.0,
            "reward_reveal": 1.0,
            "reward_invalid": -1.0,
            "max_reward_per_step": 10.0
        },
        "training_execution": {
            "total_timesteps": 1_000_000,
            "n_envs": 4,
            "vec_env_type": "subproc",
            "checkpoint_freq": 50000,
            "device": "auto",
            "seed": 42
        },
        "paths_config": {
            "experiment_base_dir": "experiments",
            "model_prefix": "minesweeper_ppo"
        }
    }


@pytest.fixture
def config_yaml_file(temp_dir, sample_config_data):
    """Create a YAML configuration file for testing."""
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_data, f)
    return config_path


@pytest.fixture
def config_json_file(temp_dir, sample_config_data):
    """Create a JSON configuration file for testing."""
    config_path = os.path.join(temp_dir, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(sample_config_data, f, indent=2)
    return config_path


@pytest.fixture
def incomplete_config_file(temp_dir):
    """Create an incomplete configuration file for testing."""
    incomplete_config = {
        "model_hyperparams": {
            "learning_rate": 1e-4,
            "gamma": 0.99
        },
        "environment_config": {
            "width": 8,
            "height": 8
        }
    }
    
    config_path = os.path.join(temp_dir, "incomplete_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(incomplete_config, f)
    return config_path


class TestConfigManager:
    """Test suite for ConfigManager with new priority system."""
    
    def test_init_without_config_file(self):
        """Test ConfigManager initialization without config file."""
        manager = ConfigManager()
        assert manager.config is None
        assert manager.config_sources['file'] is None
        assert manager.config_sources['args'] is None
        assert manager.config_sources['continue_train'] is None
    
    def test_init_with_config_file(self, config_yaml_file):
        """Test ConfigManager initialization with config file."""
        manager = ConfigManager(config_yaml_file)
        assert manager.config_sources['file'] is not None
        assert manager.config is None  # Not built yet
    
    def test_load_yaml_config_file(self, config_yaml_file):
        """Test loading YAML configuration file."""
        manager = ConfigManager()
        manager.load_config_file(config_yaml_file)
        
        assert manager.config_sources['file'] is not None
        assert manager.config_sources['file']['model_hyperparams']['learning_rate'] == 1e-4
        assert manager.config_sources['file']['environment_config']['width'] == 16
    
    def test_load_json_config_file(self, config_json_file):
        """Test loading JSON configuration file."""
        manager = ConfigManager()
        manager.load_config_file(config_json_file)
        
        assert manager.config_sources['file'] is not None
        assert manager.config_sources['file']['model_hyperparams']['learning_rate'] == 1e-4
        assert manager.config_sources['file']['environment_config']['width'] == 16
    
    def test_load_nonexistent_config_file(self):
        """Test loading non-existent configuration file."""
        manager = ConfigManager()
        with pytest.raises(FileNotFoundError):
            manager.load_config_file("/nonexistent/path.yaml")
    
    def test_load_invalid_yaml_file(self, temp_dir):
        """Test loading invalid YAML file."""
        config_path = os.path.join(temp_dir, "invalid.yaml")
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        manager = ConfigManager()
        with pytest.raises(ConfigurationError):
            manager.load_config_file(config_path)
    
    def test_load_unsupported_file_format(self, temp_dir):
        """Test loading unsupported file format."""
        config_path = os.path.join(temp_dir, "config.txt")
        with open(config_path, 'w') as f:
            f.write("some content")
        
        manager = ConfigManager()
        with pytest.raises(ConfigurationError):
            manager.load_config_file(config_path)
    
    def test_load_from_args(self, config_yaml_file):
        """Test loading configuration from command-line arguments."""
        manager = ConfigManager(config_yaml_file)
        
        # Mock args
        args = Namespace(
            learning_rate=2e-4,
            width=8,
            height=8,
            total_timesteps=2_000_000,
            device="cuda",
            nonexistent_param="should_be_ignored"
        )
        
        manager.load_from_args(args)
        
        assert manager.config_sources['args'] is not None
        assert manager.config_sources['args']['model_hyperparams']['learning_rate'] == 2e-4
        assert manager.config_sources['args']['environment_config']['width'] == 8
        assert manager.config_sources['args']['training_execution']['total_timesteps'] == 2_000_000
        assert 'nonexistent_param' not in str(manager.config_sources['args'])
    
    def test_load_from_training_run(self, mock_training_run_dir):
        """Test loading configuration from training run directory."""
        manager = ConfigManager()
        manager.load_from_training_run(mock_training_run_dir)
        
        assert manager.config_sources['continue_train'] is not None
        assert manager.config_sources['continue_train']['training_execution']['total_timesteps'] == 100000
        assert manager.config_sources['continue_train']['environment_config']['width'] == 5
    
    def test_load_from_training_run_missing_config(self, temp_dir):
        """Test loading from training run directory without config file."""
        manager = ConfigManager()
        
        # Create directory without config file
        run_dir = os.path.join(temp_dir, "incomplete_run")
        os.makedirs(run_dir)
        
        with pytest.raises(FileNotFoundError):
            manager.load_from_training_run(run_dir)
    
    def test_build_config_from_file_only(self, config_yaml_file):
        """Test building configuration from file only."""
        manager = ConfigManager(config_yaml_file)
        config = manager.build_config()
        
        assert isinstance(config, TrainingConfig)
        assert config.model_hyperparams.learning_rate == 1e-4
        assert config.environment_config.width == 16
        assert config.training_execution.total_timesteps == 1_000_000
    
    def test_build_config_incomplete_file(self, incomplete_config_file):
        """Test building configuration from incomplete file."""
        manager = ConfigManager(incomplete_config_file)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.build_config()
        
        assert "Required parameters missing" in str(exc_info.value)
    
    def test_build_config_with_priority_system(self, config_yaml_file):
        """Test parameter priority system: args > file > continue_train."""
        manager = ConfigManager(config_yaml_file)
        
        # Add continue_train source (lowest priority)
        manager.config_sources['continue_train'] = {
            "model_hyperparams": {
                "learning_rate": 5e-5  # Should be overridden
            },
            "environment_config": {
                "width": 10,  # Should be overridden
                "height": 10  # Should be overridden
            }
        }
        
        # Add args source (highest priority)
        args = Namespace(
            learning_rate=3e-4,  # Should override file and continue_train
            width=12  # Should override file and continue_train
        )
        manager.load_from_args(args)
        
        config = manager.build_config()
        
        # Check priority: args > file > continue_train
        assert config.model_hyperparams.learning_rate == 3e-4  # From args
        assert config.environment_config.width == 12  # From args
        assert config.environment_config.height == 16  # From file (args didn't override)
        assert config.training_execution.total_timesteps == 1_000_000  # From file
    
    def test_save_config_yaml(self, config_yaml_file, temp_dir):
        """Test saving configuration to YAML file."""
        manager = ConfigManager(config_yaml_file)
        config = manager.build_config()
        
        save_path = os.path.join(temp_dir, "saved_config.yaml")
        manager.save_config(save_path)
        
        assert os.path.exists(save_path)
        
        # Verify saved content
        with open(save_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data["model_hyperparams"]["learning_rate"] == 1e-4
        assert saved_data["environment_config"]["width"] == 16
    
    def test_save_config_json(self, config_yaml_file, temp_dir):
        """Test saving configuration to JSON file."""
        manager = ConfigManager(config_yaml_file)
        config = manager.build_config()
        
        save_path = os.path.join(temp_dir, "saved_config.json")
        manager.save_config(save_path)
        
        assert os.path.exists(save_path)
        
        # Verify saved content
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["model_hyperparams"]["learning_rate"] == 1e-4
        assert saved_data["environment_config"]["width"] == 16
    
    def test_save_config_without_built_config(self):
        """Test saving configuration before building it."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError):
            manager.save_config("/tmp/test.yaml")
    
    def test_get_config(self, config_yaml_file):
        """Test getting configuration."""
        manager = ConfigManager(config_yaml_file)
        manager.build_config()
        
        config = manager.get_config()
        assert isinstance(config, TrainingConfig)
        assert config.model_hyperparams.learning_rate == 1e-4
    
    def test_get_config_before_build(self):
        """Test getting configuration before building it."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError):
            manager.get_config()
    
    def test_training_config_has_no_play_config(self, config_yaml_file):
        """Test that TrainingConfig no longer contains play_config."""
        manager = ConfigManager(config_yaml_file)
        config = manager.build_config()
        
        # TrainingConfig should not have play_config field anymore
        assert not hasattr(config, 'play_config')
    
    def test_config_manager_no_longer_has_get_play_config(self):
        """Test that ConfigManager no longer has get_play_config method."""
        manager = ConfigManager()
        
        # ConfigManager should not have get_play_config method anymore
        assert not hasattr(manager, 'get_play_config')
    
    def test_create_from_config_file(self, config_yaml_file):
        """Test static method to create manager from config file."""
        manager = ConfigManager.create_from_config_file(config_yaml_file)
        
        assert manager.config is not None
        assert isinstance(manager.config, TrainingConfig)
        assert manager.config.model_hyperparams.learning_rate == 1e-4
    
    def test_create_with_args(self, config_yaml_file):
        """Test static method to create manager with args."""
        args = Namespace(
            learning_rate=2e-4,
            width=8
        )
        
        manager = ConfigManager.create_with_args(config_yaml_file, args)
        
        assert manager.config is not None
        assert manager.config.model_hyperparams.learning_rate == 2e-4  # From args
        assert manager.config.environment_config.width == 8  # From args
        assert manager.config.environment_config.height == 16  # From file
    
    def test_validation_fails_for_invalid_config(self, temp_dir):
        """Test that validation fails for invalid configuration."""
        invalid_config = {
            "model_hyperparams": {
                "learning_rate": -1e-4,  # Invalid: negative learning rate
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
                "width": 0,  # Invalid: zero width
                "height": 16,
                "n_mines": 40,
                "reward_win": 10.0,
                "reward_lose": -10.0,
                "reward_reveal": 1.0,
                "reward_invalid": -1.0,
                "max_reward_per_step": 10.0
            },
            "training_execution": {
                "total_timesteps": 1_000_000,
                "n_envs": 4,
                "vec_env_type": "subproc",
                "checkpoint_freq": 50000,
                "device": "auto",
                "seed": 42
            },
            "paths_config": {
                "experiment_base_dir": "experiments",
                "model_prefix": "minesweeper_ppo"
            }
        }
        
        config_path = os.path.join(temp_dir, "invalid_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        manager = ConfigManager(config_path)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.build_config()
        
        assert "Configuration validation failed" in str(exc_info.value)
    
    def test_deep_merge(self, config_yaml_file):
        """Test deep merging of configuration dictionaries."""
        manager = ConfigManager(config_yaml_file)
        
        # Test internal _deep_merge method
        dict1 = {
            "model_hyperparams": {
                "learning_rate": 1e-4,
                "gamma": 0.99
            },
            "environment_config": {
                "width": 16
            }
        }
        
        dict2 = {
            "model_hyperparams": {
                "learning_rate": 2e-4,  # Should override
                "ent_coef": 0.01  # Should be added
            },
            "training_execution": {
                "total_timesteps": 1_000_000  # Should be added
            }
        }
        
        result = manager._deep_merge(dict1, dict2)
        
        # Check that dict2 values override dict1
        assert result["model_hyperparams"]["learning_rate"] == 2e-4
        # Check that dict1 values are preserved if not in dict2
        assert result["model_hyperparams"]["gamma"] == 0.99
        # Check that dict2 values are added
        assert result["model_hyperparams"]["ent_coef"] == 0.01
        assert result["training_execution"]["total_timesteps"] == 1_000_000
        # Check that dict1 sections are preserved
        assert result["environment_config"]["width"] == 16