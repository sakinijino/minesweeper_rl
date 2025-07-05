import pytest
import tempfile
import json
from pathlib import Path
from dataclasses import asdict

from src.config.config_schemas import (
    ModelHyperparams,
    NetworkArchitecture,
    EnvironmentConfig,
    TrainingExecutionConfig,
    PathsConfig,
    PlayConfig,
    TrainingConfig
)


class TestModelHyperparams:
    def test_default_values(self):
        """Test default hyperparameter values."""
        params = ModelHyperparams()
        assert params.learning_rate == 1e-4
        assert params.ent_coef == 0.01
        assert params.gamma == 0.99
        assert params.gae_lambda == 0.90
        assert params.clip_range == 0.2
        assert params.vf_coef == 1.0
        assert params.n_steps == 1024
        assert params.batch_size == 128
        assert params.n_epochs == 10

    def test_custom_values(self):
        """Test custom hyperparameter values."""
        params = ModelHyperparams(
            learning_rate=2e-4,
            ent_coef=0.02,
            gamma=0.95
        )
        assert params.learning_rate == 2e-4
        assert params.ent_coef == 0.02
        assert params.gamma == 0.95
        # Check defaults are preserved
        assert params.gae_lambda == 0.90

    def test_validation_range(self):
        """Test parameter validation ranges."""
        # This will be implemented when we add validation
        pass


class TestNetworkArchitecture:
    def test_default_architecture(self):
        """Test default network architecture."""
        arch = NetworkArchitecture()
        assert arch.features_dim == 128
        assert arch.pi_layers == [64, 64]
        assert arch.vf_layers == [256, 256]

    def test_custom_architecture(self):
        """Test custom network architecture."""
        arch = NetworkArchitecture(
            features_dim=256,
            pi_layers=[128, 128, 64],
            vf_layers=[512, 256, 128]
        )
        assert arch.features_dim == 256
        assert arch.pi_layers == [128, 128, 64]
        assert arch.vf_layers == [512, 256, 128]


class TestEnvironmentConfig:
    def test_default_environment(self):
        """Test default environment configuration."""
        env_config = EnvironmentConfig()
        assert env_config.width == 16
        assert env_config.height == 16
        assert env_config.n_mines == 40
        assert env_config.reward_win == 10.0
        assert env_config.reward_lose == -10.0
        assert env_config.reward_reveal == 1.0
        assert env_config.reward_invalid == -1.0
        assert env_config.max_reward_per_step == 10.0

    def test_custom_environment(self):
        """Test custom environment configuration."""
        env_config = EnvironmentConfig(
            width=8,
            height=8,
            n_mines=10,
            reward_win=20.0
        )
        assert env_config.width == 8
        assert env_config.height == 8
        assert env_config.n_mines == 10
        assert env_config.reward_win == 20.0
        # Check defaults are preserved
        assert env_config.reward_lose == -10.0

    def test_environment_validation(self):
        """Test environment parameter validation."""
        # This will be implemented when we add validation
        pass


class TestTrainingExecutionConfig:
    def test_default_training_config(self):
        """Test default training execution configuration."""
        config = TrainingExecutionConfig()
        assert config.total_timesteps == 1_000_000
        assert config.n_envs == 4
        assert config.vec_env_type == "subproc"
        assert config.checkpoint_freq == 50000
        assert config.device == "auto"
        assert config.seed is None

    def test_custom_training_config(self):
        """Test custom training execution configuration."""
        config = TrainingExecutionConfig(
            total_timesteps=2_000_000,
            n_envs=8,
            vec_env_type="dummy",
            seed=42
        )
        assert config.total_timesteps == 2_000_000
        assert config.n_envs == 8
        assert config.vec_env_type == "dummy"
        assert config.seed == 42


class TestPathsConfig:
    def test_default_paths(self):
        """Test default paths configuration."""
        paths = PathsConfig()
        assert paths.experiment_base_dir == "experiments"
        assert paths.model_prefix == "minesweeper_ppo"

    def test_custom_paths(self):
        """Test custom paths configuration."""
        paths = PathsConfig(
            experiment_base_dir="/custom/experiments",
            model_prefix="custom_model"
        )
        assert paths.experiment_base_dir == "/custom/experiments"
        assert paths.model_prefix == "custom_model"


class TestPlayConfig:
    def test_default_play_config(self):
        """Test default play configuration."""
        play_config = PlayConfig()
        assert play_config.mode == "batch"
        assert play_config.num_episodes == 100
        assert play_config.delay == 0.1
        assert play_config.checkpoint_steps is None

    def test_custom_play_config(self):
        """Test custom play configuration."""
        play_config = PlayConfig(
            mode="agent",
            num_episodes=50,
            delay=0.5,
            checkpoint_steps=100000
        )
        assert play_config.mode == "agent"
        assert play_config.num_episodes == 50
        assert play_config.delay == 0.5
        assert play_config.checkpoint_steps == 100000


class TestTrainingConfig:
    def test_default_training_config(self):
        """Test default complete training configuration."""
        config = TrainingConfig()
        assert isinstance(config.model_hyperparams, ModelHyperparams)
        assert isinstance(config.network_architecture, NetworkArchitecture)
        assert isinstance(config.environment_config, EnvironmentConfig)
        assert isinstance(config.training_execution, TrainingExecutionConfig)
        assert isinstance(config.paths_config, PathsConfig)

    def test_custom_training_config(self):
        """Test custom complete training configuration."""
        custom_hyperparams = ModelHyperparams(learning_rate=2e-4)
        custom_env = EnvironmentConfig(width=8, height=8, n_mines=10)
        
        config = TrainingConfig(
            model_hyperparams=custom_hyperparams,
            environment_config=custom_env
        )
        
        assert config.model_hyperparams.learning_rate == 2e-4
        assert config.environment_config.width == 8
        # Check that other components use defaults
        assert config.training_execution.total_timesteps == 1_000_000

    def test_to_dict_conversion(self):
        """Test converting configuration to dictionary."""
        config = TrainingConfig()
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert "model_hyperparams" in config_dict
        assert "environment_config" in config_dict
        assert "training_execution" in config_dict
        assert "paths_config" in config_dict
        
        # Test nested structure
        assert "learning_rate" in config_dict["model_hyperparams"]
        assert "width" in config_dict["environment_config"]

    def test_from_dict_conversion(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "model_hyperparams": {
                "learning_rate": 2e-4,
                "gamma": 0.95
            },
            "environment_config": {
                "width": 8,
                "height": 8,
                "n_mines": 10
            }
        }
        
        # This will be implemented when we add from_dict method
        pass