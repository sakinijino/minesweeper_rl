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
    TrainingConfig,
    validate_environment_config,
    validate_training_config,
    create_config_from_dict
)


@pytest.fixture
def sample_model_hyperparams():
    """Sample model hyperparameters for testing."""
    return ModelHyperparams(
        learning_rate=1e-4,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.90,
        clip_range=0.2,
        vf_coef=1.0,
        n_steps=1024,
        batch_size=128,
        n_epochs=10
    )


@pytest.fixture
def sample_network_architecture():
    """Sample network architecture for testing."""
    return NetworkArchitecture(
        features_dim=128,
        pi_layers=[64, 64],
        vf_layers=[256, 256]
    )


@pytest.fixture
def sample_environment_config():
    """Sample environment configuration for testing."""
    return EnvironmentConfig(
        width=16,
        height=16,
        n_mines=40,
        reward_win=10.0,
        reward_lose=-10.0,
        reward_reveal=1.0,
        reward_invalid=-1.0,
        max_reward_per_step=10.0
    )


@pytest.fixture
def sample_training_execution():
    """Sample training execution configuration for testing."""
    return TrainingExecutionConfig(
        total_timesteps=1_000_000,
        n_envs=4,
        vec_env_type="subproc",
        checkpoint_freq=50000,
        device="auto",
        seed=42
    )


@pytest.fixture
def sample_paths_config():
    """Sample paths configuration for testing."""
    return PathsConfig(
        experiment_base_dir="experiments",
        model_prefix="minesweeper_ppo"
    )


@pytest.fixture
def sample_play_config(sample_environment_config):
    """Sample play configuration for testing."""
    return PlayConfig(
        mode="batch",
        num_episodes=100,
        delay=0.1,
        checkpoint_steps=None,
        environment_config=sample_environment_config
    )


class TestModelHyperparams:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            ModelHyperparams()  # Should fail - no defaults

    def test_custom_values(self, sample_model_hyperparams):
        """Test custom hyperparameter values."""
        params = ModelHyperparams(
            learning_rate=2e-4,
            ent_coef=0.02,
            gamma=0.95,
            gae_lambda=0.85,
            clip_range=0.1,
            vf_coef=0.5,
            n_steps=512,
            batch_size=64,
            n_epochs=5
        )
        assert params.learning_rate == 2e-4
        assert params.ent_coef == 0.02
        assert params.gamma == 0.95
        assert params.gae_lambda == 0.85
        assert params.clip_range == 0.1
        assert params.vf_coef == 0.5
        assert params.n_steps == 512
        assert params.batch_size == 64
        assert params.n_epochs == 5

    def test_sample_values(self, sample_model_hyperparams):
        """Test sample hyperparameter values."""
        assert sample_model_hyperparams.learning_rate == 1e-4
        assert sample_model_hyperparams.ent_coef == 0.01
        assert sample_model_hyperparams.gamma == 0.99


class TestNetworkArchitecture:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            NetworkArchitecture()  # Should fail - no defaults

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

    def test_sample_architecture(self, sample_network_architecture):
        """Test sample network architecture."""
        assert sample_network_architecture.features_dim == 128
        assert sample_network_architecture.pi_layers == [64, 64]
        assert sample_network_architecture.vf_layers == [256, 256]


class TestEnvironmentConfig:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            EnvironmentConfig()  # Should fail - no defaults

    def test_custom_environment(self):
        """Test custom environment configuration."""
        env_config = EnvironmentConfig(
            width=8,
            height=8,
            n_mines=10,
            reward_win=20.0,
            reward_lose=-5.0,
            reward_reveal=2.0,
            reward_invalid=-2.0,
            max_reward_per_step=15.0
        )
        assert env_config.width == 8
        assert env_config.height == 8
        assert env_config.n_mines == 10
        assert env_config.reward_win == 20.0
        assert env_config.reward_lose == -5.0
        assert env_config.reward_reveal == 2.0
        assert env_config.reward_invalid == -2.0
        assert env_config.max_reward_per_step == 15.0

    def test_sample_environment(self, sample_environment_config):
        """Test sample environment configuration."""
        assert sample_environment_config.width == 16
        assert sample_environment_config.height == 16
        assert sample_environment_config.n_mines == 40
        assert sample_environment_config.reward_win == 10.0

    def test_environment_validation_valid(self, sample_environment_config):
        """Test environment parameter validation with valid config."""
        assert validate_environment_config(sample_environment_config) is True

    def test_environment_validation_invalid_dimensions(self):
        """Test environment validation with invalid dimensions."""
        invalid_config = EnvironmentConfig(
            width=0,  # Invalid
            height=16,
            n_mines=40,
            reward_win=10.0,
            reward_lose=-10.0,
            reward_reveal=1.0,
            reward_invalid=-1.0,
            max_reward_per_step=10.0
        )
        assert validate_environment_config(invalid_config) is False

    def test_environment_validation_too_many_mines(self):
        """Test environment validation with too many mines."""
        invalid_config = EnvironmentConfig(
            width=4,
            height=4,
            n_mines=16,  # Invalid: equal to total cells
            reward_win=10.0,
            reward_lose=-10.0,
            reward_reveal=1.0,
            reward_invalid=-1.0,
            max_reward_per_step=10.0
        )
        assert validate_environment_config(invalid_config) is False


class TestTrainingExecutionConfig:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            TrainingExecutionConfig()  # Should fail - no defaults

    def test_custom_training_config(self):
        """Test custom training execution configuration."""
        config = TrainingExecutionConfig(
            total_timesteps=2_000_000,
            n_envs=8,
            vec_env_type="dummy",
            checkpoint_freq=100000,
            device="cuda",
            seed=42
        )
        assert config.total_timesteps == 2_000_000
        assert config.n_envs == 8
        assert config.vec_env_type == "dummy"
        assert config.checkpoint_freq == 100000
        assert config.device == "cuda"
        assert config.seed == 42

    def test_sample_training_config(self, sample_training_execution):
        """Test sample training execution configuration."""
        assert sample_training_execution.total_timesteps == 1_000_000
        assert sample_training_execution.n_envs == 4
        assert sample_training_execution.vec_env_type == "subproc"
        assert sample_training_execution.device == "auto"


class TestPathsConfig:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            PathsConfig()  # Should fail - no defaults

    def test_custom_paths(self):
        """Test custom paths configuration."""
        paths = PathsConfig(
            experiment_base_dir="/custom/experiments",
            model_prefix="custom_model"
        )
        assert paths.experiment_base_dir == "/custom/experiments"
        assert paths.model_prefix == "custom_model"

    def test_sample_paths(self, sample_paths_config):
        """Test sample paths configuration."""
        assert sample_paths_config.experiment_base_dir == "experiments"
        assert sample_paths_config.model_prefix == "minesweeper_ppo"


class TestPlayConfig:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            PlayConfig()  # Should fail - no defaults

    def test_custom_play_config(self, sample_environment_config):
        """Test custom play configuration."""
        play_config = PlayConfig(
            mode="agent",
            num_episodes=50,
            delay=0.5,
            checkpoint_steps=100000,
            environment_config=sample_environment_config
        )
        assert play_config.mode == "agent"
        assert play_config.num_episodes == 50
        assert play_config.delay == 0.5
        assert play_config.checkpoint_steps == 100000
        assert play_config.environment_config == sample_environment_config

    def test_sample_play_config(self, sample_play_config):
        """Test sample play configuration."""
        assert sample_play_config.mode == "batch"
        assert sample_play_config.num_episodes == 100
        assert sample_play_config.delay == 0.1
        assert sample_play_config.checkpoint_steps is None


class TestTrainingConfig:
    def test_required_parameters(self):
        """Test that all parameters are required."""
        with pytest.raises(TypeError):
            TrainingConfig()  # Should fail - no defaults

    def test_custom_training_config(self,
                                   sample_model_hyperparams,
                                   sample_network_architecture,
                                   sample_training_execution,
                                   sample_paths_config):
        """Test custom complete training configuration."""
        custom_env = EnvironmentConfig(
            width=8,
            height=8,
            n_mines=10,
            reward_win=5.0,
            reward_lose=-5.0,
            reward_reveal=0.5,
            reward_invalid=-0.5,
            max_reward_per_step=5.0
        )
        
        config = TrainingConfig(
            model_hyperparams=sample_model_hyperparams,
            network_architecture=sample_network_architecture,
            environment_config=custom_env,
            training_execution=sample_training_execution,
            paths_config=sample_paths_config,
            play_config=None
        )
        
        assert config.model_hyperparams.learning_rate == 1e-4
        assert config.environment_config.width == 8
        assert config.training_execution.total_timesteps == 1_000_000
        assert config.play_config is None

    def test_to_dict_conversion(self,
                               sample_model_hyperparams,
                               sample_network_architecture,
                               sample_environment_config,
                               sample_training_execution,
                               sample_paths_config):
        """Test converting configuration to dictionary."""
        config = TrainingConfig(
            model_hyperparams=sample_model_hyperparams,
            network_architecture=sample_network_architecture,
            environment_config=sample_environment_config,
            training_execution=sample_training_execution,
            paths_config=sample_paths_config,
            play_config=None
        )
        
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert "model_hyperparams" in config_dict
        assert "environment_config" in config_dict
        assert "training_execution" in config_dict
        assert "paths_config" in config_dict
        
        # Test nested structure
        assert "learning_rate" in config_dict["model_hyperparams"]
        assert "width" in config_dict["environment_config"]

    def test_validation_valid_config(self,
                                    sample_model_hyperparams,
                                    sample_network_architecture,
                                    sample_environment_config,
                                    sample_training_execution,
                                    sample_paths_config):
        """Test validation of valid training configuration."""
        config = TrainingConfig(
            model_hyperparams=sample_model_hyperparams,
            network_architecture=sample_network_architecture,
            environment_config=sample_environment_config,
            training_execution=sample_training_execution,
            paths_config=sample_paths_config,
            play_config=None
        )
        
        assert validate_training_config(config) is True

    def test_validation_invalid_config(self,
                                      sample_network_architecture,
                                      sample_environment_config,
                                      sample_training_execution,
                                      sample_paths_config):
        """Test validation of invalid training configuration."""
        invalid_hyperparams = ModelHyperparams(
            learning_rate=-1e-4,  # Invalid: negative learning rate
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.90,
            clip_range=0.2,
            vf_coef=1.0,
            n_steps=1024,
            batch_size=128,
            n_epochs=10
        )
        
        config = TrainingConfig(
            model_hyperparams=invalid_hyperparams,
            network_architecture=sample_network_architecture,
            environment_config=sample_environment_config,
            training_execution=sample_training_execution,
            paths_config=sample_paths_config,
            play_config=None
        )
        
        assert validate_training_config(config) is False


class TestConfigCreation:
    def test_create_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "model_hyperparams": {
                "learning_rate": 2e-4,
                "ent_coef": 0.01,
                "gamma": 0.95,
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
                "width": 8,
                "height": 8,
                "n_mines": 10,
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
        
        config = create_config_from_dict(config_dict)
        
        assert isinstance(config, TrainingConfig)
        assert config.model_hyperparams.learning_rate == 2e-4
        assert config.environment_config.width == 8
        assert config.environment_config.height == 8
        assert config.training_execution.total_timesteps == 1_000_000

    def test_create_config_from_dict_with_play_config(self):
        """Test creating configuration from dictionary with play config."""
        config_dict = {
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
            },
            "play_config": {
                "mode": "agent",
                "num_episodes": 50,
                "delay": 0.5,
                "checkpoint_steps": None,
                "environment_config": {
                    "width": 8,
                    "height": 8,
                    "n_mines": 10,
                    "reward_win": 5.0,
                    "reward_lose": -5.0,
                    "reward_reveal": 0.5,
                    "reward_invalid": -0.5,
                    "max_reward_per_step": 5.0
                }
            }
        }
        
        config = create_config_from_dict(config_dict)
        
        assert isinstance(config, TrainingConfig)
        assert config.play_config is not None
        assert config.play_config.mode == "agent"
        assert config.play_config.environment_config.width == 8