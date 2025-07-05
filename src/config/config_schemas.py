"""
Configuration data classes for the minesweeper RL training system.

This module defines the configuration schemas used throughout the training and
playing pipeline. The configuration is organized into logical groups:
- Model hyperparameters
- Network architecture
- Environment configuration
- Training execution settings
- Paths and output configuration
- Play configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelHyperparams:
    """Model training hyperparameters for PPO algorithm."""
    learning_rate: float = 1e-4
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.90
    clip_range: float = 0.2
    vf_coef: float = 1.0
    n_steps: int = 1024
    batch_size: int = 128
    n_epochs: int = 10


@dataclass
class NetworkArchitecture:
    """Neural network architecture configuration."""
    features_dim: int = 128
    pi_layers: List[int] = field(default_factory=lambda: [64, 64])
    vf_layers: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class EnvironmentConfig:
    """Minesweeper environment configuration."""
    width: int = 16
    height: int = 16
    n_mines: int = 40
    reward_win: float = 10.0
    reward_lose: float = -10.0
    reward_reveal: float = 1.0
    reward_invalid: float = -1.0
    max_reward_per_step: float = 10.0


@dataclass
class TrainingExecutionConfig:
    """Training execution and system configuration."""
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    vec_env_type: str = "subproc"
    checkpoint_freq: int = 50000
    device: str = "auto"
    seed: Optional[int] = None


@dataclass
class PathsConfig:
    """Paths and file naming configuration."""
    experiment_base_dir: str = "experiments"
    model_prefix: str = "minesweeper_ppo"


@dataclass
class PlayConfig:
    """Play mode configuration."""
    mode: str = "batch"
    num_episodes: int = 100
    delay: float = 0.1
    checkpoint_steps: Optional[int] = None
    environment_config: Optional[EnvironmentConfig] = None


@dataclass
class TrainingConfig:
    """Complete training configuration container."""
    model_hyperparams: ModelHyperparams = field(default_factory=ModelHyperparams)
    network_architecture: NetworkArchitecture = field(default_factory=NetworkArchitecture)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training_execution: TrainingExecutionConfig = field(default_factory=TrainingExecutionConfig)
    paths_config: PathsConfig = field(default_factory=PathsConfig)
    play_config: Optional[PlayConfig] = None


# Utility functions for configuration schema operations

def validate_environment_config(config: EnvironmentConfig) -> bool:
    """
    Validate environment configuration parameters.
    
    Args:
        config: Environment configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if config.width <= 0 or config.height <= 0:
        return False
    if config.n_mines <= 0 or config.n_mines >= (config.width * config.height):
        return False
    return True


def validate_training_config(config: TrainingConfig) -> bool:
    """
    Validate complete training configuration.
    
    Args:
        config: Training configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    # Validate environment config
    if not validate_environment_config(config.environment_config):
        return False
    
    # Validate hyperparameters
    if config.model_hyperparams.learning_rate <= 0:
        return False
    if config.model_hyperparams.gamma < 0 or config.model_hyperparams.gamma > 1:
        return False
    if config.model_hyperparams.ent_coef < 0:
        return False
    
    # Validate training execution
    if config.training_execution.total_timesteps <= 0:
        return False
    if config.training_execution.n_envs <= 0:
        return False
    
    # Validate network architecture
    if config.network_architecture.features_dim <= 0:
        return False
    if not config.network_architecture.pi_layers or not config.network_architecture.vf_layers:
        return False
    
    return True


def create_config_from_dict(config_dict: dict) -> TrainingConfig:
    """
    Create TrainingConfig from dictionary representation.
    
    Args:
        config_dict: Dictionary containing configuration data
        
    Returns:
        TrainingConfig: Constructed configuration object
    """
    # Extract sub-configurations
    model_hyperparams_dict = config_dict.get("model_hyperparams", {})
    network_architecture_dict = config_dict.get("network_architecture", {})
    environment_config_dict = config_dict.get("environment_config", {})
    training_execution_dict = config_dict.get("training_execution", {})
    paths_config_dict = config_dict.get("paths_config", {})
    play_config_dict = config_dict.get("play_config", {})
    
    # Create components
    model_hyperparams = ModelHyperparams(**model_hyperparams_dict)
    network_architecture = NetworkArchitecture(**network_architecture_dict)
    environment_config = EnvironmentConfig(**environment_config_dict)
    training_execution = TrainingExecutionConfig(**training_execution_dict)
    paths_config = PathsConfig(**paths_config_dict)
    
    play_config = None
    if play_config_dict:
        # Handle nested environment config in play config
        play_env_config = play_config_dict.pop("environment_config", None)
        if play_env_config:
            play_config_dict["environment_config"] = EnvironmentConfig(**play_env_config)
        play_config = PlayConfig(**play_config_dict)
    
    return TrainingConfig(
        model_hyperparams=model_hyperparams,
        network_architecture=network_architecture,
        environment_config=environment_config,
        training_execution=training_execution,
        paths_config=paths_config,
        play_config=play_config
    )