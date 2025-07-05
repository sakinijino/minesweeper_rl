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
    learning_rate: float
    ent_coef: float
    gamma: float
    gae_lambda: float
    clip_range: float
    vf_coef: float
    n_steps: int
    batch_size: int
    n_epochs: int


@dataclass
class NetworkArchitecture:
    """Neural network architecture configuration."""
    features_dim: int
    pi_layers: List[int]
    vf_layers: List[int]


@dataclass
class EnvironmentConfig:
    """Minesweeper environment configuration."""
    width: int
    height: int
    n_mines: int
    reward_win: float
    reward_lose: float
    reward_reveal: float
    reward_invalid: float
    max_reward_per_step: float


@dataclass
class TrainingExecutionConfig:
    """Training execution and system configuration."""
    total_timesteps: int
    n_envs: int
    vec_env_type: str
    checkpoint_freq: int
    device: str
    seed: Optional[int]


@dataclass
class PathsConfig:
    """Paths and file naming configuration."""
    experiment_base_dir: str
    model_prefix: str


@dataclass
class PlayConfig:
    """Play mode configuration."""
    mode: str
    num_episodes: int
    delay: float
    checkpoint_steps: Optional[int]
    environment_config: Optional[EnvironmentConfig]


@dataclass
class TrainingConfig:
    """Complete training configuration container."""
    model_hyperparams: ModelHyperparams
    network_architecture: NetworkArchitecture
    environment_config: EnvironmentConfig
    training_execution: TrainingExecutionConfig
    paths_config: PathsConfig
    play_config: Optional[PlayConfig]


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