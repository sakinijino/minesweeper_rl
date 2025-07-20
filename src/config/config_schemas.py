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
    max_reward_per_step: Optional[float] = None


@dataclass
class BoardSizeConfig:
    """Configuration for a specific board size."""
    width: int
    height: int
    n_mines: int
    mine_density: Optional[float] = None  # Alternative to n_mines, calculated as percentage
    
    def __post_init__(self):
        """Validate and compute mine density if needed."""
        if self.mine_density is not None and self.n_mines is None:
            # Calculate n_mines from density
            total_cells = self.width * self.height
            self.n_mines = max(1, int(total_cells * self.mine_density))
        elif self.mine_density is None and self.n_mines is not None:
            # Calculate density from n_mines
            total_cells = self.width * self.height
            self.mine_density = self.n_mines / total_cells if total_cells > 0 else 0.0


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning progression."""
    enabled: bool = False
    progression_type: str = "linear"  # "linear", "exponential", "manual"
    start_size: BoardSizeConfig = None
    end_size: BoardSizeConfig = None
    progression_steps: int = 10
    step_duration: int = 50000  # timesteps per curriculum step
    success_threshold: float = 0.7  # win rate threshold to advance
    evaluation_episodes: int = 100  # episodes to evaluate success rate
    
    def __post_init__(self):
        """Set default start and end sizes if not provided."""
        if self.enabled and self.start_size is None:
            self.start_size = BoardSizeConfig(width=5, height=5, n_mines=3)
        if self.enabled and self.end_size is None:
            self.end_size = BoardSizeConfig(width=10, height=10, n_mines=15)


@dataclass
class DynamicEnvironmentConfig:
    """Dynamic environment configuration supporting variable board sizes."""
    # Fixed configuration (backward compatibility)
    fixed_config: Optional[EnvironmentConfig] = None
    
    # Dynamic configuration options
    board_sizes: Optional[List[BoardSizeConfig]] = None
    curriculum: Optional[CurriculumConfig] = None
    
    # Reward configuration (shared across all board sizes)
    reward_win: float = 1.0
    reward_lose: float = -1.0
    reward_reveal: float = 0.1
    reward_invalid: float = -0.1
    max_reward_per_step: Optional[float] = None
    
    # Sampling configuration
    random_sampling: bool = False
    sampling_weights: Optional[List[float]] = None  # Weights for board size sampling
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Ensure we have at least one configuration type
        if self.fixed_config is None and self.board_sizes is None:
            # Default to simple 5x5 board
            self.fixed_config = EnvironmentConfig(
                width=5, height=5, n_mines=3,
                reward_win=self.reward_win, reward_lose=self.reward_lose,
                reward_reveal=self.reward_reveal, reward_invalid=self.reward_invalid,
                max_reward_per_step=self.max_reward_per_step
            )
        
        # Validate sampling weights
        if self.sampling_weights is not None and self.board_sizes is not None:
            if len(self.sampling_weights) != len(self.board_sizes):
                raise ValueError("Number of sampling weights must match number of board sizes")
    
    def is_dynamic(self) -> bool:
        """Check if this configuration supports dynamic board sizes."""
        return self.board_sizes is not None or (self.curriculum is not None and self.curriculum.enabled)
    
    def get_board_sizes(self) -> List[BoardSizeConfig]:
        """Get all possible board sizes for this configuration."""
        if self.board_sizes is not None:
            return self.board_sizes
        elif self.curriculum is not None and self.curriculum.enabled:
            return [self.curriculum.start_size, self.curriculum.end_size]
        elif self.fixed_config is not None:
            return [BoardSizeConfig(
                width=self.fixed_config.width,
                height=self.fixed_config.height,
                n_mines=self.fixed_config.n_mines
            )]
        return []


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
    environment_config: Union[EnvironmentConfig, DynamicEnvironmentConfig]
    training_execution: TrainingExecutionConfig
    paths_config: PathsConfig


# Utility functions for configuration schema operations

def validate_board_size_config(config: BoardSizeConfig) -> bool:
    """
    Validate board size configuration parameters.
    
    Args:
        config: Board size configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if config.width <= 0 or config.height <= 0:
        return False
    if config.n_mines <= 0 or config.n_mines >= (config.width * config.height):
        return False
    return True


def validate_environment_config(config: Union[EnvironmentConfig, DynamicEnvironmentConfig]) -> bool:
    """
    Validate environment configuration parameters.
    
    Args:
        config: Environment configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if isinstance(config, EnvironmentConfig):
        # Validate traditional environment config
        if config.width <= 0 or config.height <= 0:
            return False
        if config.n_mines <= 0 or config.n_mines >= (config.width * config.height):
            return False
        return True
    
    elif isinstance(config, DynamicEnvironmentConfig):
        # Validate dynamic environment config
        if config.fixed_config is not None:
            if not validate_environment_config(config.fixed_config):
                return False
        
        if config.board_sizes is not None:
            for board_config in config.board_sizes:
                if not validate_board_size_config(board_config):
                    return False
        
        if config.curriculum is not None and config.curriculum.enabled:
            if not validate_board_size_config(config.curriculum.start_size):
                return False
            if not validate_board_size_config(config.curriculum.end_size):
                return False
        
        return True
    
    return False


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
    dynamic_environment_config_dict = config_dict.get("dynamic_environment_config", {})
    
    # Create components
    model_hyperparams = ModelHyperparams(**model_hyperparams_dict)
    network_architecture = NetworkArchitecture(**network_architecture_dict)
    training_execution = TrainingExecutionConfig(**training_execution_dict)
    paths_config = PathsConfig(**paths_config_dict)
    
    # Determine which environment config to use
    if dynamic_environment_config_dict:
        # Create dynamic environment config
        environment_config = _create_dynamic_environment_config_from_dict(dynamic_environment_config_dict)
    else:
        # Create traditional environment config (backward compatibility)
        environment_config = EnvironmentConfig(**environment_config_dict)
    
    return TrainingConfig(
        model_hyperparams=model_hyperparams,
        network_architecture=network_architecture,
        environment_config=environment_config,
        training_execution=training_execution,
        paths_config=paths_config
    )


def _create_dynamic_environment_config_from_dict(config_dict: dict) -> DynamicEnvironmentConfig:
    """
    Create DynamicEnvironmentConfig from dictionary representation.
    
    Args:
        config_dict: Dictionary containing dynamic environment configuration data
        
    Returns:
        DynamicEnvironmentConfig: Constructed configuration object
    """
    # Extract sub-configurations
    fixed_config_dict = config_dict.get("fixed_config")
    board_sizes_dict = config_dict.get("board_sizes")
    curriculum_dict = config_dict.get("curriculum")
    
    # Create fixed config if provided
    fixed_config = None
    if fixed_config_dict:
        fixed_config = EnvironmentConfig(**fixed_config_dict)
    
    # Create board sizes if provided
    board_sizes = None
    if board_sizes_dict:
        board_sizes = []
        for board_dict in board_sizes_dict:
            board_sizes.append(BoardSizeConfig(**board_dict))
    
    # Create curriculum config if provided
    curriculum = None
    if curriculum_dict:
        start_size_dict = curriculum_dict.get("start_size")
        end_size_dict = curriculum_dict.get("end_size")
        
        start_size = None
        if start_size_dict:
            start_size = BoardSizeConfig(**start_size_dict)
        
        end_size = None
        if end_size_dict:
            end_size = BoardSizeConfig(**end_size_dict)
        
        curriculum = CurriculumConfig(
            enabled=curriculum_dict.get("enabled", False),
            progression_type=curriculum_dict.get("progression_type", "linear"),
            start_size=start_size,
            end_size=end_size,
            progression_steps=curriculum_dict.get("progression_steps", 10),
            step_duration=curriculum_dict.get("step_duration", 50000),
            success_threshold=curriculum_dict.get("success_threshold", 0.7),
            evaluation_episodes=curriculum_dict.get("evaluation_episodes", 100)
        )
    
    return DynamicEnvironmentConfig(
        fixed_config=fixed_config,
        board_sizes=board_sizes,
        curriculum=curriculum,
        reward_win=config_dict.get("reward_win", 1.0),
        reward_lose=config_dict.get("reward_lose", -1.0),
        reward_reveal=config_dict.get("reward_reveal", 0.1),
        reward_invalid=config_dict.get("reward_invalid", -0.1),
        max_reward_per_step=config_dict.get("max_reward_per_step"),
        random_sampling=config_dict.get("random_sampling", False),
        sampling_weights=config_dict.get("sampling_weights")
    )