"""
Environment Factory Module

This module provides factory functions for creating Minesweeper environments with
consistent configuration. It centralizes environment creation logic to eliminate
code duplication between train.py and play.py.
"""

import os
from typing import Dict, Optional, Tuple, Any, Callable

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from ..env.minesweeper_env import MinesweeperEnv
from ..config.config_manager import ConfigManager
from ..config.config_schemas import DynamicEnvironmentConfig, EnvironmentConfig
from ..wrappers.curriculum_wrapper import CurriculumLearningWrapper, DynamicEnvironmentWrapper


class EnvironmentCreationError(Exception):
    """Custom exception for environment creation errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.__cause__ = cause


def create_env_config(
    config_manager: ConfigManager,
    render_mode: Optional[str] = None,
    render_fps: Optional[int] = None,
    board_size_override: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Create environment configuration dictionary from ConfigManager.
    
    Args:
        config_manager: ConfigManager instance containing all configuration
        render_mode: Rendering mode ('human', 'rgb_array', None)
        render_fps: Frame rate for rendering (defaults to mode-appropriate value)
        board_size_override: Optional override for board size (width, height, n_mines)
        
    Returns:
        Dictionary containing environment configuration
        
    Raises:
        ValueError: If config_manager is None or contains None values
    """
    if config_manager is None:
        raise ValueError("config_manager is required")
    
    env_config_obj = config_manager.config.environment_config
    
    # Determine appropriate FPS based on mode if not specified
    if render_fps is None:
        if render_mode == 'human':
            render_fps = 60  # High FPS for responsive human interaction
        elif render_mode == 'rgb_array':
            render_fps = 30  # Medium FPS for agent visualization
        else:
            render_fps = 30  # Default FPS for other cases
    
    # Get reward configuration
    reward_config = config_manager.get_reward_config()
    
    # Handle board size
    if board_size_override:
        # Use override values
        width = board_size_override['width']
        height = board_size_override['height']
        n_mines = board_size_override['n_mines']
    elif isinstance(env_config_obj, DynamicEnvironmentConfig):
        # For dynamic config, use the first available board size or fixed config
        if env_config_obj.fixed_config:
            width = env_config_obj.fixed_config.width
            height = env_config_obj.fixed_config.height
            n_mines = env_config_obj.fixed_config.n_mines
        elif env_config_obj.board_sizes:
            board_size = env_config_obj.board_sizes[0]
            width = board_size.width
            height = board_size.height
            n_mines = board_size.n_mines
        elif env_config_obj.curriculum and env_config_obj.curriculum.enabled:
            board_size = env_config_obj.curriculum.start_size
            width = board_size.width
            height = board_size.height
            n_mines = board_size.n_mines
        else:
            raise ValueError("DynamicEnvironmentConfig must have at least one board size configuration")
    else:
        # Traditional environment config
        width = env_config_obj.width
        height = env_config_obj.height
        n_mines = env_config_obj.n_mines
    
    env_config = {
        'width': width,
        'height': height,
        'n_mines': n_mines,
        'render_mode': render_mode,
        'render_fps': render_fps,
        **reward_config
    }
    
    return env_config


def create_base_environment(env_config: Dict[str, Any]) -> Any:
    """
    Create a base MinesweeperEnv instance.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        MinesweeperEnv instance
        
    Raises:
        EnvironmentCreationError: If environment creation fails
    """
    try:
        env = MinesweeperEnv(**env_config)
        return env
    except Exception as e:
        raise EnvironmentCreationError(f"Failed to create base environment") from e


def create_dynamic_environment(config_manager: ConfigManager, seed: Optional[int] = None) -> Any:
    """
    Create an environment with dynamic board size support.
    
    This function creates the appropriate environment based on the configuration:
    - Traditional fixed-size environment for EnvironmentConfig
    - Curriculum learning environment for DynamicEnvironmentConfig with curriculum
    - Multi-size sampling environment for DynamicEnvironmentConfig with board_sizes
    
    Args:
        config_manager: ConfigManager instance containing all configuration
        seed: Random seed for reproducibility
        
    Returns:
        Environment instance (potentially wrapped for dynamic behavior)
        
    Raises:
        EnvironmentCreationError: If environment creation fails
    """
    try:
        env_config_obj = config_manager.config.environment_config
        
        if isinstance(env_config_obj, DynamicEnvironmentConfig):
            # Handle dynamic environment configuration
            if env_config_obj.curriculum is not None and env_config_obj.curriculum.enabled:
                # Create curriculum learning environment
                print(f"Creating curriculum learning environment: {env_config_obj.curriculum.start_size.width}x{env_config_obj.curriculum.start_size.height} -> {env_config_obj.curriculum.end_size.width}x{env_config_obj.curriculum.end_size.height}")
                
                # Create initial base environment with start size
                initial_env_config = create_env_config(
                    config_manager=config_manager,
                    render_mode=None,
                    board_size_override={
                        'width': env_config_obj.curriculum.start_size.width,
                        'height': env_config_obj.curriculum.start_size.height,
                        'n_mines': env_config_obj.curriculum.start_size.n_mines
                    }
                )
                
                base_env = create_base_environment(initial_env_config)
                reward_config = config_manager.get_reward_config()
                
                # Wrap with curriculum learning
                wrapped_env = CurriculumLearningWrapper(
                    env=base_env,
                    curriculum_config=env_config_obj.curriculum,
                    reward_config=reward_config,
                    seed=seed
                )
                
                return wrapped_env
                
            elif env_config_obj.board_sizes is not None:
                # Create multi-size sampling environment
                print(f"Creating dynamic multi-size environment with {len(env_config_obj.board_sizes)} board sizes")
                
                # Create initial base environment with first board size
                initial_board_size = env_config_obj.board_sizes[0]
                initial_env_config = create_env_config(
                    config_manager=config_manager,
                    render_mode=None,
                    board_size_override={
                        'width': initial_board_size.width,
                        'height': initial_board_size.height,
                        'n_mines': initial_board_size.n_mines
                    }
                )
                
                base_env = create_base_environment(initial_env_config)
                reward_config = config_manager.get_reward_config()
                
                # Wrap with dynamic environment wrapper
                wrapped_env = DynamicEnvironmentWrapper(
                    env=base_env,
                    board_sizes=env_config_obj.board_sizes,
                    reward_config=reward_config,
                    sampling_weights=env_config_obj.sampling_weights,
                    seed=seed
                )
                
                return wrapped_env
                
            elif env_config_obj.fixed_config is not None:
                # Use fixed configuration from dynamic config
                print(f"Creating fixed-size environment from dynamic config: {env_config_obj.fixed_config.width}x{env_config_obj.fixed_config.height}")
                
                env_config = create_env_config(config_manager=config_manager, render_mode=None)
                return create_base_environment(env_config)
                
            else:
                raise ValueError("DynamicEnvironmentConfig must have at least one configuration type")
                
        else:
            # Traditional fixed-size environment
            print(f"Creating traditional fixed-size environment: {env_config_obj.width}x{env_config_obj.height}")
            
            env_config = create_env_config(config_manager=config_manager, render_mode=None)
            return create_base_environment(env_config)
            
    except Exception as e:
        if isinstance(e, (EnvironmentCreationError, ValueError)):
            raise
        else:
            raise EnvironmentCreationError(f"Failed to create dynamic environment") from e


def create_vectorized_environment(
    env_fn: Callable,
    n_envs: int = 1,
    vec_env_type: str = 'auto',
    seed: Optional[int] = None
) -> Any:
    """
    Create a vectorized environment.
    
    Args:
        env_fn: Function that creates a single environment instance
        n_envs: Number of parallel environments
        vec_env_type: Type of vectorized environment ('auto', 'subproc', 'dummy')
        seed: Random seed for environment
        
    Returns:
        Vectorized environment (DummyVecEnv or SubprocVecEnv)
    """
    # Determine vec_env_cls based on type and number of environments
    if vec_env_type == 'dummy' or n_envs == 1:
        # Use DummyVecEnv for single environment or explicit dummy request
        vec_env = DummyVecEnv([env_fn for _ in range(n_envs)])
    else:
        # Use make_vec_env for multiple environments with automatic type selection
        if vec_env_type == 'auto':
            # Auto-select based on number of environments
            vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        elif vec_env_type == 'subproc':
            vec_env_cls = SubprocVecEnv
        else:
            vec_env_cls = DummyVecEnv
        
        vec_env = make_vec_env(
            env_fn,
            n_envs=n_envs,
            seed=seed,
            vec_env_cls=vec_env_cls
        )
    
    return vec_env


def load_vecnormalize_stats(
    env: Any,
    stats_path: Optional[str],
    training_mode: bool = True,
    norm_reward: bool = True
) -> Any:
    """
    Load VecNormalize statistics if available.
    
    Args:
        env: Environment to load stats into
        stats_path: Path to VecNormalize stats file (optional)
        training_mode: Whether to set environment to training mode
        norm_reward: Whether to normalize rewards
        
    Returns:
        Environment with loaded stats, or original environment if stats not available
    """
    if stats_path is None or not os.path.exists(stats_path):
        return env
    
    try:
        # Load VecNormalize stats
        env_with_stats = VecNormalize.load(stats_path, env)
        
        # Configure training mode
        env_with_stats.training = training_mode
        env_with_stats.norm_reward = norm_reward
        
        return env_with_stats
    except Exception:
        # If loading fails, return original environment
        return env


def create_training_environment(
    config_manager: ConfigManager,
    vecnormalize_stats_path: Optional[str] = None
) -> Any:
    """
    Create a training environment with VecNormalize and dynamic board size support.
    
    Args:
        config_manager: ConfigManager instance containing all configuration
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        VecNormalize environment ready for training
        
    Raises:
        EnvironmentCreationError: If environment creation fails
        ValueError: If config_manager is None or incomplete
    """
    try:
        if config_manager is None:
            raise ValueError("config_manager is required")
        
        # Get training parameters from ConfigManager
        training_config = config_manager.config.training_execution
        n_envs = training_config.n_envs
        vec_env_type = training_config.vec_env_type
        seed = training_config.seed
        gamma = config_manager.config.model_hyperparams.gamma
        
        # Create environment function with dynamic board size support
        def create_env():
            return create_dynamic_environment(config_manager, seed=seed)
        
        # Create vectorized environment
        vec_env = create_vectorized_environment(
            env_fn=create_env,
            n_envs=n_envs,
            vec_env_type=vec_env_type,
            seed=seed
        )
        
        # Apply VecNormalize for training
        # Normalize rewards, but not observations (as observations are already normalized in env)
        normalized_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            gamma=gamma
        )
        
        # Load existing stats if provided
        if vecnormalize_stats_path:
            normalized_env = load_vecnormalize_stats(
                normalized_env,
                vecnormalize_stats_path,
                training_mode=True
            )
        
        return normalized_env
        
    except Exception as e:
        if isinstance(e, (EnvironmentCreationError, ValueError)):
            raise
        else:
            raise EnvironmentCreationError(f"Failed to create training environment") from e


def create_inference_environment(
    config_manager: ConfigManager,
    mode: str = 'batch',
    vecnormalize_stats_path: Optional[str] = None
) -> Tuple[Any, Optional[Any]]:
    """
    Create an inference environment.
    
    Args:
        config_manager: ConfigManager instance containing all configuration
        mode: Play mode ('human', 'agent', 'batch')
              - 'human'/'agent': Interactive mode with rendering for human/AI play
              - 'batch': Batch mode without rendering for AI evaluation
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        Tuple of (vectorized_environment, raw_environment)
        raw_environment is None for batch mode, the actual env for interactive modes
        
    Raises:
        EnvironmentCreationError: If environment creation fails
        ValueError: If config_manager is None
    """
    try:
        if config_manager is None:
            raise ValueError("config_manager is required")
        
        if mode in ['human', 'agent']:
            # For interactive modes (human/agent), create both raw and vectorized environments with rendering
            render_mode = 'human'
            env_config = create_env_config(config_manager=config_manager, render_mode=render_mode)
            
            # Create raw environment for direct access (e.g., cell_size, mouse clicks)
            raw_env = create_base_environment(env_config)
            
            # Wrap in DummyVecEnv for SB3 compatibility
            vec_env = DummyVecEnv([lambda: raw_env])
            
        else:
            # For batch mode, only need vectorized environment without rendering
            env_config = create_env_config(config_manager=config_manager, render_mode=None)
            
            def create_env():
                return create_base_environment(env_config)
            
            # Get seed from ConfigManager
            seed = config_manager.config.training_execution.seed
            
            # Use DummyVecEnv with single environment for inference
            vec_env = create_vectorized_environment(
                env_fn=create_env,
                n_envs=1,
                vec_env_type='dummy',
                seed=seed
            )
            
            raw_env = None
        
        # Set random seed if specified
        seed = config_manager.config.training_execution.seed
        if seed is not None:
            set_random_seed(seed)
            vec_env.seed(seed)
        
        # Load VecNormalize stats if provided
        if vecnormalize_stats_path:
            vec_env = load_vecnormalize_stats(
                vec_env,
                vecnormalize_stats_path,
                training_mode=False,
                norm_reward=False
            )
        
        return vec_env, raw_env
        
    except Exception as e:
        if isinstance(e, (EnvironmentCreationError, ValueError)):
            raise
        else:
            raise EnvironmentCreationError(f"Failed to create inference environment") from e




def get_environment_info(env: Any) -> Dict[str, Any]:
    """
    Get information about an environment.
    
    Args:
        env: Environment instance
        
    Returns:
        Dictionary containing environment information
    """
    info = {}
    
    # Get basic environment info
    if hasattr(env, 'observation_space'):
        info['observation_space'] = str(env.observation_space)
    if hasattr(env, 'action_space'):
        info['action_space'] = str(env.action_space)
    
    # Check if it's a vectorized environment
    if hasattr(env, 'num_envs'):
        info['num_envs'] = env.num_envs
        info['is_vectorized'] = True
    else:
        info['num_envs'] = 1
        info['is_vectorized'] = False
    
    # Check if it has VecNormalize
    if hasattr(env, 'training'):
        info['has_vecnormalize'] = True
        info['training_mode'] = env.training
        info['norm_reward'] = getattr(env, 'norm_reward', None)
        info['norm_obs'] = getattr(env, 'norm_obs', None)
    else:
        info['has_vecnormalize'] = False
    
    # Get environment type
    info['env_type'] = type(env).__name__
    
    return info