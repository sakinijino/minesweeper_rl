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
from ..utils import config


class EnvironmentCreationError(Exception):
    """Custom exception for environment creation errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.__cause__ = cause


def create_env_config(
    width: Optional[int] = None,
    height: Optional[int] = None,
    n_mines: Optional[int] = None,
    reward_win: Optional[float] = None,
    reward_lose: Optional[float] = None,
    reward_reveal: Optional[float] = None,
    reward_invalid: Optional[float] = None,
    max_reward_per_step: Optional[float] = None,
    render_mode: Optional[str] = None,
    args: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Create environment configuration dictionary.
    
    Args:
        width: Width of the Minesweeper grid
        height: Height of the Minesweeper grid
        n_mines: Number of mines in the grid
        reward_win: Reward for winning the game
        reward_lose: Penalty for hitting a mine
        reward_reveal: Reward for revealing a safe cell
        reward_invalid: Penalty for clicking revealed cells
        max_reward_per_step: Maximum reward in one step
        render_mode: Rendering mode ('human', 'rgb_array', None)
        args: Arguments object to extract values from
        
    Returns:
        Dictionary containing environment configuration
    """
    # Use args object if provided, otherwise use explicit parameters or defaults
    if args is not None:
        env_config = {
            'width': getattr(args, 'width', config.WIDTH),
            'height': getattr(args, 'height', config.HEIGHT),
            'n_mines': getattr(args, 'n_mines', config.N_MINES),
            'reward_win': getattr(args, 'reward_win', config.REWARD_WIN),
            'reward_lose': getattr(args, 'reward_lose', config.REWARD_LOSE),
            'reward_reveal': getattr(args, 'reward_reveal', config.REWARD_REVEAL),
            'reward_invalid': getattr(args, 'reward_invalid', config.REWARD_INVALID),
            'max_reward_per_step': getattr(args, 'max_reward_per_step', config.MAX_REWARD_PER_STEP),
            'render_mode': render_mode
        }
    else:
        env_config = {
            'width': width if width is not None else config.WIDTH,
            'height': height if height is not None else config.HEIGHT,
            'n_mines': n_mines if n_mines is not None else config.N_MINES,
            'reward_win': reward_win if reward_win is not None else config.REWARD_WIN,
            'reward_lose': reward_lose if reward_lose is not None else config.REWARD_LOSE,
            'reward_reveal': reward_reveal if reward_reveal is not None else config.REWARD_REVEAL,
            'reward_invalid': reward_invalid if reward_invalid is not None else config.REWARD_INVALID,
            'max_reward_per_step': max_reward_per_step if max_reward_per_step is not None else config.MAX_REWARD_PER_STEP,
            'render_mode': render_mode
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
    args: Any,
    vecnormalize_stats_path: Optional[str] = None
) -> Any:
    """
    Create a training environment with VecNormalize.
    
    Args:
        args: Arguments object containing environment configuration
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        VecNormalize environment ready for training
        
    Raises:
        EnvironmentCreationError: If environment creation fails
    """
    try:
        # Create environment configuration
        env_config = create_env_config(args=args, render_mode=None)
        
        # Create environment function
        def create_env():
            return create_base_environment(env_config)
        
        # Create vectorized environment
        vec_env = create_vectorized_environment(
            env_fn=create_env,
            n_envs=args.n_envs,
            vec_env_type=args.vec_env_type,
            seed=args.seed
        )
        
        # Apply VecNormalize for training
        # Normalize rewards, but not observations (as observations are already normalized in env)
        normalized_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            gamma=args.gamma
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
        if isinstance(e, EnvironmentCreationError):
            raise
        else:
            raise EnvironmentCreationError(f"Failed to create training environment") from e


def create_inference_environment(
    args: Any,
    mode: str = 'batch',
    vecnormalize_stats_path: Optional[str] = None
) -> Tuple[Any, Optional[Any]]:
    """
    Create an inference environment.
    
    Args:
        args: Arguments object containing environment configuration
        mode: Inference mode ('batch' for no rendering, 'interactive' for human rendering)
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        Tuple of (vectorized_environment, raw_environment)
        raw_environment is None for batch mode, the actual env for interactive mode
        
    Raises:
        EnvironmentCreationError: If environment creation fails
    """
    try:
        if mode == 'interactive':
            # For interactive mode, create both raw and vectorized environments
            render_mode = 'human'
            env_config = create_env_config(args=args, render_mode=render_mode)
            
            # Create raw environment for direct access (e.g., cell_size, mouse clicks)
            raw_env = create_base_environment(env_config)
            
            # Wrap in DummyVecEnv for SB3 compatibility
            vec_env = DummyVecEnv([lambda: raw_env])
            
        else:
            # For batch mode, only need vectorized environment
            env_config = create_env_config(args=args, render_mode=None)
            
            def create_env():
                return create_base_environment(env_config)
            
            # Use DummyVecEnv with single environment for inference
            vec_env = create_vectorized_environment(
                env_fn=create_env,
                n_envs=1,
                vec_env_type='dummy',
                seed=args.seed
            )
            
            raw_env = None
        
        # Set random seed if specified
        if args.seed is not None:
            set_random_seed(args.seed)
            vec_env.seed(args.seed)
        
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
        if isinstance(e, EnvironmentCreationError):
            raise
        else:
            raise EnvironmentCreationError(f"Failed to create inference environment") from e


# Convenience functions for specific use cases

def create_batch_environment(args: Any, vecnormalize_stats_path: Optional[str] = None) -> Any:
    """
    Convenience function for creating batch inference environments.
    
    Args:
        args: Arguments object containing environment configuration
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        Vectorized environment ready for batch inference
    """
    env, _ = create_inference_environment(
        args=args,
        mode='batch',
        vecnormalize_stats_path=vecnormalize_stats_path
    )
    return env


def create_interactive_environment(args: Any, vecnormalize_stats_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Convenience function for creating interactive environments.
    
    Args:
        args: Arguments object containing environment configuration
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        Tuple of (vectorized_environment, raw_environment)
    """
    return create_inference_environment(
        args=args,
        mode='interactive',
        vecnormalize_stats_path=vecnormalize_stats_path
    )


def create_human_environment(args: Any, vecnormalize_stats_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Convenience function for creating human play environments.
    
    Args:
        args: Arguments object containing environment configuration
        vecnormalize_stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        Tuple of (vectorized_environment, raw_environment) for human interaction
    """
    vec_env, raw_env = create_interactive_environment(args, vecnormalize_stats_path)
    
    # For human mode, we might want to handle VecNormalize differently
    # Load stats but ensure proper settings for human play
    if vecnormalize_stats_path and os.path.exists(vecnormalize_stats_path):
        try:
            vec_env = VecNormalize.load(vecnormalize_stats_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        except Exception:
            # If loading fails, continue with unnormalized environment
            pass
    
    return vec_env, raw_env


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