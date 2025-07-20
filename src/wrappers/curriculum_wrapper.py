"""
Curriculum Learning Environment Wrapper

This wrapper implements curriculum learning for minesweeper environments,
progressively increasing difficulty from small to large board sizes.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Discrete

from ..env.minesweeper_env import MinesweeperEnv
from ..config.config_schemas import (
    BoardSizeConfig, 
    CurriculumConfig, 
    DynamicEnvironmentConfig,
    EnvironmentConfig
)


class CurriculumLearningWrapper(Wrapper):
    """
    Curriculum learning wrapper that progressively increases board size difficulty.
    
    This wrapper dynamically creates new environments with increasing board sizes
    based on the agent's performance and training progress.
    """
    
    def __init__(
        self,
        env: MinesweeperEnv,
        curriculum_config: CurriculumConfig,
        reward_config: Dict[str, float],
        seed: Optional[int] = None
    ):
        """
        Initialize curriculum learning wrapper.
        
        Args:
            env: Base minesweeper environment (will be replaced during curriculum)
            curriculum_config: Curriculum configuration
            reward_config: Reward configuration dictionary
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        
        self.curriculum_config = curriculum_config
        self.reward_config = reward_config
        self.seed = seed
        
        # Curriculum state
        self.current_step = 0
        self.total_timesteps = 0
        self.episodes_completed = 0
        self.recent_wins = []
        self.current_board_size = curriculum_config.start_size
        
        # Performance tracking
        self.evaluation_wins = 0
        self.evaluation_episodes = 0
        self.last_evaluation_step = 0
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Create initial environment
        self._create_new_environment()
        
        # Update observation and action spaces to handle maximum possible size
        self._update_spaces()
    
    def _update_spaces(self):
        """Update observation and action spaces for the maximum possible environment size."""
        # Use the end size for space definition to ensure compatibility
        max_size = self.curriculum_config.end_size
        max_width = max_size.width
        max_height = max_size.height
        
        # Update observation space to accommodate the largest possible board
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(1, max_height, max_width),
            dtype=np.float32
        )
        
        # Update action space to accommodate the largest possible board
        self.action_space = Discrete(max_height * max_width)
    
    def _create_new_environment(self):
        """Create a new environment with the current board size."""
        env_config = {
            'width': self.current_board_size.width,
            'height': self.current_board_size.height,
            'n_mines': self.current_board_size.n_mines,
            'render_mode': self.env.render_mode,
            'render_fps': self.env.render_fps,
            **self.reward_config
        }
        
        # Create new environment
        new_env = MinesweeperEnv(**env_config)
        
        # Replace the wrapped environment
        self.env = new_env
        
        print(f"Curriculum: Advanced to board size {self.current_board_size.width}x{self.current_board_size.height} with {self.current_board_size.n_mines} mines")
    
    def _calculate_current_board_size(self) -> BoardSizeConfig:
        """Calculate the current board size based on curriculum progress."""
        if not self.curriculum_config.enabled:
            return self.curriculum_config.start_size
        
        # Calculate progression based on timesteps
        progress_by_time = min(
            self.total_timesteps / (self.curriculum_config.progression_steps * self.curriculum_config.step_duration),
            1.0
        )
        
        # Calculate progression based on performance
        progress_by_performance = self._calculate_performance_progress()
        
        # Use the minimum of both progressions (conservative approach)
        progress = min(progress_by_time, progress_by_performance)
        
        # Apply progression curve
        if self.curriculum_config.progression_type == "exponential":
            progress = progress ** 2
        elif self.curriculum_config.progression_type == "manual":
            # Manual progression based on explicit success thresholds
            progress = progress_by_performance
        
        # Interpolate between start and end sizes
        start_size = self.curriculum_config.start_size
        end_size = self.curriculum_config.end_size
        
        width = int(start_size.width + (end_size.width - start_size.width) * progress)
        height = int(start_size.height + (end_size.height - start_size.height) * progress)
        n_mines = int(start_size.n_mines + (end_size.n_mines - start_size.n_mines) * progress)
        
        return BoardSizeConfig(width=width, height=height, n_mines=n_mines)
    
    def _calculate_performance_progress(self) -> float:
        """Calculate curriculum progress based on agent performance."""
        if len(self.recent_wins) < self.curriculum_config.evaluation_episodes:
            return 0.0
        
        # Calculate recent win rate
        recent_win_rate = np.mean(self.recent_wins[-self.curriculum_config.evaluation_episodes:])
        
        # Progress based on success threshold
        if recent_win_rate >= self.curriculum_config.success_threshold:
            # Calculate how much above threshold we are
            excess = recent_win_rate - self.curriculum_config.success_threshold
            remaining = 1.0 - self.curriculum_config.success_threshold
            
            if remaining > 0:
                return min(1.0, excess / remaining)
            else:
                return 1.0
        
        return 0.0
    
    def _should_advance_curriculum(self) -> bool:
        """Check if curriculum should advance to the next level."""
        if not self.curriculum_config.enabled:
            return False
        
        # Check if enough time has passed
        if self.total_timesteps < self.last_evaluation_step + self.curriculum_config.step_duration:
            return False
        
        # Check if we have enough recent performance data
        if len(self.recent_wins) < self.curriculum_config.evaluation_episodes:
            return False
        
        # Check if performance threshold is met
        recent_win_rate = np.mean(self.recent_wins[-self.curriculum_config.evaluation_episodes:])
        
        return recent_win_rate >= self.curriculum_config.success_threshold
    
    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to match the maximum observation space size."""
        max_size = self.curriculum_config.end_size
        max_height = max_size.height
        max_width = max_size.width
        
        # Current observation shape: (1, current_height, current_width)
        current_height = obs.shape[1]
        current_width = obs.shape[2]
        
        if current_height == max_height and current_width == max_width:
            return obs
        
        # Create padded observation
        padded_obs = np.zeros((1, max_height, max_width), dtype=obs.dtype)
        
        # Copy current observation to top-left corner
        padded_obs[0, :current_height, :current_width] = obs[0]
        
        # Pad remaining areas with a recognizable pattern (e.g., -1 normalized to 0)
        # This ensures the agent learns to ignore padded regions
        
        return padded_obs
    
    def _adjust_action(self, action: int) -> int:
        """Adjust action from maximum action space to current environment action space."""
        max_width = self.curriculum_config.end_size.width
        current_width = self.current_board_size.width
        current_height = self.current_board_size.height
        
        # Convert action to row, col in maximum space
        row = action // max_width
        col = action % max_width
        
        # Check if action is within current environment bounds
        if row < current_height and col < current_width:
            # Convert to current environment action space
            return row * current_width + col
        else:
            # Action is outside current environment, return a random valid action
            return self.rng.randint(0, current_height * current_width)
    
    def reset(self, **kwargs):
        """Reset the environment, potentially advancing curriculum."""
        # Update curriculum if needed
        new_board_size = self._calculate_current_board_size()
        
        if (new_board_size.width != self.current_board_size.width or
            new_board_size.height != self.current_board_size.height or
            new_board_size.n_mines != self.current_board_size.n_mines):
            
            self.current_board_size = new_board_size
            self._create_new_environment()
        
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        
        # Pad observation to match maximum space
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, info
    
    def step(self, action: int):
        """Step the environment with curriculum learning."""
        self.total_timesteps += 1
        
        # Adjust action for current environment size
        adjusted_action = self._adjust_action(action)
        
        # Take step in current environment
        obs, reward, terminated, truncated, info = self.env.step(adjusted_action)
        
        # Track performance
        if terminated or truncated:
            self.episodes_completed += 1
            is_win = info.get('is_success', False)
            self.recent_wins.append(1.0 if is_win else 0.0)
            
            # Keep only recent episodes for evaluation
            if len(self.recent_wins) > self.curriculum_config.evaluation_episodes * 2:
                self.recent_wins = self.recent_wins[-self.curriculum_config.evaluation_episodes:]
        
        # Pad observation to match maximum space
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, reward, terminated, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """Get action masks for the current environment, padded to maximum size."""
        # Get masks from current environment
        current_masks = self.env.action_masks()
        
        max_size = self.curriculum_config.end_size
        max_actions = max_size.height * max_size.width
        
        # Create padded mask
        padded_masks = np.zeros(max_actions, dtype=bool)
        
        # Map current environment masks to padded space
        current_height = self.current_board_size.height
        current_width = self.current_board_size.width
        max_width = max_size.width
        
        for current_action in range(len(current_masks)):
            if current_masks[current_action]:
                # Convert current action to row, col
                row = current_action // current_width
                col = current_action % current_width
                
                # Convert to padded action space
                padded_action = row * max_width + col
                padded_masks[padded_action] = True
        
        return padded_masks
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get information about current curriculum state."""
        recent_win_rate = np.mean(self.recent_wins[-self.curriculum_config.evaluation_episodes:]) if len(self.recent_wins) >= self.curriculum_config.evaluation_episodes else 0.0
        
        return {
            'current_board_size': f"{self.current_board_size.width}x{self.current_board_size.height}",
            'current_mines': self.current_board_size.n_mines,
            'total_timesteps': self.total_timesteps,
            'episodes_completed': self.episodes_completed,
            'recent_win_rate': recent_win_rate,
            'evaluation_episodes': len(self.recent_wins),
            'curriculum_progress': self._calculate_performance_progress()
        }


class DynamicEnvironmentWrapper(Wrapper):
    """
    Dynamic environment wrapper that can randomly sample from different board sizes.
    
    This wrapper is useful for training on multiple board sizes simultaneously
    without curriculum learning progression.
    """
    
    def __init__(
        self,
        env: MinesweeperEnv,
        board_sizes: List[BoardSizeConfig],
        reward_config: Dict[str, float],
        sampling_weights: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize dynamic environment wrapper.
        
        Args:
            env: Base minesweeper environment
            board_sizes: List of possible board sizes
            reward_config: Reward configuration dictionary
            sampling_weights: Optional weights for sampling board sizes
            seed: Random seed for reproducibility
        """
        super().__init__(env)
        
        self.board_sizes = board_sizes
        self.reward_config = reward_config
        self.sampling_weights = sampling_weights
        self.seed = seed
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Current board size
        self.current_board_size = self._sample_board_size()
        
        # Update spaces for maximum possible size
        self._update_spaces()
        
        # Create initial environment
        self._create_new_environment()
    
    def _update_spaces(self):
        """Update observation and action spaces for the maximum possible environment size."""
        # Find maximum dimensions
        max_width = max(size.width for size in self.board_sizes)
        max_height = max(size.height for size in self.board_sizes)
        
        # Update observation space
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(1, max_height, max_width),
            dtype=np.float32
        )
        
        # Update action space
        self.action_space = Discrete(max_height * max_width)
    
    def _sample_board_size(self) -> BoardSizeConfig:
        """Sample a board size from the available options."""
        if self.sampling_weights is not None:
            # Weighted sampling
            idx = self.rng.choice(len(self.board_sizes), p=self.sampling_weights)
        else:
            # Uniform sampling
            idx = self.rng.choice(len(self.board_sizes))
        
        return self.board_sizes[idx]
    
    def _create_new_environment(self):
        """Create a new environment with the current board size."""
        env_config = {
            'width': self.current_board_size.width,
            'height': self.current_board_size.height,
            'n_mines': self.current_board_size.n_mines,
            'render_mode': self.env.render_mode,
            'render_fps': self.env.render_fps,
            **self.reward_config
        }
        
        # Create new environment
        new_env = MinesweeperEnv(**env_config)
        
        # Replace the wrapped environment
        self.env = new_env
    
    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to match the maximum observation space size."""
        max_width = max(size.width for size in self.board_sizes)
        max_height = max(size.height for size in self.board_sizes)
        
        # Current observation shape: (1, current_height, current_width)
        current_height = obs.shape[1]
        current_width = obs.shape[2]
        
        if current_height == max_height and current_width == max_width:
            return obs
        
        # Create padded observation
        padded_obs = np.zeros((1, max_height, max_width), dtype=obs.dtype)
        
        # Copy current observation to top-left corner
        padded_obs[0, :current_height, :current_width] = obs[0]
        
        return padded_obs
    
    def _adjust_action(self, action: int) -> int:
        """Adjust action from maximum action space to current environment action space."""
        max_width = max(size.width for size in self.board_sizes)
        current_width = self.current_board_size.width
        current_height = self.current_board_size.height
        
        # Convert action to row, col in maximum space
        row = action // max_width
        col = action % max_width
        
        # Check if action is within current environment bounds
        if row < current_height and col < current_width:
            # Convert to current environment action space
            return row * current_width + col
        else:
            # Action is outside current environment, return a random valid action
            return self.rng.randint(0, current_height * current_width)
    
    def reset(self, **kwargs):
        """Reset the environment, potentially sampling a new board size."""
        # Sample new board size for each episode
        self.current_board_size = self._sample_board_size()
        self._create_new_environment()
        
        # Reset environment
        obs, info = self.env.reset(**kwargs)
        
        # Pad observation to match maximum space
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, info
    
    def step(self, action: int):
        """Step the environment with dynamic board sizing."""
        # Adjust action for current environment size
        adjusted_action = self._adjust_action(action)
        
        # Take step in current environment
        obs, reward, terminated, truncated, info = self.env.step(adjusted_action)
        
        # Pad observation to match maximum space
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, reward, terminated, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """Get action masks for the current environment, padded to maximum size."""
        # Get masks from current environment
        current_masks = self.env.action_masks()
        
        max_width = max(size.width for size in self.board_sizes)
        max_height = max(size.height for size in self.board_sizes)
        max_actions = max_height * max_width
        
        # Create padded mask
        padded_masks = np.zeros(max_actions, dtype=bool)
        
        # Map current environment masks to padded space
        current_height = self.current_board_size.height
        current_width = self.current_board_size.width
        
        for current_action in range(len(current_masks)):
            if current_masks[current_action]:
                # Convert current action to row, col
                row = current_action // current_width
                col = current_action % current_width
                
                # Convert to padded action space
                padded_action = row * max_width + col
                padded_masks[padded_action] = True
        
        return padded_masks