"""
MidGameWrapper - Environment wrapper for mid-game state training

This wrapper allows the environment to occasionally start from mid-game states
instead of always starting from the initial state. This enables more efficient
learning of mid to late-game strategies.
"""

import numpy as np
import random
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces


class MidGameWrapper(gym.Wrapper):
    """
    Environment wrapper that provides mid-game states for curriculum learning.
    
    This wrapper intercepts the reset() call and with a certain probability
    generates a mid-game state instead of starting fresh. The probability
    can be static or dynamic based on training progress.
    """
    
    def __init__(
        self,
        env: gym.Env,
        midgame_probability: float = 0.3,
        min_revealed_cells: int = 3,
        max_revealed_cells: Optional[int] = None,
        safe_first_moves: int = 2,
        seed: Optional[int] = None
    ):
        """
        Initialize the MidGameWrapper.
        
        Args:
            env: The base MinesweeperEnv to wrap
            midgame_probability: Probability of starting from mid-game (0.0 to 1.0)
            min_revealed_cells: Minimum cells to reveal in mid-game states
            max_revealed_cells: Maximum cells to reveal (None = adaptive based on board)
            safe_first_moves: Number of guaranteed safe moves to make
            seed: Random seed for reproducible mid-game generation
        """
        super().__init__(env)
        
        self.midgame_probability = midgame_probability
        self.min_revealed_cells = min_revealed_cells
        self.safe_first_moves = safe_first_moves
        
        # Adaptive max cells based on board size
        total_cells = env.width * env.height
        safe_cells = total_cells - env.n_mines
        self.max_revealed_cells = max_revealed_cells or min(safe_cells // 2, 15)
        
        # Random state for reproducible mid-game generation
        self._rng = np.random.RandomState(seed)
        
        # Statistics tracking
        self.stats = {
            'total_resets': 0,
            'midgame_resets': 0,
            'fresh_resets': 0,
            'cells_revealed_avg': 0.0
        }
    
    def set_midgame_probability(self, probability: float):
        """Update the mid-game probability dynamically."""
        self.midgame_probability = max(0.0, min(1.0, probability))
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment, potentially to a mid-game state.
        
        Returns:
            Tuple of (observation, info)
        """
        self.stats['total_resets'] += 1
        
        # Always start with a fresh reset
        obs, info = self.env.reset(**kwargs)
        
        # Decide whether to generate mid-game state
        if self._rng.random() < self.midgame_probability:
            try:
                obs, info = self._generate_midgame_state()
                self.stats['midgame_resets'] += 1
            except Exception as e:
                # If mid-game generation fails, use fresh state
                print(f"Warning: Mid-game generation failed: {e}")
                self.stats['fresh_resets'] += 1
        else:
            self.stats['fresh_resets'] += 1
        
        return obs, info
    
    def action_masks(self) -> np.ndarray:
        """Forward action_masks call to the wrapped environment."""
        return self.env.action_masks()
    
    def _generate_midgame_state(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate a valid mid-game state by making safe moves.
        
        Returns:
            Tuple of (observation, info) for the mid-game state
        """
        # Start with fresh state
        self.env.reset()
        
        # Make a few guaranteed safe moves first
        safe_moves_made = 0
        revealed_cells = 0
        
        # Get all possible actions
        available_actions = list(range(self.env.action_space.n))
        
        # Target number of cells to reveal
        target_cells = self._rng.randint(
            self.min_revealed_cells,
            self.max_revealed_cells + 1
        )
        
        # Make safe moves - be more careful about mine avoidance
        while safe_moves_made < self.safe_first_moves and available_actions:
            # Randomly select an action
            action_idx = self._rng.choice(len(available_actions))
            action = available_actions.pop(action_idx)
            
            row, col = np.unravel_index(action, (self.env.height, self.env.width))
            
            # Skip if already revealed
            if self.env.revealed[row, col]:
                continue
            
            # For the first few moves, avoid mines if known
            if safe_moves_made < 2 and self.env.mines[row, col]:
                continue
            
            # Make the move
            obs, reward, terminated, truncated, info = self.env.step(action)
            safe_moves_made += 1
            revealed_cells = info['revealed_cells']
            
            # If we've revealed enough cells or game ended, stop
            if revealed_cells >= target_cells or terminated:
                break
        
        # Continue making random safe moves until target is reached
        attempts = 0
        max_attempts = 50  # Prevent infinite loops
        
        while revealed_cells < target_cells and not self.env.game_over and attempts < max_attempts:
            attempts += 1
            
            # Get valid non-mine actions (unrevealed, safe cells)
            valid_safe_actions = []
            for action in range(self.env.action_space.n):
                row, col = np.unravel_index(action, (self.env.height, self.env.width))
                if not self.env.revealed[row, col] and not self.env.mines[row, col]:
                    valid_safe_actions.append(action)
            
            if not valid_safe_actions:
                break
            
            # Select random valid safe action
            action = self._rng.choice(valid_safe_actions)
            
            # Make the move
            obs, reward, terminated, truncated, info = self.env.step(action)
            revealed_cells = info['revealed_cells']
            
            # Stop if we've reached our target or exceeded it
            if revealed_cells >= target_cells or terminated:
                break
        
        # Update statistics
        if revealed_cells > 0 and self.stats['midgame_resets'] > 0:
            # Calculate running average safely
            if self.stats['midgame_resets'] == 1:
                self.stats['cells_revealed_avg'] = float(revealed_cells)
            else:
                self.stats['cells_revealed_avg'] = (
                    (self.stats['cells_revealed_avg'] * (self.stats['midgame_resets'] - 1) + revealed_cells) /
                    self.stats['midgame_resets']
                )
        
        return self.env._get_obs(), self.env._get_info()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        stats = self.stats.copy()
        if stats['total_resets'] > 0:
            stats['midgame_ratio'] = stats['midgame_resets'] / stats['total_resets']
        else:
            stats['midgame_ratio'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset wrapper statistics."""
        self.stats = {
            'total_resets': 0,
            'midgame_resets': 0,
            'fresh_resets': 0,
            'cells_revealed_avg': 0.0
        }