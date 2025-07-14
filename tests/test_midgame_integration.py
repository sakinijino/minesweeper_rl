"""
Integration tests for MidGame wrapper automatic application
"""
import pytest
from unittest.mock import Mock
from src.config.config_manager import ConfigManager
from src.factories.environment_factory import (
    create_training_environment,
    should_use_midgame_wrapper,
    apply_midgame_wrapper
)
from src.env.minesweeper_env import MinesweeperEnv
from src.wrappers.midgame_wrapper import MidGameWrapper


class TestMidGameIntegration:
    """Test automatic MidGame wrapper integration"""
    
    def test_should_use_midgame_wrapper_conditions(self):
        """Test the conditions for applying MidGameWrapper"""
        # Create mock config manager with different timestep values
        
        # Short training - should not use wrapper
        config_manager = Mock()
        config_manager.config.training_execution.total_timesteps = 50000
        assert not should_use_midgame_wrapper(config_manager)
        
        # Medium training - should use wrapper
        config_manager.config.training_execution.total_timesteps = 200000
        assert should_use_midgame_wrapper(config_manager)
        
        # Long training - should use wrapper
        config_manager.config.training_execution.total_timesteps = 1000000
        assert should_use_midgame_wrapper(config_manager)
    
    def test_apply_midgame_wrapper_parameters(self):
        """Test that MidGameWrapper is applied with appropriate parameters"""
        # Small environment
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        config_manager = Mock()
        config_manager.config.training_execution.seed = 42
        
        wrapped_env = apply_midgame_wrapper(env, config_manager)
        
        assert isinstance(wrapped_env, MidGameWrapper)
        assert wrapped_env.env is env
        assert wrapped_env.midgame_probability == 0.2  # Small board setting
        assert wrapped_env.min_revealed_cells == 2
    
    def test_training_environment_with_midgame_wrapper(self):
        """Test that training environment automatically gets wrapper for long training"""
        # Create a mock config manager that should trigger wrapper
        config_manager = Mock()
        config_manager.config.training_execution.total_timesteps = 500000
        config_manager.config.training_execution.n_envs = 1
        config_manager.config.training_execution.vec_env_type = 'dummy'
        config_manager.config.training_execution.seed = 42
        config_manager.config.environment_config.width = 5
        config_manager.config.environment_config.height = 5
        config_manager.config.environment_config.n_mines = 3
        config_manager.config.environment_config.reward_win = 1.0
        config_manager.config.environment_config.reward_lose = -1.0
        config_manager.config.environment_config.reward_reveal = 0.1
        config_manager.config.environment_config.reward_invalid = -0.1
        config_manager.config.environment_config.max_reward_per_step = None
        config_manager.config.model_hyperparams.gamma = 0.99
        
        # Create training environment
        env = create_training_environment(config_manager)
        
        # The actual environment should be wrapped
        # We need to dig through VecNormalize -> DummyVecEnv to get to the base env
        base_env = env.venv.envs[0]
        
        # Should be wrapped with MidGameWrapper
        assert isinstance(base_env, MidGameWrapper)
        assert isinstance(base_env.env, MinesweeperEnv)
        
        env.close()
    
    def test_training_environment_without_midgame_wrapper(self):
        """Test that training environment doesn't get wrapper for short training"""
        # Create a mock config manager that should NOT trigger wrapper
        config_manager = Mock()
        config_manager.config.training_execution.total_timesteps = 50000  # Short training
        config_manager.config.training_execution.n_envs = 1
        config_manager.config.training_execution.vec_env_type = 'dummy'
        config_manager.config.training_execution.seed = 42
        config_manager.config.environment_config.width = 5
        config_manager.config.environment_config.height = 5
        config_manager.config.environment_config.n_mines = 3
        config_manager.config.environment_config.reward_win = 1.0
        config_manager.config.environment_config.reward_lose = -1.0
        config_manager.config.environment_config.reward_reveal = 0.1
        config_manager.config.environment_config.reward_invalid = -0.1
        config_manager.config.environment_config.max_reward_per_step = None
        config_manager.config.model_hyperparams.gamma = 0.99
        
        # Create training environment
        env = create_training_environment(config_manager)
        
        # The actual environment should NOT be wrapped
        base_env = env.venv.envs[0]
        
        # Should be bare MinesweeperEnv, not wrapped
        assert isinstance(base_env, MinesweeperEnv)
        assert not isinstance(base_env, MidGameWrapper)
        
        env.close()
    
    def test_different_board_sizes_get_appropriate_parameters(self):
        """Test that different board sizes get appropriate wrapper parameters"""
        config_manager = Mock()
        config_manager.config.training_execution.seed = 42
        
        # Small board (5x5)
        small_env = MinesweeperEnv(width=5, height=5, n_mines=3)
        small_wrapped = apply_midgame_wrapper(small_env, config_manager)
        assert small_wrapped.midgame_probability == 0.2
        assert small_wrapped.min_revealed_cells == 2
        
        # Medium board (8x8)
        medium_env = MinesweeperEnv(width=8, height=8, n_mines=10)
        medium_wrapped = apply_midgame_wrapper(medium_env, config_manager)
        assert medium_wrapped.midgame_probability == 0.3
        assert medium_wrapped.min_revealed_cells == 3
        
        # Large board (12x12)
        large_env = MinesweeperEnv(width=12, height=12, n_mines=20)
        large_wrapped = apply_midgame_wrapper(large_env, config_manager)
        assert large_wrapped.midgame_probability == 0.4
        assert large_wrapped.min_revealed_cells == 5
    
    def test_midgame_wrapper_preserves_functionality(self):
        """Test that wrapped environment still works correctly"""
        # Create a mock config manager that triggers wrapper
        config_manager = Mock()
        config_manager.config.training_execution.total_timesteps = 500000
        config_manager.config.training_execution.n_envs = 1
        config_manager.config.training_execution.vec_env_type = 'dummy'
        config_manager.config.training_execution.seed = 42
        config_manager.config.environment_config.width = 5
        config_manager.config.environment_config.height = 5
        config_manager.config.environment_config.n_mines = 3
        config_manager.config.environment_config.reward_win = 1.0
        config_manager.config.environment_config.reward_lose = -1.0
        config_manager.config.environment_config.reward_reveal = 0.1
        config_manager.config.environment_config.reward_invalid = -0.1
        config_manager.config.environment_config.max_reward_per_step = None
        config_manager.config.model_hyperparams.gamma = 0.99
        
        # Create training environment with wrapper
        env = create_training_environment(config_manager)
        
        # Test basic functionality
        obs = env.reset()
        assert obs[0].shape == (1, 5, 5)
        
        # Test step
        action = [0]  # Vectorized environments expect lists
        obs, reward, done, info = env.step(action)
        assert obs[0].shape == (1, 5, 5)
        # Accept numpy types
        assert isinstance(reward[0], (int, float)) or hasattr(reward[0], 'dtype')
        assert isinstance(done[0], (bool, type(True))) or hasattr(done[0], 'dtype')
        assert isinstance(info[0], dict)
        
        env.close()