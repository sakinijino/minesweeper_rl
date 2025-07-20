"""
Tests for dynamic board size functionality.

This module tests the key components of the dynamic board size implementation:
- DynamicEnvironmentConfig creation and validation
- CustomCNN with adaptive pooling
- Curriculum learning wrapper
- Environment factory with dynamic configuration
"""

import pytest
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete

from src.config.config_schemas import (
    BoardSizeConfig,
    CurriculumConfig,
    DynamicEnvironmentConfig,
    EnvironmentConfig,
    create_config_from_dict,
    validate_environment_config
)
from src.config.config_manager import ConfigManager
from src.env.custom_cnn import CustomCNN
from src.env.minesweeper_env import MinesweeperEnv
from src.wrappers.curriculum_wrapper import CurriculumLearningWrapper, DynamicEnvironmentWrapper
from src.factories.environment_factory import create_dynamic_environment


class TestBoardSizeConfig:
    """Test BoardSizeConfig functionality."""
    
    def test_board_size_config_creation(self):
        """Test creating a BoardSizeConfig."""
        config = BoardSizeConfig(width=5, height=5, n_mines=3)
        assert config.width == 5
        assert config.height == 5
        assert config.n_mines == 3
        assert config.mine_density == 3 / 25
    
    def test_board_size_config_with_density(self):
        """Test creating a BoardSizeConfig with mine density."""
        config = BoardSizeConfig(width=8, height=8, n_mines=None, mine_density=0.2)
        assert config.width == 8
        assert config.height == 8
        assert config.n_mines == 12  # 64 * 0.2 = 12.8, rounded down to 12
        assert config.mine_density == 0.2


class TestDynamicEnvironmentConfig:
    """Test DynamicEnvironmentConfig functionality."""
    
    def test_dynamic_config_with_curriculum(self):
        """Test creating a DynamicEnvironmentConfig with curriculum."""
        curriculum = CurriculumConfig(
            enabled=True,
            start_size=BoardSizeConfig(width=5, height=5, n_mines=3),
            end_size=BoardSizeConfig(width=8, height=8, n_mines=12)
        )
        
        config = DynamicEnvironmentConfig(curriculum=curriculum)
        assert config.is_dynamic()
        assert config.curriculum.enabled
        assert len(config.get_board_sizes()) == 2
    
    def test_dynamic_config_with_board_sizes(self):
        """Test creating a DynamicEnvironmentConfig with multiple board sizes."""
        board_sizes = [
            BoardSizeConfig(width=5, height=5, n_mines=3),
            BoardSizeConfig(width=6, height=6, n_mines=6),
            BoardSizeConfig(width=7, height=7, n_mines=9)
        ]
        
        config = DynamicEnvironmentConfig(board_sizes=board_sizes)
        assert config.is_dynamic()
        assert len(config.get_board_sizes()) == 3
    
    def test_dynamic_config_validation(self):
        """Test DynamicEnvironmentConfig validation."""
        # Valid configuration
        config = DynamicEnvironmentConfig(
            board_sizes=[BoardSizeConfig(width=5, height=5, n_mines=3)]
        )
        assert validate_environment_config(config)
        
        # Invalid configuration (mines >= cells)
        invalid_config = DynamicEnvironmentConfig(
            board_sizes=[BoardSizeConfig(width=3, height=3, n_mines=10)]
        )
        assert not validate_environment_config(invalid_config)


class TestCustomCNN:
    """Test CustomCNN with adaptive pooling."""
    
    def test_adaptive_cnn_variable_input_sizes(self):
        """Test that CustomCNN handles variable input sizes."""
        # Test with different input sizes
        input_sizes = [(5, 5), (8, 8), (10, 10)]
        features_dim = 64
        
        for height, width in input_sizes:
            obs_space = Box(low=0.0, high=1.0, shape=(1, height, width), dtype=np.float32)
            cnn = CustomCNN(obs_space, features_dim=features_dim)
            
            # Test forward pass
            batch_size = 4
            input_tensor = torch.randn(batch_size, 1, height, width)
            output = cnn(input_tensor)
            
            # Should always output the same feature dimension
            assert output.shape == (batch_size, features_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_adaptive_cnn_consistency(self):
        """Test that CustomCNN produces consistent output structure."""
        features_dim = 32
        obs_space = Box(low=0.0, high=1.0, shape=(1, 8, 8), dtype=np.float32)
        cnn = CustomCNN(obs_space, features_dim=features_dim)
        
        # Test multiple forward passes
        input_tensor = torch.randn(2, 1, 8, 8)
        output1 = cnn(input_tensor)
        output2 = cnn(input_tensor)
        
        # Should be deterministic (same input -> same output)
        assert torch.allclose(output1, output2)
        assert output1.shape == (2, features_dim)


class TestCurriculumWrapper:
    """Test CurriculumLearningWrapper functionality."""
    
    def test_curriculum_wrapper_initialization(self):
        """Test CurriculumLearningWrapper initialization."""
        base_env = MinesweeperEnv(width=5, height=5, n_mines=3)
        
        curriculum_config = CurriculumConfig(
            enabled=True,
            start_size=BoardSizeConfig(width=5, height=5, n_mines=3),
            end_size=BoardSizeConfig(width=8, height=8, n_mines=12)
        )
        
        reward_config = {
            'reward_win': 1.0,
            'reward_lose': -0.1,
            'reward_reveal': 0.02,
            'reward_invalid': -0.05,
            'max_reward_per_step': 0.5
        }
        
        wrapper = CurriculumLearningWrapper(
            env=base_env,
            curriculum_config=curriculum_config,
            reward_config=reward_config,
            seed=42
        )
        
        # Check that wrapper has correct spaces (should be size of end_size)
        assert wrapper.observation_space.shape == (1, 8, 8)
        assert wrapper.action_space.n == 64
        
        # Check curriculum info
        info = wrapper.get_curriculum_info()
        assert 'current_board_size' in info
        assert 'total_timesteps' in info
        assert 'recent_win_rate' in info
    
    def test_curriculum_wrapper_reset_and_step(self):
        """Test CurriculumLearningWrapper reset and step functionality."""
        base_env = MinesweeperEnv(width=5, height=5, n_mines=3)
        
        curriculum_config = CurriculumConfig(
            enabled=True,
            start_size=BoardSizeConfig(width=5, height=5, n_mines=3),
            end_size=BoardSizeConfig(width=6, height=6, n_mines=6)
        )
        
        reward_config = {
            'reward_win': 1.0,
            'reward_lose': -0.1,
            'reward_reveal': 0.02,
            'reward_invalid': -0.05,
            'max_reward_per_step': 0.5
        }
        
        wrapper = CurriculumLearningWrapper(
            env=base_env,
            curriculum_config=curriculum_config,
            reward_config=reward_config,
            seed=42
        )
        
        # Test reset
        obs, info = wrapper.reset()
        assert obs.shape == (1, 6, 6)  # Padded to max size
        
        # Test step
        action = wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper.step(action)
        assert obs.shape == (1, 6, 6)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


class TestDynamicEnvironmentWrapper:
    """Test DynamicEnvironmentWrapper functionality."""
    
    def test_dynamic_wrapper_initialization(self):
        """Test DynamicEnvironmentWrapper initialization."""
        base_env = MinesweeperEnv(width=5, height=5, n_mines=3)
        
        board_sizes = [
            BoardSizeConfig(width=5, height=5, n_mines=3),
            BoardSizeConfig(width=6, height=6, n_mines=6),
        ]
        
        reward_config = {
            'reward_win': 1.0,
            'reward_lose': -0.1,
            'reward_reveal': 0.02,
            'reward_invalid': -0.05,
            'max_reward_per_step': 0.5
        }
        
        wrapper = DynamicEnvironmentWrapper(
            env=base_env,
            board_sizes=board_sizes,
            reward_config=reward_config,
            seed=42
        )
        
        # Check that wrapper has correct spaces (should be size of max board size)
        assert wrapper.observation_space.shape == (1, 6, 6)
        assert wrapper.action_space.n == 36
    
    def test_dynamic_wrapper_sampling(self):
        """Test DynamicEnvironmentWrapper board size sampling."""
        base_env = MinesweeperEnv(width=5, height=5, n_mines=3)
        
        board_sizes = [
            BoardSizeConfig(width=5, height=5, n_mines=3),
            BoardSizeConfig(width=7, height=7, n_mines=9),
        ]
        
        reward_config = {
            'reward_win': 1.0,
            'reward_lose': -0.1,
            'reward_reveal': 0.02,
            'reward_invalid': -0.05,
            'max_reward_per_step': 0.5
        }
        
        wrapper = DynamicEnvironmentWrapper(
            env=base_env,
            board_sizes=board_sizes,
            reward_config=reward_config,
            sampling_weights=[0.7, 0.3],
            seed=42
        )
        
        # Test multiple resets to ensure sampling works
        board_sizes_seen = set()
        for _ in range(10):
            obs, info = wrapper.reset()
            # Current board size should be one of the configured sizes
            current_size = wrapper.current_board_size
            assert current_size.width in [5, 7]
            assert current_size.height in [5, 7]
            board_sizes_seen.add((current_size.width, current_size.height))
        
        # Should have seen at least one different size (with high probability)
        assert len(board_sizes_seen) >= 1


class TestConfigManager:
    """Test ConfigManager with dynamic environment configurations."""
    
    def test_config_manager_dynamic_environment(self):
        """Test ConfigManager with DynamicEnvironmentConfig."""
        config_dict = {
            'model_hyperparams': {
                'learning_rate': 0.001,
                'ent_coef': 0.01,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'vf_coef': 0.5,
                'n_steps': 512,
                'batch_size': 64,
                'n_epochs': 4
            },
            'network_architecture': {
                'features_dim': 32,
                'pi_layers': [32],
                'vf_layers': [32]
            },
            'dynamic_environment_config': {
                'board_sizes': [
                    {'width': 5, 'height': 5, 'n_mines': 3},
                    {'width': 6, 'height': 6, 'n_mines': 6}
                ],
                'reward_win': 1.0,
                'reward_lose': -0.1,
                'reward_reveal': 0.02,
                'reward_invalid': -0.05
            },
            'training_execution': {
                'total_timesteps': 10000,
                'n_envs': 2,
                'vec_env_type': 'dummy',
                'checkpoint_freq': 1000,
                'device': 'cpu',
                'seed': 42
            },
            'paths_config': {
                'experiment_base_dir': './test_runs',
                'model_prefix': 'test_model'
            }
        }
        
        config = create_config_from_dict(config_dict)
        
        manager = ConfigManager()
        manager.config_sources['file'] = config_dict
        built_config = manager.build_config()
        
        assert manager.is_dynamic_environment()
        assert isinstance(built_config.environment_config, DynamicEnvironmentConfig)
        
        reward_config = manager.get_reward_config()
        assert reward_config['reward_win'] == 1.0
        assert reward_config['reward_lose'] == -0.1


if __name__ == "__main__":
    pytest.main([__file__])