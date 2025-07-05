import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import gymnasium as gym
import numpy as np

# Import modules we'll be testing (will be created)
from src.factories.environment_factory import (
    create_env_config,
    create_base_environment,
    create_vectorized_environment,
    create_training_environment,
    create_inference_environment,
    load_vecnormalize_stats,
    EnvironmentCreationError
)

# Import new configuration system for testing
from src.config.config_manager import ConfigManager
from src.config.config_schemas import EnvironmentConfig


class TestEnvConfig:
    """Test environment configuration creation."""
    
    def test_create_env_config_default_values(self):
        """Test creating environment config with default values."""
        config = create_env_config()
        
        assert config is not None
        assert 'width' in config
        assert 'height' in config
        assert 'n_mines' in config
        assert 'reward_win' in config
        assert 'reward_lose' in config
        assert 'reward_reveal' in config
        assert 'reward_invalid' in config
        assert 'max_reward_per_step' in config
        
        # Check that all values are numeric (except render_mode and max_reward_per_step which can be None)
        for key, value in config.items():
            if key not in ['render_mode', 'max_reward_per_step']:
                assert isinstance(value, (int, float)), f"Key {key} has value {value} of type {type(value)}"
            elif key == 'max_reward_per_step':
                assert value is None or isinstance(value, (int, float)), f"max_reward_per_step should be None or numeric, got {value}"
    
    def test_create_env_config_custom_values(self):
        """Test creating environment config with custom values."""
        config = create_env_config(
            width=10,
            height=8,
            n_mines=15,
            reward_win=100.0,
            reward_lose=-10.0,
            reward_reveal=1.0,
            reward_invalid=-1.0,
            max_reward_per_step=5.0,
            render_mode='human'
        )
        
        assert config['width'] == 10
        assert config['height'] == 8
        assert config['n_mines'] == 15
        assert config['reward_win'] == 100.0
        assert config['reward_lose'] == -10.0
        assert config['reward_reveal'] == 1.0
        assert config['reward_invalid'] == -1.0
        assert config['max_reward_per_step'] == 5.0
        assert config['render_mode'] == 'human'
    
    def test_create_env_config_from_args(self):
        """Test creating environment config from args object."""
        mock_args = Mock()
        mock_args.width = 12
        mock_args.height = 12
        mock_args.n_mines = 20
        mock_args.reward_win = 50.0
        mock_args.reward_lose = -5.0
        mock_args.reward_reveal = 0.5
        mock_args.reward_invalid = -0.5
        mock_args.max_reward_per_step = 3.0
        
        config = create_env_config(args=mock_args, render_mode='rgb_array')
        
        assert config['width'] == 12
        assert config['height'] == 12
        assert config['n_mines'] == 20
        assert config['render_mode'] == 'rgb_array'


class TestBaseEnvironment:
    """Test base environment creation."""
    
    def test_create_base_environment_with_config(self):
        """Test creating base environment with configuration."""
        config = {
            'width': 5,
            'height': 5,
            'n_mines': 3,
            'reward_win': 10.0,
            'reward_lose': -10.0,
            'reward_reveal': 1.0,
            'reward_invalid': -1.0,
            'max_reward_per_step': 2.0,
            'render_mode': None
        }
        
        with patch('src.factories.environment_factory.MinesweeperEnv') as mock_env_class:
            mock_env = Mock()
            mock_env_class.return_value = mock_env
            
            env = create_base_environment(config)
            
            assert env == mock_env
            mock_env_class.assert_called_once_with(**config)
    
    def test_create_base_environment_with_custom_render_mode(self):
        """Test creating base environment with custom render mode."""
        config = create_env_config(render_mode='human')
        
        with patch('src.factories.environment_factory.MinesweeperEnv') as mock_env_class:
            mock_env = Mock()
            mock_env_class.return_value = mock_env
            
            env = create_base_environment(config)
            
            mock_env_class.assert_called_once()
            call_kwargs = mock_env_class.call_args[1]
            assert call_kwargs['render_mode'] == 'human'
    
    def test_create_env_config_from_config_manager(self):
        """Test creating environment config from ConfigManager."""
        config_manager = ConfigManager()
        config_manager.config.environment_config.width = 10
        config_manager.config.environment_config.height = 8
        config_manager.config.environment_config.n_mines = 15
        config_manager.config.environment_config.reward_win = 100.0
        
        # Test the new interface we'll add
        config = create_env_config(config_manager=config_manager, render_mode='human')
        
        assert config['width'] == 10
        assert config['height'] == 8
        assert config['n_mines'] == 15
        assert config['reward_win'] == 100.0
        assert config['render_mode'] == 'human'
    
    def test_create_base_environment_error_handling(self):
        """Test error handling in base environment creation."""
        config = create_env_config()
        
        with patch('src.factories.environment_factory.MinesweeperEnv') as mock_env_class:
            mock_env_class.side_effect = Exception("Environment creation failed")
            
            with pytest.raises(EnvironmentCreationError, match="Failed to create base environment"):
                create_base_environment(config)


class TestVectorizedEnvironment:
    """Test vectorized environment creation."""
    
    @pytest.fixture
    def mock_base_env(self):
        """Create a mock base environment."""
        mock_env = Mock()
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 5, 5))
        mock_env.action_space = gym.spaces.Discrete(25)
        return mock_env
    
    def test_create_vectorized_environment_single_env(self, mock_base_env):
        """Test creating vectorized environment with single environment."""
        with patch('src.factories.environment_factory.DummyVecEnv') as mock_vec_env:
            mock_vectorized = Mock()
            mock_vec_env.return_value = mock_vectorized
            
            env_fn = lambda: mock_base_env
            
            vec_env = create_vectorized_environment(
                env_fn=env_fn,
                n_envs=1,
                vec_env_type='dummy'
            )
            
            assert vec_env == mock_vectorized
            mock_vec_env.assert_called_once()
    
    def test_create_vectorized_environment_multi_env_subproc(self, mock_base_env):
        """Test creating vectorized environment with multiple SubprocVecEnv."""
        with patch('src.factories.environment_factory.make_vec_env') as mock_make_vec_env:
            mock_vectorized = Mock()
            mock_make_vec_env.return_value = mock_vectorized
            
            env_fn = lambda: mock_base_env
            
            vec_env = create_vectorized_environment(
                env_fn=env_fn,
                n_envs=4,
                vec_env_type='subproc',
                seed=42
            )
            
            assert vec_env == mock_vectorized
            mock_make_vec_env.assert_called_once()
            call_kwargs = mock_make_vec_env.call_args[1]
            assert call_kwargs['n_envs'] == 4
            assert call_kwargs['seed'] == 42
    
    def test_create_vectorized_environment_multi_env_dummy_fallback(self, mock_base_env):
        """Test fallback to DummyVecEnv for dummy type with multiple envs."""
        with patch('src.factories.environment_factory.DummyVecEnv') as mock_vec_env:
            mock_vectorized = Mock()
            mock_vec_env.return_value = mock_vectorized
            
            env_fn = lambda: mock_base_env
            
            vec_env = create_vectorized_environment(
                env_fn=env_fn,
                n_envs=4,
                vec_env_type='dummy'
            )
            
            assert vec_env == mock_vectorized
            mock_vec_env.assert_called_once()
    
    def test_create_vectorized_environment_auto_type_selection(self, mock_base_env):
        """Test automatic vec_env_type selection."""
        with patch('src.factories.environment_factory.make_vec_env') as mock_make_vec_env:
            mock_vectorized = Mock()
            mock_make_vec_env.return_value = mock_vectorized
            
            env_fn = lambda: mock_base_env
            
            # Should use subproc for multiple envs
            vec_env = create_vectorized_environment(
                env_fn=env_fn,
                n_envs=4
            )
            
            mock_make_vec_env.assert_called_once()
            call_kwargs = mock_make_vec_env.call_args[1]
            # Check that SubprocVecEnv was used (not DummyVecEnv)
            from stable_baselines3.common.vec_env import SubprocVecEnv
            assert call_kwargs['vec_env_cls'] == SubprocVecEnv


class TestVecNormalizeHandling:
    """Test VecNormalize statistics handling."""
    
    def test_load_vecnormalize_stats_success(self, temp_dir):
        """Test successful loading of VecNormalize stats."""
        stats_path = os.path.join(temp_dir, "stats.pkl")
        Path(stats_path).touch()
        
        mock_env = Mock()
        mock_normalized_env = Mock()
        
        with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
            mock_vecnorm.load.return_value = mock_normalized_env
            
            result_env = load_vecnormalize_stats(mock_env, stats_path)
            
            assert result_env == mock_normalized_env
            mock_vecnorm.load.assert_called_once_with(stats_path, mock_env)
    
    def test_load_vecnormalize_stats_nonexistent_file(self, temp_dir):
        """Test loading VecNormalize stats from non-existent file."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.pkl")
        mock_env = Mock()
        
        result_env = load_vecnormalize_stats(mock_env, nonexistent_path)
        assert result_env == mock_env
    
    def test_load_vecnormalize_stats_none_path(self):
        """Test loading VecNormalize stats with None path."""
        mock_env = Mock()
        
        result_env = load_vecnormalize_stats(mock_env, None)
        assert result_env == mock_env
    
    def test_load_vecnormalize_stats_with_training_mode(self, temp_dir):
        """Test loading VecNormalize stats with training mode configuration."""
        stats_path = os.path.join(temp_dir, "stats.pkl")
        Path(stats_path).touch()
        
        mock_env = Mock()
        mock_normalized_env = Mock()
        
        with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
            mock_vecnorm.load.return_value = mock_normalized_env
            
            result_env = load_vecnormalize_stats(
                mock_env, 
                stats_path, 
                training_mode=False, 
                norm_reward=False
            )
            
            assert result_env == mock_normalized_env
            assert mock_normalized_env.training == False
            assert mock_normalized_env.norm_reward == False


class TestTrainingEnvironment:
    """Test training environment creation."""
    
    def test_create_training_environment_default(self):
        """Test creating training environment with default settings."""
        mock_args = Mock()
        mock_args.width = 5
        mock_args.height = 5
        mock_args.n_mines = 3
        mock_args.reward_win = 10.0
        mock_args.reward_lose = -10.0
        mock_args.reward_reveal = 1.0
        mock_args.reward_invalid = -1.0
        mock_args.max_reward_per_step = 2.0
        mock_args.n_envs = 4
        mock_args.vec_env_type = 'subproc'
        mock_args.seed = None
        mock_args.gamma = 0.99
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                env = create_training_environment(mock_args)
                
                assert env == mock_normalized_env
                mock_create_vec.assert_called_once()
                mock_vecnorm.assert_called_once_with(
                    mock_vec_env,
                    norm_obs=False,
                    norm_reward=True,
                    clip_obs=10.0,
                    gamma=0.99
                )
    
    def test_create_training_environment_with_stats(self, temp_dir):
        """Test creating training environment with VecNormalize stats."""
        stats_path = os.path.join(temp_dir, "stats.pkl")
        Path(stats_path).touch()
        
        mock_args = Mock()
        mock_args.width = 5
        mock_args.height = 5
        mock_args.n_mines = 3
        mock_args.reward_win = 10.0
        mock_args.reward_lose = -10.0
        mock_args.reward_reveal = 1.0
        mock_args.reward_invalid = -1.0
        mock_args.max_reward_per_step = 2.0
        mock_args.n_envs = 1
        mock_args.vec_env_type = 'dummy'
        mock_args.seed = 42
        mock_args.gamma = 0.95
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                with patch('src.factories.environment_factory.load_vecnormalize_stats') as mock_load_stats:
                    mock_vec_env = Mock()
                    mock_normalized_env = Mock()
                    mock_stats_env = Mock()
                    
                    mock_create_vec.return_value = mock_vec_env
                    mock_vecnorm.return_value = mock_normalized_env
                    mock_load_stats.return_value = mock_stats_env
                    
                    env = create_training_environment(
                        mock_args, 
                        vecnormalize_stats_path=stats_path
                    )
                    
                    assert env == mock_stats_env
                    mock_load_stats.assert_called_once_with(
                        mock_normalized_env, 
                        stats_path, 
                        training_mode=True
                    )
    
    def test_create_training_environment_with_config_manager(self):
        """Test creating training environment with ConfigManager."""
        config_manager = ConfigManager()
        config_manager.config.environment_config.width = 8
        config_manager.config.environment_config.height = 8
        config_manager.config.environment_config.n_mines = 12
        config_manager.config.training_execution.n_envs = 2
        config_manager.config.training_execution.vec_env_type = 'dummy'
        config_manager.config.training_execution.seed = 123
        config_manager.config.model_hyperparams.gamma = 0.95
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                # Test the new interface we'll add
                env = create_training_environment(config_manager=config_manager)
                
                assert env == mock_normalized_env
                mock_create_vec.assert_called_once()
                mock_vecnorm.assert_called_once_with(
                    mock_vec_env,
                    norm_obs=False,
                    norm_reward=True,
                    clip_obs=10.0,
                    gamma=0.95
                )


class TestInferenceEnvironment:
    """Test inference environment creation."""
    
    def test_create_inference_environment_batch_mode(self):
        """Test creating inference environment for batch mode."""
        mock_args = Mock()
        mock_args.width = 8
        mock_args.height = 8
        mock_args.n_mines = 10
        mock_args.reward_win = 20.0
        mock_args.reward_lose = -20.0
        mock_args.reward_reveal = 2.0
        mock_args.reward_invalid = -2.0
        mock_args.max_reward_per_step = 5.0
        mock_args.seed = 123
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            mock_vec_env = Mock()
            mock_create_vec.return_value = mock_vec_env
            
            env, raw_env = create_inference_environment(
                mock_args, 
                mode='batch'
            )
            
            assert env == mock_vec_env
            assert raw_env is None
            mock_create_vec.assert_called_once()
            call_args = mock_create_vec.call_args
            # Should use DummyVecEnv for inference
            assert call_args[1]['vec_env_type'] == 'dummy'
            assert call_args[1]['n_envs'] == 1
    
    def test_create_inference_environment_interactive_mode(self):
        """Test creating inference environment for interactive mode."""
        mock_args = Mock()
        mock_args.width = 6
        mock_args.height = 6
        mock_args.n_mines = 8
        mock_args.reward_win = 15.0
        mock_args.reward_lose = -15.0
        mock_args.reward_reveal = 1.5
        mock_args.reward_invalid = -1.5
        mock_args.max_reward_per_step = 3.0
        mock_args.seed = 456
        
        with patch('src.factories.environment_factory.create_base_environment') as mock_create_base:
            with patch('src.factories.environment_factory.DummyVecEnv') as mock_dummy_vec:
                mock_base_env = Mock()
                mock_vec_env = Mock()
                mock_create_base.return_value = mock_base_env
                mock_dummy_vec.return_value = mock_vec_env
                
                env, raw_env = create_inference_environment(
                    mock_args, 
                    mode='interactive'
                )
                
                assert env == mock_vec_env
                assert raw_env == mock_base_env
                mock_create_base.assert_called_once()
                mock_dummy_vec.assert_called_once()
    
    def test_create_inference_environment_with_stats(self, temp_dir):
        """Test creating inference environment with VecNormalize stats."""
        stats_path = os.path.join(temp_dir, "inference_stats.pkl")
        Path(stats_path).touch()
        
        mock_args = Mock()
        mock_args.width = 5
        mock_args.height = 5
        mock_args.n_mines = 3
        mock_args.reward_win = 10.0
        mock_args.reward_lose = -10.0
        mock_args.reward_reveal = 1.0
        mock_args.reward_invalid = -1.0
        mock_args.max_reward_per_step = 2.0
        mock_args.seed = None
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.load_vecnormalize_stats') as mock_load_stats:
                mock_vec_env = Mock()
                mock_stats_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_load_stats.return_value = mock_stats_env
                
                env, raw_env = create_inference_environment(
                    mock_args,
                    mode='batch',
                    vecnormalize_stats_path=stats_path
                )
                
                assert env == mock_stats_env
                mock_load_stats.assert_called_once_with(
                    mock_vec_env,
                    stats_path,
                    training_mode=False,
                    norm_reward=False
                )


class TestSeedHandling:
    """Test random seed handling functionality."""
    
    def test_seed_handling_in_vectorized_env(self):
        """Test that seed is properly passed to vectorized environment."""
        mock_args = Mock()
        mock_args.width = 5
        mock_args.height = 5
        mock_args.n_mines = 3
        mock_args.reward_win = 10.0
        mock_args.reward_lose = -10.0
        mock_args.reward_reveal = 1.0
        mock_args.reward_invalid = -1.0
        mock_args.max_reward_per_step = 2.0
        mock_args.n_envs = 2
        mock_args.vec_env_type = 'dummy'
        mock_args.seed = 789
        mock_args.gamma = 0.99
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                env = create_training_environment(mock_args)
                
                # Check that seed was passed to vectorized environment creation
                call_kwargs = mock_create_vec.call_args[1]
                assert call_kwargs['seed'] == 789


class TestErrorHandling:
    """Test error handling throughout environment factory."""
    
    def test_environment_creation_error_instantiation(self):
        """Test that EnvironmentCreationError can be instantiated."""
        error = EnvironmentCreationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_environment_creation_error_with_cause(self):
        """Test EnvironmentCreationError with underlying cause."""
        original_error = ValueError("Original error")
        error = EnvironmentCreationError("Wrapper error", original_error)
        
        assert str(error) == "Wrapper error"
        assert error.__cause__ == original_error
    
    def test_training_environment_creation_error(self):
        """Test error handling in training environment creation."""
        mock_args = Mock()
        mock_args.width = 5
        mock_args.height = 5
        mock_args.n_mines = 3
        mock_args.reward_win = 10.0
        mock_args.reward_lose = -10.0
        mock_args.reward_reveal = 1.0
        mock_args.reward_invalid = -1.0
        mock_args.max_reward_per_step = 2.0
        mock_args.n_envs = 1
        mock_args.vec_env_type = 'dummy'
        mock_args.seed = None
        mock_args.gamma = 0.99
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            mock_create_vec.side_effect = Exception("Vectorization failed")
            
            with pytest.raises(EnvironmentCreationError, match="Failed to create training environment"):
                create_training_environment(mock_args)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_train_py_workflow_simulation(self):
        """Test simulating the train.py environment creation workflow."""
        mock_args = Mock()
        mock_args.width = 10
        mock_args.height = 10
        mock_args.n_mines = 15
        mock_args.reward_win = 100.0
        mock_args.reward_lose = -100.0
        mock_args.reward_reveal = 1.0
        mock_args.reward_invalid = -1.0
        mock_args.max_reward_per_step = 10.0
        mock_args.n_envs = 8
        mock_args.vec_env_type = 'subproc'
        mock_args.seed = 42
        mock_args.gamma = 0.99
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                env = create_training_environment(mock_args)
                
                assert env == mock_normalized_env
                # Verify training-specific VecNormalize settings
                vecnorm_call = mock_vecnorm.call_args[1]
                assert vecnorm_call['norm_obs'] == False
                assert vecnorm_call['norm_reward'] == True
    
    def test_play_py_workflow_simulation(self):
        """Test simulating the play.py environment creation workflow."""
        mock_args = Mock()
        mock_args.width = 8
        mock_args.height = 8
        mock_args.n_mines = 12
        mock_args.reward_win = 50.0
        mock_args.reward_lose = -50.0
        mock_args.reward_reveal = 0.5
        mock_args.reward_invalid = -0.5
        mock_args.max_reward_per_step = 5.0
        mock_args.seed = 123
        
        # Test batch mode
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            mock_vec_env = Mock()
            mock_create_vec.return_value = mock_vec_env
            
            env, raw_env = create_inference_environment(mock_args, mode='batch')
            
            assert env == mock_vec_env
            assert raw_env is None
            
            # Verify inference-specific settings
            call_kwargs = mock_create_vec.call_args[1]
            assert call_kwargs['n_envs'] == 1
            assert call_kwargs['vec_env_type'] == 'dummy'
        
        # Test interactive mode
        with patch('src.factories.environment_factory.create_base_environment') as mock_create_base:
            with patch('src.factories.environment_factory.DummyVecEnv') as mock_dummy_vec:
                mock_base_env = Mock()
                mock_vec_env = Mock()
                mock_create_base.return_value = mock_base_env
                mock_dummy_vec.return_value = mock_vec_env
                
                env, raw_env = create_inference_environment(mock_args, mode='interactive')
                
                assert env == mock_vec_env
                assert raw_env == mock_base_env