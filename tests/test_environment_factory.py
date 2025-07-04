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

# Import test helper
from tests.test_config_helper import create_test_config_manager


class TestEnvConfig:
    """Test environment configuration creation."""
    
    def test_create_env_config_from_config_manager(self):
        """Test creating environment config from ConfigManager with default values."""
        config_manager = create_test_config_manager()
        config = create_env_config(config_manager=config_manager)
        
        assert config is not None
        assert 'width' in config
        assert 'height' in config
        assert 'n_mines' in config
        assert 'reward_win' in config
        assert 'reward_lose' in config
        assert 'reward_reveal' in config
        assert 'reward_invalid' in config
        assert 'max_reward_per_step' in config
        assert 'render_mode' in config
        
        # All values should be non-None except render_mode and max_reward_per_step
        for key, value in config.items():
            if key not in ['render_mode', 'max_reward_per_step']:  # these can be None
                assert value is not None, f"ConfigManager should provide non-None default for {key}"
    
    def test_create_env_config_with_render_mode(self):
        """Test creating environment config with custom render mode."""
        config_manager = create_test_config_manager()
        config = create_env_config(config_manager=config_manager, render_mode='human')
        
        assert config['render_mode'] == 'human'
        # Other values should come from ConfigManager
        assert config['width'] == config_manager.get_config().environment_config.width
        assert config['height'] == config_manager.get_config().environment_config.height
    
    def test_create_env_config_missing_config_manager(self):
        """Test that factory requires ConfigManager."""
        with pytest.raises(TypeError):
            create_env_config()
    
    def test_create_env_config_incomplete_config_manager(self):
        """Test that factory validates ConfigManager completeness."""
        config_manager = create_test_config_manager()
        # Intentionally set a required value to None
        config_manager.get_config().environment_config.width = None
        
        with pytest.raises(ValueError, match="ConfigManager.environment_config.width is None"):
            create_env_config(config_manager=config_manager)


class TestBaseEnvironment:
    """Test base environment creation."""
    
    def test_create_base_environment_success(self):
        """Test successful creation of base environment."""
        config_manager = create_test_config_manager()
        config = create_env_config(config_manager=config_manager)
        
        with patch('src.factories.environment_factory.MinesweeperEnv') as mock_env_class:
            mock_env = Mock()
            mock_env_class.return_value = mock_env
            
            env = create_base_environment(config)
            
            assert env == mock_env
            mock_env_class.assert_called_once_with(**config)
    
    def test_create_base_environment_with_custom_render_mode(self):
        """Test creating base environment with custom render mode."""
        config_manager = create_test_config_manager()
        config = create_env_config(config_manager=config_manager, render_mode='human')
        
        with patch('src.factories.environment_factory.MinesweeperEnv') as mock_env_class:
            mock_env = Mock()
            mock_env_class.return_value = mock_env
            
            env = create_base_environment(config)
            
            mock_env_class.assert_called_once()
            call_kwargs = mock_env_class.call_args[1]
            assert call_kwargs['render_mode'] == 'human'
    
    def test_create_base_environment_error_handling(self):
        """Test error handling in base environment creation."""
        config_manager = create_test_config_manager()
        config = create_env_config(config_manager=config_manager)
        
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
    
    def test_create_training_environment_success(self):
        """Test creating training environment with ConfigManager."""
        config_manager = create_test_config_manager()
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                env = create_training_environment(config_manager)
                
                assert env == mock_normalized_env
                mock_create_vec.assert_called_once()
                mock_vecnorm.assert_called_once_with(
                    mock_vec_env,
                    norm_obs=False,
                    norm_reward=True,
                    clip_obs=10.0,
                    gamma=config_manager.get_config().model_hyperparams.gamma
                )
    
    def test_create_training_environment_with_stats(self, temp_dir):
        """Test creating training environment with VecNormalize stats."""
        stats_path = os.path.join(temp_dir, "stats.pkl")
        Path(stats_path).touch()
        
        config_manager = create_test_config_manager()
        
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
                        config_manager, 
                        vecnormalize_stats_path=stats_path
                    )
                    
                    assert env == mock_stats_env
                    mock_load_stats.assert_called_once_with(
                        mock_normalized_env, 
                        stats_path, 
                        training_mode=True
                    )
    
    def test_create_training_environment_missing_config_manager(self):
        """Test that training environment requires ConfigManager."""
        with pytest.raises(TypeError):
            create_training_environment()


class TestInferenceEnvironment:
    """Test inference environment creation."""
    
    def test_create_inference_environment_batch_mode(self):
        """Test creating inference environment for batch mode."""
        config_manager = create_test_config_manager()
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            mock_vec_env = Mock()
            mock_create_vec.return_value = mock_vec_env
            
            env, raw_env = create_inference_environment(
                config_manager, 
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
        """Test creating inference environment for interactive mode (human/agent)."""
        config_manager = create_test_config_manager()
        
        # Test human mode
        with patch('src.factories.environment_factory.create_base_environment') as mock_create_base:
            with patch('src.factories.environment_factory.DummyVecEnv') as mock_dummy_vec:
                mock_base_env = Mock()
                mock_vec_env = Mock()
                mock_create_base.return_value = mock_base_env
                mock_dummy_vec.return_value = mock_vec_env
                
                env, raw_env = create_inference_environment(
                    config_manager, 
                    mode='human'
                )
                
                assert env == mock_vec_env
                assert raw_env == mock_base_env
                mock_create_base.assert_called_once()
                mock_dummy_vec.assert_called_once()
        
        # Test agent mode  
        with patch('src.factories.environment_factory.create_base_environment') as mock_create_base:
            with patch('src.factories.environment_factory.DummyVecEnv') as mock_dummy_vec:
                mock_base_env = Mock()
                mock_vec_env = Mock()
                mock_create_base.return_value = mock_base_env
                mock_dummy_vec.return_value = mock_vec_env
                
                env, raw_env = create_inference_environment(
                    config_manager, 
                    mode='agent'
                )
                
                assert env == mock_vec_env
                assert raw_env == mock_base_env
                mock_create_base.assert_called_once()
                mock_dummy_vec.assert_called_once()
    
    def test_create_inference_environment_with_stats(self, temp_dir):
        """Test creating inference environment with VecNormalize stats."""
        stats_path = os.path.join(temp_dir, "inference_stats.pkl")
        Path(stats_path).touch()
        
        config_manager = create_test_config_manager()
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.load_vecnormalize_stats') as mock_load_stats:
                mock_vec_env = Mock()
                mock_stats_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_load_stats.return_value = mock_stats_env
                
                env, raw_env = create_inference_environment(
                    config_manager,
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
    
    def test_create_inference_environment_missing_config_manager(self):
        """Test that inference environment requires ConfigManager."""
        with pytest.raises(TypeError):
            create_inference_environment(mode='batch')


class TestSeedHandling:
    """Test random seed handling functionality."""
    
    def test_seed_handling_in_vectorized_env(self):
        """Test that seed is properly passed to vectorized environment."""
        config_manager = create_test_config_manager()
        config_manager.get_config().environment_config.width = 5
        config_manager.get_config().environment_config.height = 5
        config_manager.get_config().environment_config.n_mines = 3
        config_manager.get_config().environment_config.reward_win = 10.0
        config_manager.get_config().environment_config.reward_lose = -10.0
        config_manager.get_config().environment_config.reward_reveal = 1.0
        config_manager.get_config().environment_config.reward_invalid = -1.0
        config_manager.get_config().environment_config.max_reward_per_step = 2.0
        config_manager.get_config().training_execution.n_envs = 2
        config_manager.get_config().training_execution.vec_env_type = 'dummy'
        config_manager.get_config().training_execution.seed = 789
        config_manager.get_config().model_hyperparams.gamma = 0.99
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                env = create_training_environment(config_manager)
                
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
        config_manager = create_test_config_manager()
        # Set custom training configuration
        config_manager.get_config().environment_config.width = 10
        config_manager.get_config().environment_config.height = 10
        config_manager.get_config().environment_config.n_mines = 15
        config_manager.get_config().training_execution.n_envs = 8
        config_manager.get_config().training_execution.vec_env_type = 'subproc'
        config_manager.get_config().training_execution.seed = 42
        config_manager.get_config().model_hyperparams.gamma = 0.99
        
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            with patch('src.factories.environment_factory.VecNormalize') as mock_vecnorm:
                mock_vec_env = Mock()
                mock_normalized_env = Mock()
                mock_create_vec.return_value = mock_vec_env
                mock_vecnorm.return_value = mock_normalized_env
                
                env = create_training_environment(config_manager)
                
                assert env == mock_normalized_env
                # Verify training-specific VecNormalize settings
                vecnorm_call = mock_vecnorm.call_args[1]
                assert vecnorm_call['norm_obs'] == False
                assert vecnorm_call['norm_reward'] == True
                assert vecnorm_call['gamma'] == 0.99
    
    def test_play_py_workflow_simulation(self):
        """Test simulating the play.py environment creation workflow."""
        config_manager = create_test_config_manager()
        # Set custom play configuration
        config_manager.get_config().environment_config.width = 8
        config_manager.get_config().environment_config.height = 8
        config_manager.get_config().environment_config.n_mines = 12
        config_manager.get_config().training_execution.seed = 123
        
        # Test batch mode
        with patch('src.factories.environment_factory.create_vectorized_environment') as mock_create_vec:
            mock_vec_env = Mock()
            mock_create_vec.return_value = mock_vec_env
            
            env, raw_env = create_inference_environment(config_manager, mode='batch')
            
            assert env == mock_vec_env
            assert raw_env is None
            
            # Verify inference-specific settings
            call_kwargs = mock_create_vec.call_args[1]
            assert call_kwargs['n_envs'] == 1
            assert call_kwargs['vec_env_type'] == 'dummy'
        
        # Test human mode
        with patch('src.factories.environment_factory.create_base_environment') as mock_create_base:
            with patch('src.factories.environment_factory.DummyVecEnv') as mock_dummy_vec:
                mock_base_env = Mock()
                mock_vec_env = Mock()
                mock_create_base.return_value = mock_base_env
                mock_dummy_vec.return_value = mock_vec_env
                
                env, raw_env = create_inference_environment(config_manager, mode='human')
                
                assert env == mock_vec_env
                assert raw_env == mock_base_env