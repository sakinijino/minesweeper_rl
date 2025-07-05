import pytest
import os
import torch
import gymnasium as gym
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import modules we'll be testing (will be created)
from src.factories.model_factory import (
    create_policy_kwargs,
    create_new_model,
    load_model_from_checkpoint,
    create_model,
    ModelCreationError
)
from src.env.custom_cnn import CustomCNN
from src.env.minesweeper_env import MinesweeperEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

# Import new configuration system for testing
from src.config.config_manager import ConfigManager
from src.config.config_schemas import ModelHyperparams, NetworkArchitecture


class TestPolicyKwargs:
    """Test policy kwargs creation functionality."""
    
    def test_create_policy_kwargs_default_values(self):
        """Test creating policy kwargs with default values."""
        policy_kwargs = create_policy_kwargs()
        
        assert policy_kwargs is not None
        assert 'features_extractor_class' in policy_kwargs
        assert 'features_extractor_kwargs' in policy_kwargs
        assert 'net_arch' in policy_kwargs
        
        # Check default values
        assert policy_kwargs['features_extractor_class'] == CustomCNN
        assert policy_kwargs['features_extractor_kwargs']['features_dim'] == 128
        assert policy_kwargs['net_arch'] == {'pi': [64, 64], 'vf': [256, 256]}
    
    def test_create_policy_kwargs_custom_features_dim(self):
        """Test creating policy kwargs with custom features dimension."""
        policy_kwargs = create_policy_kwargs(features_dim=256)
        
        assert policy_kwargs['features_extractor_kwargs']['features_dim'] == 256
    
    def test_create_policy_kwargs_custom_net_arch(self):
        """Test creating policy kwargs with custom network architecture."""
        pi_layers = [128, 64]
        vf_layers = [512, 256, 128]
        
        policy_kwargs = create_policy_kwargs(
            pi_layers=pi_layers,
            vf_layers=vf_layers
        )
        
        assert policy_kwargs['net_arch']['pi'] == pi_layers
        assert policy_kwargs['net_arch']['vf'] == vf_layers
    
    def test_create_policy_kwargs_empty_layers(self):
        """Test creating policy kwargs with empty layer lists."""
        policy_kwargs = create_policy_kwargs(
            pi_layers=[],
            vf_layers=[]
        )
        
        assert policy_kwargs['net_arch']['pi'] == []
        assert policy_kwargs['net_arch']['vf'] == []
    
    def test_create_policy_kwargs_single_layer(self):
        """Test creating policy kwargs with single layer networks."""
        policy_kwargs = create_policy_kwargs(
            pi_layers=[64],
            vf_layers=[128]
        )
        
        assert policy_kwargs['net_arch']['pi'] == [64]
        assert policy_kwargs['net_arch']['vf'] == [128]
    
    def test_create_policy_kwargs_from_config_manager(self):
        """Test creating policy kwargs from ConfigManager."""
        config_manager = ConfigManager()
        config_manager.config.network_architecture.features_dim = 256
        config_manager.config.network_architecture.pi_layers = [128, 64]
        config_manager.config.network_architecture.vf_layers = [512, 256]
        
        # Test the new interface we'll add
        policy_kwargs = create_policy_kwargs(config_manager=config_manager)
        
        assert policy_kwargs['features_extractor_kwargs']['features_dim'] == 256
        assert policy_kwargs['net_arch']['pi'] == [128, 64]
        assert policy_kwargs['net_arch']['vf'] == [512, 256]


class TestNewModelCreation:
    """Test new model creation functionality."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        mock_env = Mock()
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 5, 5))
        mock_env.action_space = gym.spaces.Discrete(25)
        return mock_env
    
    @pytest.fixture
    def sample_args(self):
        """Create sample arguments for model creation."""
        return {
            'n_steps': 1024,
            'batch_size': 128,
            'n_epochs': 10,
            'learning_rate': 1e-4,
            'ent_coef': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.90,
            'clip_range': 0.2,
            'vf_coef': 1.0,
            'device': 'cpu',
            'seed': 42
        }
    
    def test_create_new_model_success(self, mock_env, sample_args):
        """Test successful creation of new model."""
        with patch('src.factories.model_factory.MaskablePPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            # Add policy_kwargs to the sample_args
            policy_kwargs = create_policy_kwargs()
            
            model = create_new_model(
                env=mock_env,
                tensorboard_log="/tmp/logs",
                policy_kwargs=policy_kwargs,
                **sample_args
            )
            
            assert model == mock_model
            mock_ppo.assert_called_once()
            
            # Check that MaskablePPO was called with correct arguments
            call_args = mock_ppo.call_args
            # All arguments should be keyword arguments
            assert call_args[1]['policy'] == "CnnPolicy"
            assert call_args[1]['env'] == mock_env
            assert call_args[1]['verbose'] == 1
            assert call_args[1]['tensorboard_log'] == "/tmp/logs"
            assert call_args[1]['policy_kwargs'] == policy_kwargs
    
    def test_create_new_model_with_custom_policy_kwargs(self, mock_env, sample_args):
        """Test creating new model with custom policy kwargs."""
        custom_policy_kwargs = {
            'features_extractor_class': CustomCNN,
            'features_extractor_kwargs': {'features_dim': 256},
            'net_arch': {'pi': [128], 'vf': [256]}
        }
        
        with patch('src.factories.model_factory.MaskablePPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            model = create_new_model(
                env=mock_env,
                tensorboard_log="/tmp/logs",
                policy_kwargs=custom_policy_kwargs,
                **sample_args
            )
            
            call_args = mock_ppo.call_args
            assert call_args[1]['policy_kwargs'] == custom_policy_kwargs
    
    def test_create_new_model_without_tensorboard_log(self, mock_env, sample_args):
        """Test creating new model without tensorboard logging."""
        with patch('src.factories.model_factory.MaskablePPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.return_value = mock_model
            
            model = create_new_model(
                env=mock_env,
                **sample_args
            )
            
            call_args = mock_ppo.call_args
            assert 'tensorboard_log' not in call_args[1]
    
    def test_create_new_model_missing_required_args(self, mock_env):
        """Test creating new model with missing required arguments."""
        with pytest.raises(ValueError):
            create_new_model(env=mock_env, learning_rate=1e-4)  # Missing other required args


class TestModelLoadingFromCheckpoint:
    """Test model loading from checkpoint functionality."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        mock_env = Mock()
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 5, 5))
        mock_env.action_space = gym.spaces.Discrete(25)
        return mock_env
    
    def test_load_model_from_checkpoint_success(self, mock_env, temp_dir):
        """Test successful loading of model from checkpoint."""
        # Create a mock checkpoint file
        checkpoint_path = os.path.join(temp_dir, "model.zip")
        Path(checkpoint_path).touch()
        
        with patch('src.factories.model_factory.MaskablePPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.load.return_value = mock_model
            
            model = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                env=mock_env,
                device='cpu'
            )
            
            assert model == mock_model
            mock_ppo.load.assert_called_once_with(
                checkpoint_path,
                env=mock_env,
                device='cpu'
            )
    
    def test_load_model_from_checkpoint_with_tensorboard_log(self, mock_env, temp_dir):
        """Test loading model with tensorboard log setting."""
        checkpoint_path = os.path.join(temp_dir, "model.zip")
        Path(checkpoint_path).touch()
        
        with patch('src.factories.model_factory.MaskablePPO') as mock_ppo:
            mock_model = Mock()
            mock_ppo.load.return_value = mock_model
            
            model = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                env=mock_env,
                device='cpu',
                tensorboard_log="/tmp/logs"
            )
            
            assert model.tensorboard_log == "/tmp/logs"
    
    def test_load_model_from_checkpoint_nonexistent_file(self, mock_env, temp_dir):
        """Test loading model from non-existent checkpoint file."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.zip")
        
        with pytest.raises(ModelCreationError, match="Checkpoint file not found"):
            load_model_from_checkpoint(
                checkpoint_path=nonexistent_path,
                env=mock_env,
                device='cpu'
            )
    
    def test_load_model_from_checkpoint_load_error(self, mock_env, temp_dir):
        """Test handling of model loading errors."""
        checkpoint_path = os.path.join(temp_dir, "model.zip")
        Path(checkpoint_path).touch()
        
        with patch('src.factories.model_factory.MaskablePPO') as mock_ppo:
            mock_ppo.load.side_effect = Exception("Load failed")
            
            with pytest.raises(ModelCreationError, match="Failed to load model"):
                load_model_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    env=mock_env,
                    device='cpu'
                )


class TestVecNormalizeHandling:
    """Test VecNormalize stats handling functionality."""
    
    def test_load_vecnormalize_stats_success(self, temp_dir):
        """Test successful loading of VecNormalize stats."""
        # Create mock stats file
        stats_path = os.path.join(temp_dir, "stats.pkl")
        Path(stats_path).touch()
        
        mock_env = Mock()
        mock_updated_env = Mock()
        
        with patch('src.factories.model_factory.VecNormalize') as mock_vecnorm:
            mock_vecnorm.load.return_value = mock_updated_env
            
            # Import the function we'll create
            from src.factories.model_factory import load_vecnormalize_stats
            
            updated_env = load_vecnormalize_stats(mock_env, stats_path)
            
            assert updated_env == mock_updated_env
            mock_vecnorm.load.assert_called_once_with(stats_path, mock_env)
    
    def test_load_vecnormalize_stats_nonexistent_file(self, temp_dir):
        """Test loading VecNormalize stats from non-existent file."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent.pkl")
        mock_env = Mock()
        
        from src.factories.model_factory import load_vecnormalize_stats
        
        # Should return original environment when stats file doesn't exist
        result_env = load_vecnormalize_stats(mock_env, nonexistent_path)
        assert result_env == mock_env
    
    def test_load_vecnormalize_stats_none_path(self):
        """Test loading VecNormalize stats with None path."""
        mock_env = Mock()
        
        from src.factories.model_factory import load_vecnormalize_stats
        
        result_env = load_vecnormalize_stats(mock_env, None)
        assert result_env == mock_env


class TestCreateModelUnified:
    """Test the unified create_model function."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        mock_env = Mock()
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 5, 5))
        mock_env.action_space = gym.spaces.Discrete(25)
        return mock_env
    
    @pytest.fixture
    def sample_model_config(self):
        """Create sample model configuration."""
        return {
            'n_steps': 1024,
            'batch_size': 128,
            'n_epochs': 10,
            'learning_rate': 1e-4,
            'ent_coef': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.90,
            'clip_range': 0.2,
            'vf_coef': 1.0,
            'device': 'cpu',
            'seed': 42,
            'features_dim': 128,
            'pi_layers': [64, 64],
            'vf_layers': [256, 256]
        }
    
    def test_create_model_new_model(self, mock_env, sample_model_config):
        """Test creating new model through unified interface."""
        with patch('src.factories.model_factory.create_new_model') as mock_create_new:
            mock_model = Mock()
            mock_create_new.return_value = mock_model
            
            model, env = create_model(
                env=mock_env,
                tensorboard_log="/tmp/logs",
                **sample_model_config
            )
            
            assert model == mock_model
            assert env == mock_env
            mock_create_new.assert_called_once()
    
    def test_create_model_from_checkpoint(self, mock_env, sample_model_config, temp_dir):
        """Test creating model from checkpoint through unified interface."""
        checkpoint_path = os.path.join(temp_dir, "model.zip")
        Path(checkpoint_path).touch()
        
        with patch('src.factories.model_factory.load_model_from_checkpoint') as mock_load:
            with patch('src.factories.model_factory.load_vecnormalize_stats') as mock_load_stats:
                mock_model = Mock()
                mock_updated_env = Mock()
                mock_load.return_value = mock_model
                mock_load_stats.return_value = mock_updated_env
                
                model, env = create_model(
                    env=mock_env,
                    checkpoint_path=checkpoint_path,
                    vecnormalize_stats_path="/tmp/stats.pkl",
                    **sample_model_config
                )
                
                assert model == mock_model
                assert env == mock_updated_env
                mock_load.assert_called_once_with(
                    checkpoint_path=checkpoint_path,
                    env=mock_updated_env,  # Should be updated env, not original
                    device='cpu',
                    tensorboard_log=None
                )
                mock_load_stats.assert_called_once_with(
                    mock_env,
                    "/tmp/stats.pkl"
                )
    
    def test_create_model_invalid_config(self, mock_env):
        """Test creating model with invalid configuration."""
        with pytest.raises(ModelCreationError):
            create_model(
                env=mock_env,
                # Missing required configuration
                device='cpu'
            )


class TestModelCreationError:
    """Test custom exception handling."""
    
    def test_model_creation_error_instantiation(self):
        """Test that ModelCreationError can be instantiated."""
        error = ModelCreationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_model_creation_error_with_cause(self):
        """Test ModelCreationError with underlying cause."""
        original_error = ValueError("Original error")
        error = ModelCreationError("Wrapper error", original_error)
        
        assert str(error) == "Wrapper error"
        assert error.__cause__ == original_error


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def mock_full_env_setup(self):
        """Create a more realistic environment setup."""
        # Mock MinesweeperEnv
        mock_base_env = Mock()
        mock_base_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 5, 5))
        mock_base_env.action_space = gym.spaces.Discrete(25)
        
        # Mock DummyVecEnv
        mock_vec_env = Mock()
        mock_vec_env.observation_space = mock_base_env.observation_space
        mock_vec_env.action_space = mock_base_env.action_space
        
        # Mock VecNormalize
        mock_normalized_env = Mock()
        mock_normalized_env.observation_space = mock_base_env.observation_space
        mock_normalized_env.action_space = mock_base_env.action_space
        
        return mock_base_env, mock_vec_env, mock_normalized_env
    
    def test_train_py_workflow_simulation(self, mock_full_env_setup):
        """Test simulating the train.py workflow."""
        mock_base_env, mock_vec_env, mock_normalized_env = mock_full_env_setup
        
        # Simulate train.py model creation
        model_config = {
            'n_steps': 1024,
            'batch_size': 128,
            'n_epochs': 10,
            'learning_rate': 1e-4,
            'ent_coef': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.90,
            'clip_range': 0.2,
            'vf_coef': 1.0,
            'device': 'cpu',
            'seed': 42,
            'features_dim': 128,
            'pi_layers': [64, 64],
            'vf_layers': [256, 256]
        }
        
        with patch('src.factories.model_factory.create_new_model') as mock_create_new:
            mock_model = Mock()
            mock_create_new.return_value = mock_model
            
            model, env = create_model(
                env=mock_normalized_env,
                tensorboard_log="/tmp/logs",
                **model_config
            )
            
            assert model == mock_model
            assert env == mock_normalized_env
    
    def test_create_model_with_config_manager(self):
        """Test creating model through unified interface with ConfigManager."""
        config_manager = ConfigManager()
        config_manager.config.model_hyperparams.learning_rate = 2e-4
        config_manager.config.model_hyperparams.gamma = 0.95
        config_manager.config.network_architecture.features_dim = 256
        config_manager.config.network_architecture.pi_layers = [128, 64]
        config_manager.config.network_architecture.vf_layers = [512, 256]
        config_manager.config.training_execution.device = 'cpu'
        config_manager.config.training_execution.seed = 42
        
        mock_env = Mock()
        mock_env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 5, 5))
        mock_env.action_space = gym.spaces.Discrete(25)
        
        with patch('src.factories.model_factory.create_new_model') as mock_create_new:
            mock_model = Mock()
            mock_create_new.return_value = mock_model
            
            # Test the new interface we'll add
            model, env = create_model(env=mock_env, config_manager=config_manager)
            
            assert model == mock_model
            assert env == mock_env
            mock_create_new.assert_called_once()
    
    def test_play_py_workflow_simulation(self, mock_full_env_setup, temp_dir):
        """Test simulating the play.py workflow."""
        mock_base_env, mock_vec_env, mock_normalized_env = mock_full_env_setup
        
        # Create mock checkpoint
        checkpoint_path = os.path.join(temp_dir, "model.zip")
        Path(checkpoint_path).touch()
        
        with patch('src.factories.model_factory.load_model_from_checkpoint') as mock_load:
            with patch('src.factories.model_factory.load_vecnormalize_stats') as mock_load_stats:
                mock_model = Mock()
                mock_load.return_value = mock_model
                mock_load_stats.return_value = mock_normalized_env
                
                model, env = create_model(
                    env=mock_vec_env,
                    checkpoint_path=checkpoint_path,
                    vecnormalize_stats_path="/tmp/stats.pkl",
                    device='cpu'
                )
                
                assert model == mock_model
                assert env == mock_normalized_env