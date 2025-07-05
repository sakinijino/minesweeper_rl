"""
Model Factory Module

This module provides factory functions for creating and loading MaskablePPO models
with consistent configuration. It centralizes the model creation logic to eliminate
code duplication between train.py and play.py.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from ..env.custom_cnn import CustomCNN
from ..config.config_manager import ConfigManager


class ModelCreationError(Exception):
    """Custom exception for model creation errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.__cause__ = cause


def create_policy_kwargs(
    config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Create policy kwargs for MaskablePPO model.
    
    Args:
        config_manager: ConfigManager instance containing all configuration
        
    Returns:
        Dictionary containing policy kwargs for MaskablePPO
        
    Raises:
        ValueError: If config_manager is None or contains None values
    """
    if config_manager is None:
        raise ValueError("config_manager is required")
    
    net_arch = config_manager.config.network_architecture
    
    # Validate that ConfigManager has all required values
    required_values = ['features_dim', 'pi_layers', 'vf_layers']
    for attr in required_values:
        if getattr(net_arch, attr) is None:
            raise ValueError(f"ConfigManager.network_architecture.{attr} is None. "
                           f"ConfigManager must provide all default values.")
    
    return {
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': {'features_dim': net_arch.features_dim},
        'net_arch': {'pi': net_arch.pi_layers, 'vf': net_arch.vf_layers}
    }


def create_new_model(
    env,
    config_manager: ConfigManager,
    tensorboard_log: Optional[str] = None
) -> MaskablePPO:
    """
    Create a new MaskablePPO model with specified configuration.
    
    Args:
        env: Training environment
        config_manager: ConfigManager instance containing all configuration
        tensorboard_log: Path for tensorboard logging (optional)
        
    Returns:
        Configured MaskablePPO model
        
    Raises:
        ValueError: If config_manager is None or incomplete
    """
    if config_manager is None:
        raise ValueError("config_manager is required")
    
    model_hyperparams = config_manager.config.model_hyperparams
    training_execution = config_manager.config.training_execution
    
    # Validate that ConfigManager has all required values
    required_model_params = ['n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'ent_coef', 
                            'gamma', 'gae_lambda', 'clip_range', 'vf_coef']
    for attr in required_model_params:
        if getattr(model_hyperparams, attr) is None:
            raise ValueError(f"ConfigManager.model_hyperparams.{attr} is None. "
                           f"ConfigManager must provide all default values.")
    
    required_execution_params = ['device']
    for attr in required_execution_params:
        if getattr(training_execution, attr) is None:
            raise ValueError(f"ConfigManager.training_execution.{attr} is None. "
                           f"ConfigManager must provide all default values.")
    
    # Create policy kwargs
    policy_kwargs = create_policy_kwargs(config_manager=config_manager)
    
    # Create model arguments
    model_args = {
        'policy': "CnnPolicy",
        'env': env,
        'verbose': 1,
        'n_steps': model_hyperparams.n_steps,
        'batch_size': model_hyperparams.batch_size,
        'n_epochs': model_hyperparams.n_epochs,
        'learning_rate': model_hyperparams.learning_rate,
        'ent_coef': model_hyperparams.ent_coef,
        'gamma': model_hyperparams.gamma,
        'gae_lambda': model_hyperparams.gae_lambda,
        'clip_range': model_hyperparams.clip_range,
        'vf_coef': model_hyperparams.vf_coef,
        'device': training_execution.device,
        'seed': training_execution.seed,
        'policy_kwargs': policy_kwargs
    }
    
    # Add tensorboard logging if specified
    if tensorboard_log is not None:
        model_args['tensorboard_log'] = tensorboard_log
    
    # Create and return the model
    return MaskablePPO(**model_args)


def load_model_from_checkpoint(
    checkpoint_path: str,
    env,
    device: str,
    tensorboard_log: Optional[str] = None
) -> MaskablePPO:
    """
    Load a MaskablePPO model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        env: Environment to associate with the model
        device: Device to use for model (auto, cpu, cuda)
        tensorboard_log: Path for tensorboard logging (optional)
        
    Returns:
        Loaded MaskablePPO model
        
    Raises:
        ModelCreationError: If checkpoint loading fails
    """
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise ModelCreationError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        # Load the model
        model = MaskablePPO.load(
            checkpoint_path,
            env=env,
            device=device
        )
        
        # Set tensorboard logging if specified
        if tensorboard_log is not None:
            model.tensorboard_log = tensorboard_log
            
        return model
        
    except Exception as e:
        raise ModelCreationError(f"Failed to load model from checkpoint: {checkpoint_path}") from e


def load_vecnormalize_stats(
    env,
    stats_path: Optional[str]
) -> Any:
    """
    Load VecNormalize statistics if available.
    
    Args:
        env: Environment to load stats into
        stats_path: Path to VecNormalize stats file (optional)
        
    Returns:
        Environment with loaded stats, or original environment if stats not available
    """
    if stats_path is None or not os.path.exists(stats_path):
        return env
    
    try:
        # Load VecNormalize stats
        env_with_stats = VecNormalize.load(stats_path, env)
        return env_with_stats
    except Exception:
        # If loading fails, return original environment
        return env


def create_model(
    env,
    config_manager: ConfigManager,
    checkpoint_path: Optional[str] = None,
    vecnormalize_stats_path: Optional[str] = None,
    tensorboard_log: Optional[str] = None
) -> Tuple[MaskablePPO, Any]:
    """
    Unified model creation function.
    
    This function handles both new model creation and loading from checkpoint,
    providing a single interface for all model creation needs.
    
    Args:
        env: Environment for the model
        config_manager: ConfigManager instance containing all configuration
        checkpoint_path: Path to checkpoint (if loading from checkpoint)
        vecnormalize_stats_path: Path to VecNormalize stats file
        tensorboard_log: Path for tensorboard logging
        
    Returns:
        Tuple of (model, environment) where environment may be updated with stats
        
    Raises:
        ModelCreationError: If model creation fails
        ValueError: If config_manager is None
    """
    if config_manager is None:
        raise ValueError("config_manager is required")
    
    try:
        # Load VecNormalize stats if available (do this first)
        updated_env = load_vecnormalize_stats(env, vecnormalize_stats_path)
        
        if checkpoint_path is not None:
            # Load from checkpoint
            device = config_manager.config.training_execution.device
            model = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                env=updated_env,
                device=device,
                tensorboard_log=tensorboard_log
            )
        else:
            # Create new model
            model = create_new_model(
                env=updated_env,
                config_manager=config_manager,
                tensorboard_log=tensorboard_log
            )
        
        return model, updated_env
        
    except Exception as e:
        if isinstance(e, (ModelCreationError, ValueError)):
            raise
        else:
            raise ModelCreationError(f"Failed to create model: {str(e)}") from e


def get_model_summary(model: MaskablePPO) -> Dict[str, Any]:
    """
    Get a summary of model configuration.
    
    Args:
        model: MaskablePPO model
        
    Returns:
        Dictionary containing model configuration summary
    """
    return {
        'policy_class': model.policy_class.__name__,
        'device': str(model.device),
        'learning_rate': model.learning_rate,
        'n_steps': model.n_steps,
        'batch_size': model.batch_size,
        'n_epochs': model.n_epochs,
        'gamma': model.gamma,
        'gae_lambda': model.gae_lambda,
        'clip_range': model.clip_range,
        'ent_coef': model.ent_coef,
        'vf_coef': model.vf_coef,
        'seed': model.seed
    }


# For backward compatibility and convenience
def create_training_model(
    env,
    config_manager: ConfigManager,
    tensorboard_log: Optional[str] = None
) -> MaskablePPO:
    """
    Convenience function for creating training models.
    
    Args:
        env: Training environment
        config_manager: ConfigManager instance containing all configuration
        tensorboard_log: Path for tensorboard logging
        
    Returns:
        Configured MaskablePPO model for training
    """
    model, _ = create_model(
        env=env,
        config_manager=config_manager,
        tensorboard_log=tensorboard_log
    )
    return model


def create_inference_model(
    env,
    config_manager: ConfigManager,
    checkpoint_path: str,
    vecnormalize_stats_path: Optional[str] = None
) -> Tuple[MaskablePPO, Any]:
    """
    Convenience function for creating inference models.
    
    Args:
        env: Environment for inference
        config_manager: ConfigManager instance containing all configuration
        checkpoint_path: Path to model checkpoint
        vecnormalize_stats_path: Path to VecNormalize stats file
        
    Returns:
        Tuple of (model, environment) ready for inference
    """
    return create_model(
        env=env,
        config_manager=config_manager,
        checkpoint_path=checkpoint_path,
        vecnormalize_stats_path=vecnormalize_stats_path
    )