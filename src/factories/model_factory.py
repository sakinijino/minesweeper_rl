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


class ModelCreationError(Exception):
    """Custom exception for model creation errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.__cause__ = cause


def create_policy_kwargs(
    features_dim: int = 128,
    pi_layers: Optional[List[int]] = None,
    vf_layers: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Create policy kwargs for MaskablePPO model.
    
    Args:
        features_dim: Output dimension of the CNN features extractor
        pi_layers: Layer sizes for the policy network head
        vf_layers: Layer sizes for the value network head
        
    Returns:
        Dictionary containing policy kwargs for MaskablePPO
    """
    if pi_layers is None:
        pi_layers = [64, 64]
    if vf_layers is None:
        vf_layers = [256, 256]
    
    return {
        'features_extractor_class': CustomCNN,
        'features_extractor_kwargs': {'features_dim': features_dim},
        'net_arch': {'pi': pi_layers, 'vf': vf_layers}
    }


def create_new_model(
    env,
    n_steps: int,
    batch_size: int, 
    n_epochs: int,
    learning_rate: float,
    ent_coef: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    vf_coef: float,
    device: str,
    seed: Optional[int] = None,
    tensorboard_log: Optional[str] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> MaskablePPO:
    """
    Create a new MaskablePPO model with specified configuration.
    
    Args:
        env: Training environment
        n_steps: Number of steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of optimization epochs per update
        learning_rate: Learning rate
        ent_coef: Entropy coefficient
        gamma: Discount factor
        gae_lambda: Factor for Generalized Advantage Estimation
        clip_range: Clipping parameter for PPO
        vf_coef: Value function coefficient in the loss calculation
        device: Device to use for training (auto, cpu, cuda)
        seed: Random seed for reproducibility
        tensorboard_log: Path for tensorboard logging
        policy_kwargs: Policy configuration kwargs
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Configured MaskablePPO model
    """
    # Create model arguments
    model_args = {
        'policy': "CnnPolicy",
        'env': env,
        'verbose': 1,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'vf_coef': vf_coef,
        'device': device,
        'seed': seed
    }
    
    # Add policy kwargs if provided
    if policy_kwargs is not None:
        model_args['policy_kwargs'] = policy_kwargs
    
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
    device: str = 'cpu',
    checkpoint_path: Optional[str] = None,
    vecnormalize_stats_path: Optional[str] = None,
    tensorboard_log: Optional[str] = None,
    features_dim: int = 128,
    pi_layers: Optional[List[int]] = None,
    vf_layers: Optional[List[int]] = None,
    **model_config
) -> Tuple[MaskablePPO, Any]:
    """
    Unified model creation function.
    
    This function handles both new model creation and loading from checkpoint,
    providing a single interface for all model creation needs.
    
    Args:
        env: Environment for the model
        device: Device to use (auto, cpu, cuda)
        checkpoint_path: Path to checkpoint (if loading from checkpoint)
        vecnormalize_stats_path: Path to VecNormalize stats file
        tensorboard_log: Path for tensorboard logging
        features_dim: CNN features dimension
        pi_layers: Policy network layer sizes
        vf_layers: Value network layer sizes
        **model_config: Additional model configuration parameters
        
    Returns:
        Tuple of (model, environment) where environment may be updated with stats
        
    Raises:
        ModelCreationError: If model creation fails
    """
    try:
        # Load VecNormalize stats if available (do this first)
        updated_env = load_vecnormalize_stats(env, vecnormalize_stats_path)
        
        if checkpoint_path is not None:
            # Load from checkpoint
            model = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                env=updated_env,
                device=device,
                tensorboard_log=tensorboard_log
            )
        else:
            # Create new model
            # First create policy kwargs
            policy_kwargs = create_policy_kwargs(
                features_dim=features_dim,
                pi_layers=pi_layers,
                vf_layers=vf_layers
            )
            
            # Create the model
            model = create_new_model(
                env=updated_env,
                device=device,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                **model_config
            )
        
        return model, updated_env
        
    except Exception as e:
        if isinstance(e, ModelCreationError):
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
    tensorboard_log: str,
    features_dim: int = 128,
    pi_layers: Optional[List[int]] = None,
    vf_layers: Optional[List[int]] = None,
    **training_args
) -> MaskablePPO:
    """
    Convenience function for creating training models.
    
    Args:
        env: Training environment
        tensorboard_log: Path for tensorboard logging
        features_dim: CNN features dimension
        pi_layers: Policy network layer sizes
        vf_layers: Value network layer sizes
        **training_args: Training configuration arguments
        
    Returns:
        Configured MaskablePPO model for training
    """
    model, _ = create_model(
        env=env,
        tensorboard_log=tensorboard_log,
        features_dim=features_dim,
        pi_layers=pi_layers,
        vf_layers=vf_layers,
        **training_args
    )
    return model


def create_inference_model(
    env,
    checkpoint_path: str,
    vecnormalize_stats_path: Optional[str] = None,
    device: str = 'cpu'
) -> Tuple[MaskablePPO, Any]:
    """
    Convenience function for creating inference models.
    
    Args:
        env: Environment for inference
        checkpoint_path: Path to model checkpoint
        vecnormalize_stats_path: Path to VecNormalize stats file
        device: Device to use for inference
        
    Returns:
        Tuple of (model, environment) ready for inference
    """
    return create_model(
        env=env,
        checkpoint_path=checkpoint_path,
        vecnormalize_stats_path=vecnormalize_stats_path,
        device=device
    )