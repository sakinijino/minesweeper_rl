"""
Configuration manager for the minesweeper RL training system.

This module provides the ConfigManager class that handles loading, saving,
merging, and validating configurations. It supports loading from files,
command-line arguments, and provides utilities for configuration management.
"""

import json
import os
from typing import Dict, Any, Optional, Union
from dataclasses import asdict, fields
from argparse import Namespace

from .config_schemas import (
    TrainingConfig,
    ModelHyperparams,
    NetworkArchitecture,
    EnvironmentConfig,
    TrainingExecutionConfig,
    PathsConfig,
    PlayConfig,
    validate_training_config,
    create_config_from_dict
)


class ConfigManager:
    """
    Manages configuration loading, saving, and updating for the training system.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize ConfigManager with optional configuration.
        
        Args:
            config: Optional TrainingConfig instance. If None, creates default.
        """
        self.config = config or TrainingConfig()
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to JSON configuration file
            
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        self.config = create_config_from_dict(config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path where to save the configuration
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        config_dict = asdict(self.config)
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=4, sort_keys=True)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary with configuration updates
        """
        # Create a new config from current state + updates
        current_dict = asdict(self.config)
        
        # Deep merge updates
        for section, section_updates in updates.items():
            if section in current_dict and current_dict[section] is not None and isinstance(section_updates, dict):
                current_dict[section].update(section_updates)
            else:
                current_dict[section] = section_updates
        
        self.config = create_config_from_dict(current_dict)
    
    def update_from_args(self, args: Namespace) -> None:
        """
        Update configuration from argparse Namespace.
        
        Args:
            args: Parsed command-line arguments
        """
        # Map argument names to configuration sections
        arg_to_config_map = self._create_arg_mapping()
        
        updates = {}
        for arg_name, value in vars(args).items():
            if arg_name in arg_to_config_map and value is not None:
                section, param_name = arg_to_config_map[arg_name]
                if section not in updates:
                    updates[section] = {}
                updates[section][param_name] = value
        
        if updates:
            self.update_from_dict(updates)
    
    def _create_arg_mapping(self) -> Dict[str, tuple]:
        """
        Create mapping from argument names to configuration sections.
        
        Returns:
            Dict mapping argument names to (section, parameter) tuples
        """
        mapping = {}
        
        # Model hyperparameters
        for field in fields(ModelHyperparams):
            mapping[field.name] = ("model_hyperparams", field.name)
        
        # Network architecture
        for field in fields(NetworkArchitecture):
            mapping[field.name] = ("network_architecture", field.name)
        
        # Environment config
        for field in fields(EnvironmentConfig):
            mapping[field.name] = ("environment_config", field.name)
        
        # Training execution
        for field in fields(TrainingExecutionConfig):
            mapping[field.name] = ("training_execution", field.name)
        
        # Paths config
        for field in fields(PathsConfig):
            mapping[field.name] = ("paths_config", field.name)
        
        # Play config
        for field in fields(PlayConfig):
            if field.name != "environment_config":  # Skip nested config
                mapping[field.name] = ("play_config", field.name)
        
        return mapping
    
    def get_training_config(self) -> TrainingConfig:
        """
        Get the complete training configuration.
        
        Returns:
            TrainingConfig: Current training configuration
        """
        return self.config
    
    def get_environment_config(self) -> EnvironmentConfig:
        """
        Get the environment configuration.
        
        Returns:
            EnvironmentConfig: Current environment configuration
        """
        return self.config.environment_config
    
    def get_play_config(self) -> PlayConfig:
        """
        Get the play configuration.
        
        Returns:
            PlayConfig: Current play configuration
        """
        if self.config.play_config is None:
            # Create default play config with current environment config
            self.config.play_config = PlayConfig(
                environment_config=self.config.environment_config
            )
        return self.config.play_config
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        return validate_training_config(self.config)
    
    def merge_configs(self, other_config: TrainingConfig) -> TrainingConfig:
        """
        Merge another configuration with the current one.
        
        Args:
            other_config: Configuration to merge
            
        Returns:
            TrainingConfig: New merged configuration
        """
        # Convert both configs to dictionaries
        current_dict = asdict(self.config)
        other_dict = asdict(other_config)
        
        # Remove None values from other_dict to avoid overriding existing values
        other_dict = self._remove_none_values(other_dict)
        
        # Deep merge (other_config takes precedence)
        merged_dict = self._deep_merge(current_dict, other_dict)
        
        return create_config_from_dict(merged_dict)
    
    def _remove_none_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from dictionary recursively."""
        if not isinstance(d, dict):
            return d
        
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                cleaned_value = self._remove_none_values(value)
                if cleaned_value:  # Only add if not empty
                    result[key] = cleaned_value
            elif value is not None:
                result[key] = value
        
        return result
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)
            
        Returns:
            Dict: Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load_from_training_run(self, run_dir: str) -> None:
        """
        Load configuration from a training run directory.
        
        Args:
            run_dir: Path to training run directory
            
        Raises:
            FileNotFoundError: If training_config.json not found
        """
        config_path = os.path.join(run_dir, "training_config.json")
        self.load_from_file(config_path)
    
    @staticmethod
    def create_default_config() -> TrainingConfig:
        """
        Create a default training configuration.
        
        Returns:
            TrainingConfig: Default configuration
        """
        return TrainingConfig()
    
    @staticmethod
    def create_play_config(
        training_config: TrainingConfig,
        mode: str = "batch",
        num_episodes: int = 100,
        **kwargs
    ) -> PlayConfig:
        """
        Create play configuration from training configuration.
        
        Args:
            training_config: Source training configuration
            mode: Play mode
            num_episodes: Number of episodes
            **kwargs: Additional play configuration parameters
            
        Returns:
            PlayConfig: Play configuration with environment settings
        """
        play_config = PlayConfig(
            mode=mode,
            num_episodes=num_episodes,
            environment_config=training_config.environment_config,
            **kwargs
        )
        
        return play_config
    
    def get_model_hyperparams_dict(self) -> Dict[str, Any]:
        """
        Get model hyperparameters as dictionary suitable for model creation.
        
        Returns:
            Dict: Model hyperparameters
        """
        return asdict(self.config.model_hyperparams)
    
    def get_environment_params_dict(self) -> Dict[str, Any]:
        """
        Get environment parameters as dictionary suitable for environment creation.
        
        Returns:
            Dict: Environment parameters
        """
        return asdict(self.config.environment_config)
    
    def get_training_params_dict(self) -> Dict[str, Any]:
        """
        Get training parameters as dictionary suitable for training setup.
        
        Returns:
            Dict: Training parameters
        """
        return asdict(self.config.training_execution)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()