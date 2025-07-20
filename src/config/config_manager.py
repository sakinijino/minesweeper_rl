"""
Configuration manager for the minesweeper RL training system.

This module provides the ConfigManager class that handles loading, merging,
and validating configurations with a priority system:
1. Command-line arguments (highest priority)
2. Configuration files
3. Continue training parameters (lowest priority)

Supports YAML and JSON configuration files.
"""

import json
import os
import yaml
import dataclasses
from typing import Dict, Any, Optional, Union, List
from dataclasses import asdict, fields
from argparse import Namespace

from .config_schemas import (
    TrainingConfig,
    ModelHyperparams,
    NetworkArchitecture,
    EnvironmentConfig,
    DynamicEnvironmentConfig,
    BoardSizeConfig,
    CurriculumConfig,
    TrainingExecutionConfig,
    PathsConfig,
    PlayConfig,
    validate_training_config,
    create_config_from_dict
)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""
    pass


class ConfigManager:
    """
    Manages configuration loading, merging, and validation with priority system.
    
    Priority order (highest to lowest):
    1. Command-line arguments
    2. Configuration file
    3. Continue training parameters
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_file: Optional path to configuration file to load
            
        Raises:
            ConfigurationError: If no configuration source is provided
        """
        self.config_sources = {
            'file': None,
            'args': None,
            'continue_train': None
        }
        
        
        if config_file:
            self.load_config_file(config_file)
            
        self.config = None
    
    def load_config_file(self, file_path: str) -> None:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            file_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If file does not exist
            ConfigurationError: If file format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    config_dict = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_path}")
            
            self.config_sources['file'] = config_dict
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Invalid configuration file format: {e}")
    
    def load_from_args(self, args: Namespace) -> None:
        """
        Load configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Map argument names to configuration sections
        arg_to_config_map = self._create_arg_mapping()
        
        config_dict = {}
        for arg_name, value in vars(args).items():
            if arg_name in arg_to_config_map and value is not None:
                section, param_name = arg_to_config_map[arg_name]
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][param_name] = value
        
        if config_dict:
            self.config_sources['args'] = config_dict
    
    def load_from_training_run(self, run_dir: str) -> None:
        """
        Load configuration from a training run directory.
        
        Args:
            run_dir: Path to training run directory
            
        Raises:
            FileNotFoundError: If training_config.json not found
        """
        config_path = os.path.join(run_dir, "training_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Training config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        self.config_sources['continue_train'] = config_dict
    
    def build_config(self) -> TrainingConfig:
        """
        Build final configuration by merging all sources with priority.
        
        Returns:
            TrainingConfig: Complete configuration object
            
        Raises:
            ConfigurationError: If required parameters are missing
        """
        # Start with empty config
        merged_config = {}
        
        # Apply sources in reverse priority order
        for source_name in ['continue_train', 'file', 'args']:
            source_config = self.config_sources[source_name]
            if source_config:
                merged_config = self._deep_merge(merged_config, source_config)
        
        # Validate that all required parameters are present
        self._validate_required_params(merged_config)
        
        # Create config object
        self.config = create_config_from_dict(merged_config)
        
        # Validate configuration
        if not validate_training_config(self.config):
            raise ConfigurationError("Configuration validation failed")
        
        return self.config
    
    def _validate_required_params(self, config_dict: Dict[str, Any]) -> None:
        """
        Validate that all required parameters are present.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If required parameters are missing
        """
        required_sections = {
            'model_hyperparams': ModelHyperparams,
            'network_architecture': NetworkArchitecture,
            'training_execution': TrainingExecutionConfig,
            'paths_config': PathsConfig
        }
        
        missing_params = []
        
        # Check basic required sections
        for section_name, section_class in required_sections.items():
            if section_name not in config_dict:
                missing_params.append(f"Missing section: {section_name}")
                continue
            
            section_dict = config_dict[section_name]
            for field in fields(section_class):
                # Skip fields with default values
                if field.name not in section_dict and field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING:
                    missing_params.append(f"Missing parameter: {section_name}.{field.name}")
        
        # Check environment configuration - can be either traditional or dynamic
        if 'environment_config' not in config_dict and 'dynamic_environment_config' not in config_dict:
            missing_params.append("Missing environment configuration: need either 'environment_config' or 'dynamic_environment_config'")
        elif 'environment_config' in config_dict:
            # Validate traditional environment config
            env_dict = config_dict['environment_config']
            for field in fields(EnvironmentConfig):
                if field.name not in env_dict and field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING:
                    missing_params.append(f"Missing parameter: environment_config.{field.name}")
        
        if missing_params:
            raise ConfigurationError(f"Required parameters missing:\\n" + "\\n".join(missing_params))
    
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
        
        # PlayConfig is no longer managed by ConfigManager
        # play.py will handle play configuration directly
        
        return mapping
    
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
    
    def save_config(self, file_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path where to save the configuration
            
        Raises:
            ConfigurationError: If no configuration has been built
        """
        if self.config is None:
            raise ConfigurationError("No configuration to save. Call build_config() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        config_dict = asdict(self.config)
        
        if file_path.endswith(('.yaml', '.yml')):
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)
        else:
            raise ConfigurationError(f"Unsupported file format: {file_path}")
    
    def get_config(self) -> TrainingConfig:
        """
        Get the current configuration.
        
        Returns:
            TrainingConfig: Current configuration
            
        Raises:
            ConfigurationError: If no configuration has been built
        """
        if self.config is None:
            raise ConfigurationError("No configuration available. Call build_config() first.")
        
        return self.config
    
    
    @staticmethod
    def create_from_config_file(config_file: str) -> 'ConfigManager':
        """
        Create ConfigManager from configuration file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            ConfigManager: Configured manager instance
        """
        manager = ConfigManager(config_file)
        manager.build_config()
        return manager
    
    @staticmethod
    def create_with_args(config_file: str, args: Namespace) -> 'ConfigManager':
        """
        Create ConfigManager with config file and command-line arguments.
        
        Args:
            config_file: Path to configuration file
            args: Command-line arguments
            
        Returns:
            ConfigManager: Configured manager instance
        """
        manager = ConfigManager(config_file)
        manager.load_from_args(args)
        manager.build_config()
        return manager
    
    def is_dynamic_environment(self) -> bool:
        """
        Check if the current configuration uses dynamic environment settings.
        
        Returns:
            bool: True if dynamic environment config is used
        """
        if self.config is None:
            raise ConfigurationError("No configuration available. Call build_config() first.")
        
        return isinstance(self.config.environment_config, DynamicEnvironmentConfig)
    
    def get_environment_config(self) -> Union[EnvironmentConfig, DynamicEnvironmentConfig]:
        """
        Get the environment configuration (either traditional or dynamic).
        
        Returns:
            Union[EnvironmentConfig, DynamicEnvironmentConfig]: Environment configuration
        """
        if self.config is None:
            raise ConfigurationError("No configuration available. Call build_config() first.")
        
        return self.config.environment_config
    
    def get_current_board_size(self, timestep: int = 0) -> Optional[BoardSizeConfig]:
        """
        Get the current board size based on curriculum progress.
        
        Args:
            timestep: Current training timestep
            
        Returns:
            Optional[BoardSizeConfig]: Current board size configuration
        """
        if not self.is_dynamic_environment():
            # Convert traditional config to board size config
            env_config = self.config.environment_config
            return BoardSizeConfig(
                width=env_config.width,
                height=env_config.height,
                n_mines=env_config.n_mines
            )
        
        dynamic_config = self.config.environment_config
        
        if dynamic_config.curriculum is not None and dynamic_config.curriculum.enabled:
            # Calculate curriculum progression
            curriculum = dynamic_config.curriculum
            
            # Determine current curriculum step
            current_step = min(timestep // curriculum.step_duration, curriculum.progression_steps - 1)
            progress = current_step / max(curriculum.progression_steps - 1, 1)
            
            # Linear interpolation between start and end sizes
            start_size = curriculum.start_size
            end_size = curriculum.end_size
            
            if curriculum.progression_type == "linear":
                width = int(start_size.width + (end_size.width - start_size.width) * progress)
                height = int(start_size.height + (end_size.height - start_size.height) * progress)
                n_mines = int(start_size.n_mines + (end_size.n_mines - start_size.n_mines) * progress)
            elif curriculum.progression_type == "exponential":
                # Exponential progression (slower at first, faster later)
                exp_progress = progress ** 2
                width = int(start_size.width + (end_size.width - start_size.width) * exp_progress)
                height = int(start_size.height + (end_size.height - start_size.height) * exp_progress)
                n_mines = int(start_size.n_mines + (end_size.n_mines - start_size.n_mines) * exp_progress)
            else:
                # Default to linear
                width = int(start_size.width + (end_size.width - start_size.width) * progress)
                height = int(start_size.height + (end_size.height - start_size.height) * progress)
                n_mines = int(start_size.n_mines + (end_size.n_mines - start_size.n_mines) * progress)
            
            return BoardSizeConfig(width=width, height=height, n_mines=n_mines)
        
        elif dynamic_config.board_sizes is not None:
            # Random sampling or return first size
            return dynamic_config.board_sizes[0]
        
        return None
    
    def get_reward_config(self) -> Dict[str, float]:
        """
        Get reward configuration for environment creation.
        
        Returns:
            Dict[str, float]: Reward configuration dictionary
        """
        if self.config is None:
            raise ConfigurationError("No configuration available. Call build_config() first.")
        
        if isinstance(self.config.environment_config, DynamicEnvironmentConfig):
            dynamic_config = self.config.environment_config
            return {
                'reward_win': dynamic_config.reward_win,
                'reward_lose': dynamic_config.reward_lose,
                'reward_reveal': dynamic_config.reward_reveal,
                'reward_invalid': dynamic_config.reward_invalid,
                'max_reward_per_step': dynamic_config.max_reward_per_step
            }
        else:
            env_config = self.config.environment_config
            return {
                'reward_win': env_config.reward_win,
                'reward_lose': env_config.reward_lose,
                'reward_reveal': env_config.reward_reveal,
                'reward_invalid': env_config.reward_invalid,
                'max_reward_per_step': env_config.max_reward_per_step
            }
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ConfigManager(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()