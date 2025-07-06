"""
Checkpoint utilities for Minesweeper RL training.

This module provides functions for discovering, selecting, and loading
training checkpoints and configurations.
"""

import os
import re
import glob
import json


def find_checkpoint_files(checkpoint_dir):
    """Find all checkpoint files in the given directory.
    
    Args:
        checkpoint_dir (str): Directory to search for checkpoint files
        
    Returns:
        list: List of checkpoint file paths found
    """
    checkpoint_files = []
    
    # Pattern for checkpoint files: rl_model_<steps>_steps.zip
    pattern = os.path.join(checkpoint_dir, "rl_model_*_steps.zip")
    checkpoint_files.extend(glob.glob(pattern))
    
    # Also check for final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
    if os.path.exists(final_model_path):
        checkpoint_files.append(final_model_path)
    
    return checkpoint_files


def extract_steps_from_checkpoint(checkpoint_path):
    """Extract the number of steps from checkpoint filename.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        int or float: Number of steps, or float('inf') for final_model.zip,
                     or 0 if extraction fails
    """
    filename = os.path.basename(checkpoint_path)
    
    # Handle final_model.zip - treat as highest step count
    if filename == "final_model.zip":
        return float('inf')
    
    # Extract steps from rl_model_<steps>_steps.zip pattern
    match = re.search(r'rl_model_(\d+)_steps\.zip', filename)
    if match:
        return int(match.group(1))
    
    return 0


def find_best_checkpoint(checkpoint_dir, target_steps=None):
    """Find the best checkpoint to continue training from.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        target_steps (int, optional): Target step count. If None, returns latest checkpoint
        
    Returns:
        str: Path to the best checkpoint file
        
    Raises:
        FileNotFoundError: If no checkpoint files are found
    """
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by steps
    checkpoint_files.sort(key=extract_steps_from_checkpoint)
    
    if target_steps is None:
        # Return the latest checkpoint
        return checkpoint_files[-1]
    else:
        # Find the checkpoint closest to but not exceeding target_steps
        best_checkpoint = None
        for checkpoint in checkpoint_files:
            steps = extract_steps_from_checkpoint(checkpoint)
            if steps == float('inf'):  # Skip final model unless it's the only option
                continue
            if steps <= target_steps:
                best_checkpoint = checkpoint
            else:
                break
        
        if best_checkpoint is None and checkpoint_files:
            # If no checkpoint found within target_steps, use the earliest one
            best_checkpoint = checkpoint_files[0]
        
        return best_checkpoint


def load_training_config(config_path):
    """Load training configuration from JSON file.
    
    Args:
        config_path (str): Path to training configuration JSON file
        
    Returns:
        dict or None: Training configuration dictionary, or None if loading fails
    """
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load training config from {config_path}: {e}")
        return None


def find_vecnormalize_stats(checkpoint_path):
    """Find the corresponding VecNormalize stats file for a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        str or None: Path to the VecNormalize stats file, or None if not found
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Determine expected stats filename based on checkpoint name
    if checkpoint_name == "final_model.zip":
        stats_candidates = ["final_stats_vecnormalize.pkl"]
    else:
        # Extract steps from checkpoint filename
        match = re.search(r'rl_model_(\d+)_steps\.zip', checkpoint_name)
        if match:
            steps = match.group(1)
            # Try different naming patterns that have been used
            stats_candidates = [
                f"rl_model_vecnormalize_{steps}_steps.pkl",  # New pattern
                f"vecnormalize_{steps}_steps.pkl",           # Alternative pattern
                "final_stats_vecnormalize.pkl"               # Fallback
            ]
        else:
            stats_candidates = ["final_stats_vecnormalize.pkl"]
    
    # Try each candidate and return the first one that exists
    for stats_filename in stats_candidates:
        stats_path = os.path.join(checkpoint_dir, stats_filename)
        if os.path.exists(stats_path):
            return stats_path
    
    return None


def find_all_experiment_dirs(training_runs_dir):
    """Find all experiment directories in a training runs folder.
    
    Args:
        training_runs_dir (str): Path to the training runs directory
        
    Returns:
        list: List of experiment directory paths sorted by timestamp (newest first)
    """
    if not os.path.exists(training_runs_dir):
        return []
    
    experiment_dirs = []
    
    # Scan all subdirectories
    for item in os.listdir(training_runs_dir):
        item_path = os.path.join(training_runs_dir, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
        
        # Skip hidden directories and common non-experiment folders
        if item.startswith('.') or item in ['README', 'docs', '__pycache__']:
            continue
        
        # Check if it looks like an experiment directory
        # Typically contains timestamp pattern: YYYYMMDD-HHMMSS (14 digits)
        # Common patterns: mw_ppo_5x5x3_seed42_20250705152544
        # or with continue: mw_ppo_5x5x3_seed42_20250705160718_continue_20250705163618
        
        # Look for timestamp pattern in directory name
        timestamp_found = False
        parts = item.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 14:
                timestamp_found = True
                break
        
        # If no timestamp pattern found, skip
        if not timestamp_found:
            continue
        
        # Check if it has a models subdirectory (valid experiment)
        models_dir = os.path.join(item_path, "models")
        if os.path.exists(models_dir):
            experiment_dirs.append(item_path)
    
    # Sort by timestamp (newest first)
    def get_latest_timestamp(dir_path):
        """Extract the latest timestamp from directory name."""
        dir_name = os.path.basename(dir_path)
        parts = dir_name.split('_')
        timestamps = [part for part in parts if part.isdigit() and len(part) == 14]
        return max(timestamps) if timestamps else '0'
    
    experiment_dirs.sort(key=get_latest_timestamp, reverse=True)
    
    return experiment_dirs