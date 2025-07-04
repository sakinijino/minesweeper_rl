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