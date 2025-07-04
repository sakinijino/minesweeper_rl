import pytest
import os
import json
from pathlib import Path

# Import the functions from the new checkpoint_utils module
from src.utils.checkpoint_utils import (
    find_checkpoint_files,
    extract_steps_from_checkpoint, 
    find_best_checkpoint,
    load_training_config,
    find_vecnormalize_stats
)


class TestFindCheckpointFiles:
    """Test checkpoint file discovery functionality."""
    
    def test_find_checkpoint_files_with_multiple_files(self, mock_checkpoint_dir):
        """Test finding multiple checkpoint files."""
        files = find_checkpoint_files(mock_checkpoint_dir)
        
        # Should find all checkpoint files
        assert len(files) == 4
        
        # Check that all expected files are found
        filenames = [os.path.basename(f) for f in files]
        expected_files = [
            "rl_model_50000_steps.zip",
            "rl_model_100000_steps.zip", 
            "rl_model_150000_steps.zip",
            "final_model.zip"
        ]
        
        for expected in expected_files:
            assert expected in filenames
    
    def test_find_checkpoint_files_empty_directory(self, empty_checkpoint_dir):
        """Test behavior with empty directory."""
        files = find_checkpoint_files(empty_checkpoint_dir)
        assert files == []
    
    def test_find_checkpoint_files_only_final_model(self, temp_dir):
        """Test directory with only final_model.zip."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create only final model
        final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
        Path(final_model_path).touch()
        
        files = find_checkpoint_files(checkpoint_dir)
        assert len(files) == 1
        assert os.path.basename(files[0]) == "final_model.zip"
    
    def test_find_checkpoint_files_nonexistent_directory(self, temp_dir):
        """Test behavior with non-existent directory."""
        nonexistent_dir = os.path.join(temp_dir, "does_not_exist")
        files = find_checkpoint_files(nonexistent_dir)
        assert files == []


class TestExtractStepsFromCheckpoint:
    """Test step number extraction from checkpoint filenames."""
    
    @pytest.mark.parametrize("filename,expected_steps", [
        ("rl_model_50000_steps.zip", 50000),
        ("rl_model_100000_steps.zip", 100000),
        ("rl_model_1500000_steps.zip", 1500000),
        ("final_model.zip", float('inf')),
        ("invalid_filename.zip", 0),
        ("rl_model_steps.zip", 0),  # Missing number
        ("rl_model_abc_steps.zip", 0),  # Non-numeric
    ])
    def test_extract_steps_from_checkpoint(self, filename, expected_steps):
        """Test step extraction with various filename patterns."""
        # Create a temporary file path for testing
        test_path = f"/fake/path/{filename}"
        result = extract_steps_from_checkpoint(test_path)
        assert result == expected_steps


class TestFindBestCheckpoint:
    """Test best checkpoint selection logic."""
    
    def test_find_best_checkpoint_latest(self, mock_checkpoint_dir):
        """Test finding the latest checkpoint (no target_steps specified)."""
        best_checkpoint = find_best_checkpoint(mock_checkpoint_dir)
        
        # Should return final_model.zip as it has highest priority (inf steps)
        assert os.path.basename(best_checkpoint) == "final_model.zip"
    
    def test_find_best_checkpoint_with_target_steps(self, mock_checkpoint_dir):
        """Test finding checkpoint with specific target steps."""
        # Target 75000 steps - should return 50000 (closest without exceeding)
        best_checkpoint = find_best_checkpoint(mock_checkpoint_dir, target_steps=75000)
        assert os.path.basename(best_checkpoint) == "rl_model_50000_steps.zip"
        
        # Target 100000 steps - should return exactly 100000
        best_checkpoint = find_best_checkpoint(mock_checkpoint_dir, target_steps=100000)
        assert os.path.basename(best_checkpoint) == "rl_model_100000_steps.zip"
        
        # Target 200000 steps - should return 150000 (highest available)
        best_checkpoint = find_best_checkpoint(mock_checkpoint_dir, target_steps=200000)
        assert os.path.basename(best_checkpoint) == "rl_model_150000_steps.zip"
    
    def test_find_best_checkpoint_target_too_low(self, mock_checkpoint_dir):
        """Test behavior when target_steps is lower than any checkpoint."""
        # Target 10000 steps - should return the earliest checkpoint (50000)
        best_checkpoint = find_best_checkpoint(mock_checkpoint_dir, target_steps=10000)
        assert os.path.basename(best_checkpoint) == "rl_model_50000_steps.zip"
    
    def test_find_best_checkpoint_empty_directory(self, empty_checkpoint_dir):
        """Test behavior with empty directory."""
        with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
            find_best_checkpoint(empty_checkpoint_dir)
    
    def test_find_best_checkpoint_only_final_model(self, temp_dir):
        """Test directory with only final_model.zip."""
        checkpoint_dir = os.path.join(temp_dir, "models") 
        os.makedirs(checkpoint_dir)
        
        final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
        Path(final_model_path).touch()
        
        # Should return final_model.zip for latest
        best_checkpoint = find_best_checkpoint(checkpoint_dir)
        assert os.path.basename(best_checkpoint) == "final_model.zip"
        
        # Should skip final_model.zip when target_steps specified, but since it's the only file,
        # should still return it as fallback
        best_checkpoint = find_best_checkpoint(checkpoint_dir, target_steps=50000)
        assert os.path.basename(best_checkpoint) == "final_model.zip"


class TestLoadTrainingConfig:
    """Test training configuration loading."""
    
    def test_load_training_config_valid_file(self, mock_training_run_dir):
        """Test loading valid training configuration."""
        config_path = os.path.join(mock_training_run_dir, "training_config.json")
        config = load_training_config(config_path)
        
        assert config is not None
        assert config["total_timesteps"] == 100000
        assert config["n_envs"] == 4
        assert config["learning_rate"] == 0.0001
        assert config["width"] == 5
        assert config["height"] == 5
        assert config["n_mines"] == 3
        assert config["seed"] == 42
    
    def test_load_training_config_nonexistent_file(self, temp_dir):
        """Test behavior with non-existent config file."""
        nonexistent_path = os.path.join(temp_dir, "nonexistent_config.json")
        config = load_training_config(nonexistent_path)
        assert config is None
    
    def test_load_training_config_invalid_json(self, temp_dir):
        """Test behavior with invalid JSON file."""
        invalid_json_path = os.path.join(temp_dir, "invalid_config.json")
        
        # Create file with invalid JSON
        with open(invalid_json_path, 'w') as f:
            f.write("{ invalid json content")
        
        config = load_training_config(invalid_json_path)
        assert config is None
    
    def test_load_training_config_empty_file(self, temp_dir):
        """Test behavior with empty JSON file."""
        empty_json_path = os.path.join(temp_dir, "empty_config.json")
        
        # Create empty JSON file
        with open(empty_json_path, 'w') as f:
            json.dump({}, f)
        
        config = load_training_config(empty_json_path)
        assert config == {}


class TestPathHandling:
    """Test path handling edge cases."""
    
    @pytest.mark.parametrize("input_path,expected_basename", [
        ("/path/to/training_run/", "training_run"),
        ("/path/to/training_run", "training_run"),
        ("./training_run/", "training_run"),
        ("training_run", "training_run"),
        ("/path/with/trailing/slash/", "slash"),
    ])
    def test_path_basename_handling(self, input_path, expected_basename):
        """Test that path handling works correctly with trailing slashes."""
        # This tests the os.path.basename(os.path.normpath()) pattern we use
        import os
        result = os.path.basename(os.path.normpath(input_path))
        assert result == expected_basename


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_checkpoint_discovery_workflow(self, mock_training_run_dir):
        """Test complete workflow of discovering and selecting checkpoints."""
        # Simulate the train.py logic for continue_from parameter
        
        # Step 1: Check if it's a training run directory
        models_dir = os.path.join(mock_training_run_dir, "models")
        assert os.path.exists(models_dir)
        
        # Step 2: Find checkpoints in models directory
        checkpoint_files = find_checkpoint_files(models_dir)
        assert len(checkpoint_files) > 0
        
        # Step 3: Select best checkpoint
        best_checkpoint = find_best_checkpoint(models_dir)
        assert best_checkpoint is not None
        assert os.path.basename(best_checkpoint) == "final_model.zip"
        
        # Step 4: Load training config
        config_path = os.path.join(mock_training_run_dir, "training_config.json")
        config = load_training_config(config_path)
        assert config is not None
        assert "total_timesteps" in config
    
    def test_continue_from_specific_step_workflow(self, mock_training_run_dir):
        """Test workflow for continuing from specific step."""
        models_dir = os.path.join(mock_training_run_dir, "models")
        
        # Continue from step 50000
        best_checkpoint = find_best_checkpoint(models_dir, target_steps=50000)
        assert os.path.basename(best_checkpoint) == "rl_model_50000_steps.zip"
        
        # Continue from step 60000 (should get 50000 as closest)
        best_checkpoint = find_best_checkpoint(models_dir, target_steps=60000)
        assert os.path.basename(best_checkpoint) == "rl_model_50000_steps.zip"


class TestFindVecnormalizeStats:
    """Test VecNormalize stats file discovery functionality."""
    
    def test_find_vecnormalize_stats_final_model(self, temp_dir):
        """Test finding stats for final_model.zip."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create final model and corresponding stats
        final_checkpoint = os.path.join(checkpoint_dir, "final_model.zip")
        final_stats = os.path.join(checkpoint_dir, "final_stats_vecnormalize.pkl")
        
        Path(final_checkpoint).touch()
        Path(final_stats).touch()
        
        stats_path = find_vecnormalize_stats(final_checkpoint)
        assert stats_path is not None
        assert os.path.basename(stats_path) == "final_stats_vecnormalize.pkl"
    
    def test_find_vecnormalize_stats_step_checkpoint_new_pattern(self, temp_dir):
        """Test finding stats for step checkpoint with new naming pattern."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create step checkpoint and corresponding stats (new pattern)
        checkpoint = os.path.join(checkpoint_dir, "rl_model_50000_steps.zip")
        stats = os.path.join(checkpoint_dir, "rl_model_vecnormalize_50000_steps.pkl")
        
        Path(checkpoint).touch()
        Path(stats).touch()
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is not None
        assert os.path.basename(stats_path) == "rl_model_vecnormalize_50000_steps.pkl"
    
    def test_find_vecnormalize_stats_step_checkpoint_alt_pattern(self, temp_dir):
        """Test finding stats for step checkpoint with alternative naming pattern."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create step checkpoint and corresponding stats (alternative pattern)
        checkpoint = os.path.join(checkpoint_dir, "rl_model_100000_steps.zip")
        stats = os.path.join(checkpoint_dir, "vecnormalize_100000_steps.pkl")
        
        Path(checkpoint).touch()
        Path(stats).touch()
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is not None
        assert os.path.basename(stats_path) == "vecnormalize_100000_steps.pkl"
    
    def test_find_vecnormalize_stats_fallback_to_final(self, temp_dir):
        """Test fallback to final stats when step-specific stats not found."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create step checkpoint but only final stats
        checkpoint = os.path.join(checkpoint_dir, "rl_model_75000_steps.zip")
        final_stats = os.path.join(checkpoint_dir, "final_stats_vecnormalize.pkl")
        
        Path(checkpoint).touch()
        Path(final_stats).touch()
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is not None
        assert os.path.basename(stats_path) == "final_stats_vecnormalize.pkl"
    
    def test_find_vecnormalize_stats_no_stats_found(self, temp_dir):
        """Test behavior when no stats files are found."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create checkpoint but no stats files
        checkpoint = os.path.join(checkpoint_dir, "rl_model_50000_steps.zip")
        Path(checkpoint).touch()
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is None
    
    def test_find_vecnormalize_stats_priority_order(self, temp_dir):
        """Test that the function follows the correct priority order."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create step checkpoint and multiple stats files
        checkpoint = os.path.join(checkpoint_dir, "rl_model_50000_steps.zip")
        new_pattern_stats = os.path.join(checkpoint_dir, "rl_model_vecnormalize_50000_steps.pkl")
        alt_pattern_stats = os.path.join(checkpoint_dir, "vecnormalize_50000_steps.pkl")
        final_stats = os.path.join(checkpoint_dir, "final_stats_vecnormalize.pkl")
        
        Path(checkpoint).touch()
        Path(alt_pattern_stats).touch()
        Path(final_stats).touch()
        Path(new_pattern_stats).touch()  # Create this last to ensure it's not just picking the newest
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is not None
        # Should prefer new pattern over alternative pattern
        assert os.path.basename(stats_path) == "rl_model_vecnormalize_50000_steps.pkl"
    
    def test_find_vecnormalize_stats_invalid_checkpoint_name(self, temp_dir):
        """Test behavior with invalid checkpoint filename."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        # Create checkpoint with invalid name
        checkpoint = os.path.join(checkpoint_dir, "invalid_checkpoint.zip")
        final_stats = os.path.join(checkpoint_dir, "final_stats_vecnormalize.pkl")
        
        Path(checkpoint).touch()
        Path(final_stats).touch()
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is not None
        # Should fallback to final stats for invalid checkpoint names
        assert os.path.basename(stats_path) == "final_stats_vecnormalize.pkl"
    
    @pytest.mark.parametrize("checkpoint_name,expected_stats_candidates", [
        ("final_model.zip", ["final_stats_vecnormalize.pkl"]),
        ("rl_model_25000_steps.zip", [
            "rl_model_vecnormalize_25000_steps.pkl",
            "vecnormalize_25000_steps.pkl", 
            "final_stats_vecnormalize.pkl"
        ]),
        ("rl_model_100000_steps.zip", [
            "rl_model_vecnormalize_100000_steps.pkl",
            "vecnormalize_100000_steps.pkl",
            "final_stats_vecnormalize.pkl"
        ]),
        ("invalid_name.zip", ["final_stats_vecnormalize.pkl"]),
    ])
    def test_find_vecnormalize_stats_candidate_generation(self, temp_dir, checkpoint_name, expected_stats_candidates):
        """Test that correct stats filename candidates are generated."""
        checkpoint_dir = os.path.join(temp_dir, "models")
        os.makedirs(checkpoint_dir)
        
        checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
        Path(checkpoint).touch()
        
        # Create only the last candidate (fallback)
        last_candidate = expected_stats_candidates[-1]
        stats_file = os.path.join(checkpoint_dir, last_candidate)
        Path(stats_file).touch()
        
        stats_path = find_vecnormalize_stats(checkpoint)
        assert stats_path is not None
        assert os.path.basename(stats_path) == last_candidate