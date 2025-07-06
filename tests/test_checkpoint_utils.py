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
    find_vecnormalize_stats,
    find_all_experiment_dirs,
    find_latest_experiment_dir,
    resolve_model_paths_from_run_dir,
    resolve_continue_training_paths
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
        # Check nested structure (new config format)
        assert config["training_execution"]["total_timesteps"] == 100000
        assert config["training_execution"]["n_envs"] == 4
        assert config["model_hyperparams"]["learning_rate"] == 0.0001
        assert config["environment_config"]["width"] == 5
        assert config["environment_config"]["height"] == 5
        assert config["environment_config"]["n_mines"] == 3
        assert config["training_execution"]["seed"] == 42
    
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
        assert "training_execution" in config
        assert "total_timesteps" in config["training_execution"]
    
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


class TestFindAllExperimentDirs:
    """Test find_all_experiment_dirs functionality."""
    
    def test_find_all_experiment_dirs_basic(self, temp_dir):
        """Test finding all experiment directories in training_runs folder."""
        training_runs_dir = os.path.join(temp_dir, "training_runs")
        os.makedirs(training_runs_dir)
        
        # Create mock experiment directories with typical naming pattern
        experiment_names = [
            "mw_ppo_5x5x3_seed42_20250705152544",
            "mw_ppo_8x8x10_seed123_20250706081004",
            "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705163618"
        ]
        
        for exp_name in experiment_names:
            exp_dir = os.path.join(training_runs_dir, exp_name)
            models_dir = os.path.join(exp_dir, "models")
            os.makedirs(models_dir)
            
            # Create a final model in each
            Path(os.path.join(models_dir, "final_model.zip")).touch()
        
        # Test the function
        experiment_dirs = find_all_experiment_dirs(training_runs_dir)
        
        assert len(experiment_dirs) == 3
        assert all(os.path.isdir(exp_dir) for exp_dir in experiment_dirs)
        
        # Check that all expected directories are found
        exp_dir_names = [os.path.basename(d) for d in experiment_dirs]
        for exp_name in experiment_names:
            assert exp_name in exp_dir_names
    
    def test_find_all_experiment_dirs_empty(self, temp_dir):
        """Test behavior with empty training_runs directory."""
        training_runs_dir = os.path.join(temp_dir, "training_runs")
        os.makedirs(training_runs_dir)
        
        experiment_dirs = find_all_experiment_dirs(training_runs_dir)
        assert experiment_dirs == []
    
    def test_find_all_experiment_dirs_nonexistent(self, temp_dir):
        """Test behavior with non-existent directory."""
        nonexistent_dir = os.path.join(temp_dir, "does_not_exist")
        
        experiment_dirs = find_all_experiment_dirs(nonexistent_dir)
        assert experiment_dirs == []
    
    def test_find_all_experiment_dirs_filters_non_experiments(self, temp_dir):
        """Test that non-experiment directories are filtered out."""
        training_runs_dir = os.path.join(temp_dir, "training_runs")
        os.makedirs(training_runs_dir)
        
        # Create valid experiment directory
        exp_dir = os.path.join(training_runs_dir, "mw_ppo_5x5x3_seed42_20250705152544")
        models_dir = os.path.join(exp_dir, "models")
        os.makedirs(models_dir)
        
        # Create non-experiment directories/files
        os.makedirs(os.path.join(training_runs_dir, "README"))
        Path(os.path.join(training_runs_dir, "config.json")).touch()
        os.makedirs(os.path.join(training_runs_dir, ".git"))
        
        experiment_dirs = find_all_experiment_dirs(training_runs_dir)
        
        # Should only find the valid experiment directory
        assert len(experiment_dirs) == 1
        assert os.path.basename(experiment_dirs[0]) == "mw_ppo_5x5x3_seed42_20250705152544"
    
    def test_find_all_experiment_dirs_sorted_by_timestamp(self, temp_dir):
        """Test that experiment directories are sorted by timestamp (newest first)."""
        training_runs_dir = os.path.join(temp_dir, "training_runs")
        os.makedirs(training_runs_dir)
        
        # Create experiment directories with different timestamps
        experiment_names = [
            "mw_ppo_5x5x3_seed42_20250705152544",  # oldest
            "mw_ppo_5x5x3_seed42_20250706081004",  # newest
            "mw_ppo_5x5x3_seed42_20250705160718"   # middle
        ]
        
        for exp_name in experiment_names:
            exp_dir = os.path.join(training_runs_dir, exp_name)
            models_dir = os.path.join(exp_dir, "models")
            os.makedirs(models_dir)
        
        experiment_dirs = find_all_experiment_dirs(training_runs_dir)
        
        # Should be sorted by timestamp (newest first)
        exp_dir_names = [os.path.basename(d) for d in experiment_dirs]
        assert exp_dir_names[0] == "mw_ppo_5x5x3_seed42_20250706081004"  # newest
        assert exp_dir_names[1] == "mw_ppo_5x5x3_seed42_20250705160718"  # middle
        assert exp_dir_names[2] == "mw_ppo_5x5x3_seed42_20250705152544"  # oldest
    
    def test_find_all_experiment_dirs_with_continue_training(self, temp_dir):
        """Test handling of continue training directories."""
        training_runs_dir = os.path.join(temp_dir, "training_runs")
        os.makedirs(training_runs_dir)
        
        # Create original and continue training directories
        exp_names = [
            "mw_ppo_5x5x3_seed42_20250705160718",
            "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705163618",
            "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705170000"
        ]
        
        for exp_name in exp_names:
            exp_dir = os.path.join(training_runs_dir, exp_name)
            models_dir = os.path.join(exp_dir, "models")
            os.makedirs(models_dir)
        
        experiment_dirs = find_all_experiment_dirs(training_runs_dir)
        
        # Should find all three as separate experiments
        assert len(experiment_dirs) == 3
        
        # Check they're sorted by timestamp (newest continue first)
        exp_dir_names = [os.path.basename(d) for d in experiment_dirs]
        assert exp_dir_names[0] == "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705170000"
        assert exp_dir_names[1] == "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705163618"
        assert exp_dir_names[2] == "mw_ppo_5x5x3_seed42_20250705160718"


class TestFindLatestExperimentDir:
    """Test find_latest_experiment_dir functionality."""
    
    def test_find_latest_experiment_dir_basic(self, temp_dir):
        """Test finding the latest experiment directory."""
        experiment_base_dir = os.path.join(temp_dir, "experiments")
        os.makedirs(experiment_base_dir)
        
        # Create experiment directories with different timestamps
        exp_dirs = [
            "mw_ppo_5x5x3_seed42_20250705152544",  # oldest
            "mw_ppo_5x5x3_seed42_20250706081004",  # newest
            "mw_ppo_5x5x3_seed42_20250705160718"   # middle
        ]
        
        for exp_dir in exp_dirs:
            os.makedirs(os.path.join(experiment_base_dir, exp_dir))
        
        latest_dir = find_latest_experiment_dir(experiment_base_dir)
        
        # Should return the directory with latest timestamp
        assert os.path.basename(latest_dir) == "mw_ppo_5x5x3_seed42_20250706081004"
        assert os.path.exists(latest_dir)
    
    def test_find_latest_experiment_dir_no_experiments(self, temp_dir):
        """Test behavior when no experiment directories exist."""
        experiment_base_dir = os.path.join(temp_dir, "experiments")
        os.makedirs(experiment_base_dir)
        
        with pytest.raises(FileNotFoundError, match="No experiment directories found"):
            find_latest_experiment_dir(experiment_base_dir)
    
    def test_find_latest_experiment_dir_nonexistent_base(self, temp_dir):
        """Test behavior when base directory doesn't exist."""
        nonexistent_dir = os.path.join(temp_dir, "does_not_exist")
        
        with pytest.raises(FileNotFoundError, match="Experiment directory does not exist"):
            find_latest_experiment_dir(nonexistent_dir)
    
    def test_find_latest_experiment_dir_with_continue(self, temp_dir):
        """Test with continue training directories."""
        experiment_base_dir = os.path.join(temp_dir, "experiments")
        os.makedirs(experiment_base_dir)
        
        # Create directories including continue training
        exp_dirs = [
            "mw_ppo_5x5x3_seed42_20250705160718",
            "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705163618",
            "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705170000"  # latest
        ]
        
        for exp_dir in exp_dirs:
            os.makedirs(os.path.join(experiment_base_dir, exp_dir))
        
        latest_dir = find_latest_experiment_dir(experiment_base_dir)
        
        # Should return the continue directory with latest timestamp
        assert os.path.basename(latest_dir) == "mw_ppo_5x5x3_seed42_20250705160718_continue_20250705170000"
    
    def test_find_latest_experiment_dir_filters_non_experiments(self, temp_dir):
        """Test that non-experiment directories are filtered out."""
        experiment_base_dir = os.path.join(temp_dir, "experiments")
        os.makedirs(experiment_base_dir)
        
        # Create various directories
        os.makedirs(os.path.join(experiment_base_dir, "mw_ppo_5x5x3_seed42_20250705152544"))
        os.makedirs(os.path.join(experiment_base_dir, "README"))
        os.makedirs(os.path.join(experiment_base_dir, "config_files"))
        Path(os.path.join(experiment_base_dir, "notes.txt")).touch()
        
        latest_dir = find_latest_experiment_dir(experiment_base_dir)
        
        # Should only find the valid experiment directory
        assert os.path.basename(latest_dir) == "mw_ppo_5x5x3_seed42_20250705152544"


class TestResolveModelPathsFromRunDir:
    """Test resolve_model_paths_from_run_dir functionality."""
    
    def test_resolve_model_paths_from_training_dir(self, mock_training_run_dir):
        """Test resolving paths from a standard training run directory."""
        model_path, stats_path = resolve_model_paths_from_run_dir(mock_training_run_dir)
        
        # Should find final_model.zip
        assert model_path is not None
        assert os.path.basename(model_path) == "final_model.zip"
        assert os.path.exists(model_path)
        
        # Should find corresponding stats
        assert stats_path is not None
        assert "vecnormalize" in os.path.basename(stats_path)
    
    def test_resolve_model_paths_from_models_dir(self, mock_checkpoint_dir):
        """Test resolving paths when given models directory directly."""
        model_path, stats_path = resolve_model_paths_from_run_dir(mock_checkpoint_dir)
        
        # Should handle models directory directly
        assert model_path is not None
        assert os.path.basename(model_path) == "final_model.zip"
    
    def test_resolve_model_paths_with_specific_checkpoint(self, mock_training_run_dir):
        """Test resolving paths with specific checkpoint steps."""
        model_path, stats_path = resolve_model_paths_from_run_dir(mock_training_run_dir, checkpoint_steps=50000)
        
        # Should find the 50000 checkpoint
        assert model_path is not None
        assert "50000" in os.path.basename(model_path)
    
    def test_resolve_model_paths_no_models_dir(self, temp_dir):
        """Test behavior when no models directory exists."""
        empty_run_dir = os.path.join(temp_dir, "empty_run")
        os.makedirs(empty_run_dir)
        
        with pytest.raises(Exception, match="Could not resolve model paths"):
            resolve_model_paths_from_run_dir(empty_run_dir)
    
    def test_resolve_model_paths_no_checkpoints(self, temp_dir):
        """Test behavior when models directory exists but has no checkpoints."""
        run_dir = os.path.join(temp_dir, "run_dir")
        models_dir = os.path.join(run_dir, "models")
        os.makedirs(models_dir)
        
        with pytest.raises(Exception, match="Could not resolve model paths"):
            resolve_model_paths_from_run_dir(run_dir)
    
    def test_resolve_model_paths_output_messages(self, mock_training_run_dir, capsys):
        """Test that appropriate messages are printed."""
        model_path, stats_path = resolve_model_paths_from_run_dir(mock_training_run_dir)
        
        captured = capsys.readouterr()
        assert "Selected checkpoint:" in captured.out
        assert "Found VecNormalize stats:" in captured.out or "Warning: No VecNormalize stats" in captured.out


class TestResolveContinueTrainingPaths:
    """Test resolve_continue_training_paths functionality."""
    
    def test_resolve_continue_training_from_training_dir(self, mock_training_run_dir):
        """Test resolving continue training paths from a training run directory."""
        result = resolve_continue_training_paths(mock_training_run_dir)
        
        # Should detect it's a training directory and return correct paths
        assert result['original_run_dir'] == mock_training_run_dir
        assert result['checkpoint_dir'] == os.path.join(mock_training_run_dir, "models")
        assert result['checkpoint_path'] is not None
        assert os.path.basename(result['checkpoint_path']) == "final_model.zip"
        assert result['is_training_dir'] is True
    
    def test_resolve_continue_training_from_models_dir(self, mock_checkpoint_dir):
        """Test resolving continue training paths from a models directory directly."""
        result = resolve_continue_training_paths(mock_checkpoint_dir)
        
        # Should detect it's already a models directory
        assert result['original_run_dir'] is None
        assert result['checkpoint_dir'] == mock_checkpoint_dir
        assert result['checkpoint_path'] is not None
        assert os.path.basename(result['checkpoint_path']) == "final_model.zip"
        assert result['is_training_dir'] is False
    
    def test_resolve_continue_training_with_specific_steps(self, mock_training_run_dir):
        """Test resolving continue training paths with specific checkpoint steps."""
        result = resolve_continue_training_paths(mock_training_run_dir, continue_steps=50000)
        
        # Should find the 50000 step checkpoint
        assert result['checkpoint_path'] is not None
        assert "50000" in os.path.basename(result['checkpoint_path'])
        assert result['original_run_dir'] == mock_training_run_dir
    
    def test_resolve_continue_training_no_checkpoints(self, temp_dir):
        """Test behavior when directory has no checkpoints."""
        empty_run_dir = os.path.join(temp_dir, "empty_run")
        models_dir = os.path.join(empty_run_dir, "models")
        os.makedirs(models_dir)
        
        with pytest.raises(Exception, match="No checkpoint files found"):
            resolve_continue_training_paths(empty_run_dir)
    
    def test_resolve_continue_training_nonexistent_dir(self, temp_dir):
        """Test behavior with non-existent directory."""
        nonexistent_dir = os.path.join(temp_dir, "does_not_exist")
        
        with pytest.raises(Exception, match="Continue training directory does not exist"):
            resolve_continue_training_paths(nonexistent_dir)
    
    def test_resolve_continue_training_empty_models_dir(self, temp_dir):
        """Test behavior when models directory exists but is empty."""
        run_dir = os.path.join(temp_dir, "run_with_empty_models")
        models_dir = os.path.join(run_dir, "models")
        os.makedirs(models_dir)
        
        with pytest.raises(Exception, match="No checkpoint files found"):
            resolve_continue_training_paths(run_dir)
    
    def test_resolve_continue_training_output_messages(self, mock_training_run_dir, capsys):
        """Test that appropriate messages are printed."""
        result = resolve_continue_training_paths(mock_training_run_dir)
        
        captured = capsys.readouterr()
        assert "Found checkpoint to continue from:" in captured.out
        assert "Detected training run directory" in captured.out
    
    def test_resolve_continue_training_models_dir_output(self, mock_checkpoint_dir, capsys):
        """Test output messages when given models directory directly."""
        result = resolve_continue_training_paths(mock_checkpoint_dir)
        
        captured = capsys.readouterr()
        assert "Found checkpoint to continue from:" in captured.out
        assert "Using models directory directly" in captured.out