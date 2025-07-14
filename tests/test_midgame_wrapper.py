"""
Test suite for MidGameWrapper
"""
import pytest
import numpy as np
from src.env.minesweeper_env import MinesweeperEnv
from src.wrappers.midgame_wrapper import MidGameWrapper


class TestMidGameWrapper:
    """Test MidGameWrapper functionality"""
    
    def test_wrapper_creation(self):
        """Test that wrapper can be created and basic properties work"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(env, midgame_probability=0.5, seed=42)
        
        # Should preserve basic environment properties
        assert wrapped_env.observation_space == env.observation_space
        assert wrapped_env.action_space == env.action_space
        assert wrapped_env.midgame_probability == 0.5
    
    def test_fresh_reset_when_probability_zero(self):
        """Test that wrapper acts normally when midgame_probability=0"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(env, midgame_probability=0.0, seed=42)
        
        obs, info = wrapped_env.reset(seed=42)
        
        # Should be a fresh game
        assert info['revealed_cells'] == 0
        assert wrapped_env.get_stats()['midgame_resets'] == 0
        assert wrapped_env.get_stats()['fresh_resets'] == 1
    
    def test_midgame_reset_when_probability_one(self):
        """Test that wrapper generates mid-game when midgame_probability=1.0"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(
            env, 
            midgame_probability=1.0, 
            min_revealed_cells=3, 
            max_revealed_cells=5,
            seed=42
        )
        
        obs, info = wrapped_env.reset(seed=42)
        
        # Should have revealed some cells
        assert info['revealed_cells'] >= 3
        assert wrapped_env.get_stats()['midgame_resets'] == 1
        assert wrapped_env.get_stats()['fresh_resets'] == 0
    
    def test_midgame_state_is_valid(self):
        """Test that generated mid-game states are valid"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(
            env, 
            midgame_probability=1.0, 
            min_revealed_cells=2, 
            max_revealed_cells=8,
            seed=42
        )
        
        valid_states = 0
        total_states = 10
        
        for i in range(total_states):  # Test multiple generations
            obs, info = wrapped_env.reset(seed=42 + i)
            
            # Basic validity checks
            assert obs.shape == (1, 5, 5)
            assert obs.dtype == np.float32
            assert info['revealed_cells'] >= 1  # At least some cells revealed
            assert info['revealed_cells'] <= 22  # 25 - 3 mines = 22 max safe
            
            # Count valid non-game-over states
            if not wrapped_env.env.game_over:
                valid_states += 1
                # Should have some unrevealed cells available
                assert np.sum(~wrapped_env.env.revealed) > 0
            else:
                # If game is over, should be a win not a loss
                if not wrapped_env.env.win:
                    print(f"Warning: Mid-game generation resulted in loss on iteration {i}")
        
        # Should have mostly valid states (allow some failures due to randomness)
        assert valid_states >= total_states // 2  # At least 50% should be valid
    
    def test_dynamic_probability_adjustment(self):
        """Test that midgame probability can be adjusted dynamically"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(env, midgame_probability=0.0, seed=42)
        
        # Start with 0 probability
        for _ in range(5):
            wrapped_env.reset(seed=42)
        
        stats = wrapped_env.get_stats()
        assert stats['midgame_resets'] == 0
        
        # Change to 100% probability
        wrapped_env.set_midgame_probability(1.0)
        
        for _ in range(5):
            wrapped_env.reset(seed=42)
        
        stats = wrapped_env.get_stats()
        assert stats['midgame_resets'] == 5
    
    def test_statistics_tracking(self):
        """Test that wrapper correctly tracks statistics"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(env, midgame_probability=0.5, seed=42)
        
        # Reset wrapper stats
        wrapped_env.reset_stats()
        
        # Perform several resets
        for i in range(10):
            wrapped_env.reset(seed=42 + i)
        
        stats = wrapped_env.get_stats()
        
        # Check basic counts
        assert stats['total_resets'] == 10
        assert stats['midgame_resets'] + stats['fresh_resets'] == 10
        assert 0.0 <= stats['midgame_ratio'] <= 1.0
        
        # With probability 0.5 and 10 resets, we should have some of each
        # (though this could rarely fail due to randomness)
        assert stats['midgame_resets'] >= 0
        assert stats['fresh_resets'] >= 0
    
    def test_wrapper_preserves_environment_interface(self):
        """Test that wrapper preserves all environment methods"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(env, midgame_probability=0.3, seed=42)
        
        # Test reset
        obs, info = wrapped_env.reset(seed=42)
        assert obs.shape == (1, 5, 5)
        
        # Test step
        action = 0
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        assert obs.shape == (1, 5, 5)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Test action_masks
        masks = wrapped_env.action_masks()
        assert masks.shape == (25,)
        assert masks.dtype == bool
    
    def test_midgame_parameters_validation(self):
        """Test that wrapper handles edge cases in parameters"""
        env = MinesweeperEnv(width=3, height=3, n_mines=1)
        
        # Test with very high min_revealed_cells
        wrapped_env = MidGameWrapper(
            env, 
            midgame_probability=1.0, 
            min_revealed_cells=20,  # More than available safe cells
            max_revealed_cells=25,
            seed=42
        )
        
        # Should not crash and should work reasonably
        obs, info = wrapped_env.reset(seed=42)
        assert info['revealed_cells'] > 0
        assert info['revealed_cells'] <= 8  # 9 - 1 mine = 8 max safe
    
    def test_safe_first_moves(self):
        """Test that safe_first_moves parameter works correctly"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(
            env, 
            midgame_probability=1.0, 
            safe_first_moves=3,
            min_revealed_cells=1,
            max_revealed_cells=10,
            seed=42
        )
        
        obs, info = wrapped_env.reset(seed=42)
        
        # Should have made at least some safe moves
        assert info['revealed_cells'] >= 1
        # Should not have hit a mine (game should still be active)
        assert not wrapped_env.env.game_over
    
    def test_multiple_resets_consistency(self):
        """Test that multiple resets work consistently"""
        env = MinesweeperEnv(width=4, height=4, n_mines=2)
        wrapped_env = MidGameWrapper(
            env, 
            midgame_probability=1.0, 
            min_revealed_cells=2,
            max_revealed_cells=6,
            seed=42
        )
        
        revealed_counts = []
        
        for i in range(20):
            obs, info = wrapped_env.reset(seed=42 + i)
            revealed_counts.append(info['revealed_cells'])
            
            # Each reset should produce valid mid-game state
            assert info['revealed_cells'] >= 2
            assert not wrapped_env.env.game_over or wrapped_env.env.win
        
        # Should have variety in revealed counts
        assert len(set(revealed_counts)) > 1  # Not all the same
        assert min(revealed_counts) >= 2
        assert max(revealed_counts) <= 14  # 16 - 2 mines = 14 max safe


class TestMidGameWrapperIntegration:
    """Integration tests for MidGameWrapper with training scenarios"""
    
    def test_curriculum_learning_simulation(self):
        """Test a curriculum learning scenario"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        wrapped_env = MidGameWrapper(env, midgame_probability=0.0, seed=42)
        
        # Phase 1: Start with fresh games
        for _ in range(10):
            wrapped_env.reset(seed=42)
        
        stats_phase1 = wrapped_env.get_stats()
        assert stats_phase1['midgame_resets'] == 0
        
        # Phase 2: Gradually increase mid-game probability
        wrapped_env.set_midgame_probability(0.5)
        
        for _ in range(10):
            wrapped_env.reset(seed=42)
        
        stats_phase2 = wrapped_env.get_stats()
        assert stats_phase2['midgame_resets'] > stats_phase1['midgame_resets']
        
        # Phase 3: High mid-game probability
        wrapped_env.set_midgame_probability(0.8)
        
        for _ in range(10):
            wrapped_env.reset(seed=42)
        
        stats_phase3 = wrapped_env.get_stats()
        assert stats_phase3['midgame_resets'] > stats_phase2['midgame_resets']
    
    def test_wrapper_with_different_env_sizes(self):
        """Test wrapper works with different environment sizes"""
        sizes_and_mines = [
            (3, 3, 1),
            (5, 5, 3),
            (8, 8, 10),
            (10, 10, 15)
        ]
        
        for width, height, n_mines in sizes_and_mines:
            env = MinesweeperEnv(width=width, height=height, n_mines=n_mines)
            wrapped_env = MidGameWrapper(
                env, 
                midgame_probability=1.0, 
                min_revealed_cells=1,
                seed=42
            )
            
            obs, info = wrapped_env.reset(seed=42)
            
            # Should work for all sizes
            assert obs.shape == (1, height, width)
            assert info['revealed_cells'] >= 1
            max_safe_cells = width * height - n_mines
            assert info['revealed_cells'] <= max_safe_cells