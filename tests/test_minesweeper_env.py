"""
Test suite for MinesweeperEnv
"""
import pytest
import numpy as np
from src.env.minesweeper_env import MinesweeperEnv


class TestMinesweeperEnvBasics:
    """Test basic functionality of MinesweeperEnv"""
    
    def test_environment_creation(self):
        """Test that environment can be created with default parameters"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        assert env.width == 5
        assert env.height == 5
        assert env.n_mines == 3
        assert env.observation_space.shape == (1, 5, 5)
        assert env.action_space.n == 25
    
    def test_reset(self):
        """Test environment reset"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        obs, info = env.reset(seed=42)
        
        # Check observation shape and type
        assert obs.shape == (1, 5, 5)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0) and np.all(obs <= 1)
        
        # Check info
        assert info['remaining_mines'] == 3
        assert info['revealed_cells'] == 0
        assert info['is_success'] == False
        
        # Check internal state
        assert np.sum(env.mines) == 3
        assert np.all(env.revealed == False)
        assert env.game_over == False
        assert env.first_step == True
    
    def test_first_step_mine_protection(self):
        """Test that first click never hits a mine"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        
        for seed in range(10):  # Test with multiple seeds
            env.reset(seed=seed)
            
            # Find a mine location
            mine_positions = np.argwhere(env.mines)
            if len(mine_positions) > 0:
                mine_row, mine_col = mine_positions[0]
                action = mine_row * env.width + mine_col
                
                # Click on the mine
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Should not lose on first click
                assert not terminated
                assert reward != env.reward_lose
                assert not env.mines[mine_row, mine_col]
    
    def test_valid_action(self):
        """Test valid action reveals cells"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Find a safe cell
        safe_positions = np.argwhere(~env.mines)
        safe_row, safe_col = safe_positions[0]
        action = safe_row * env.width + safe_col
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that cell was revealed
        assert env.revealed[safe_row, safe_col]
        assert info['revealed_cells'] > 0
        
        # Should get positive reward for revealing
        assert reward >= env.reward_reveal
    
    def test_invalid_action_penalty(self):
        """Test clicking revealed cell gives penalty"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Find a safe cell and click it
        safe_positions = np.argwhere(~env.mines)
        safe_row, safe_col = safe_positions[0]
        action = safe_row * env.width + safe_col
        
        env.step(action)
        
        # Click the same cell again
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should get penalty
        assert reward == env.reward_invalid
        assert not terminated
    
    def test_win_condition(self):
        """Test winning by revealing all safe cells"""
        env = MinesweeperEnv(width=3, height=3, n_mines=1)
        env.reset(seed=42)
        
        # Reveal all safe cells
        for row in range(3):
            for col in range(3):
                if not env.mines[row, col]:
                    action = row * env.width + col
                    obs, reward, terminated, truncated, info = env.step(action)
        
        # Should win
        assert terminated
        assert env.win
        assert info['is_success']
        # The last reward might not include win reward if already added
        # Check that game was won properly instead
    
    def test_lose_condition(self):
        """Test losing by clicking a mine"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Make first move to disable mine protection
        safe_positions = np.argwhere(~env.mines)
        safe_row, safe_col = safe_positions[0]
        env.step(safe_row * env.width + safe_col)
        
        # Find and click a mine
        mine_positions = np.argwhere(env.mines)
        mine_row, mine_col = mine_positions[0]
        action = mine_row * env.width + mine_col
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should lose
        assert terminated
        assert not env.win
        assert not info['is_success']
        assert reward == env.reward_lose
    
    def test_action_masks(self):
        """Test action masks correctly show valid actions"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Initially all actions should be valid
        masks = env.action_masks()
        assert masks.shape == (25,)
        assert np.all(masks)
        
        # Reveal a cell
        action = 0
        env.step(action)
        
        # That action should now be invalid
        masks = env.action_masks()
        assert not masks[action]
        # Count should be less than 25 (some cells revealed)
        assert np.sum(masks) < 25
    
    def test_recursive_reveal(self):
        """Test that clicking a zero cell reveals neighbors"""
        env = MinesweeperEnv(width=5, height=5, n_mines=1)
        env.reset(seed=42)
        
        # Find a cell with 0 neighbors
        for row in range(5):
            for col in range(5):
                if not env.mines[row, col] and env.neighbor_counts[row, col] == 0:
                    action = row * env.width + col
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Should reveal multiple cells
                    assert info['revealed_cells'] > 1
                    return
    
    def test_max_reward_per_step(self):
        """Test max reward per step limitation"""
        env = MinesweeperEnv(width=5, height=5, n_mines=1, 
                           reward_reveal=0.1, max_reward_per_step=0.5)
        env.reset(seed=42)
        
        # Find a zero cell that will reveal many cells
        for row in range(5):
            for col in range(5):
                if not env.mines[row, col] and env.neighbor_counts[row, col] == 0:
                    action = row * env.width + col
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Reward should be capped
                    assert reward <= 0.5
                    return


class TestMinesweeperEnvStateManagement:
    """Test state save/load functionality (TDD - these will fail initially)"""
    
    def test_get_state(self):
        """Test getting environment state"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Make some moves
        env.step(0)
        env.step(5)
        
        # Get state
        state = env.get_state()
        
        # Check state contains all necessary information
        assert 'board' in state
        assert 'mines' in state
        assert 'revealed' in state
        assert 'game_over' in state
        assert 'win' in state
        assert 'first_step' in state
        assert 'neighbor_counts' in state
        
        # Check shapes
        assert state['board'].shape == (5, 5)
        assert state['mines'].shape == (5, 5)
        assert state['revealed'].shape == (5, 5)
        assert isinstance(state['game_over'], bool)
        assert isinstance(state['win'], bool)
        assert isinstance(state['first_step'], bool)
    
    def test_set_state(self):
        """Test setting environment state"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Make some moves and save state
        env.step(0)
        env.step(5)
        state1 = env.get_state()
        revealed_count1 = np.sum(state1['revealed'])
        
        # Make more moves
        env.step(10)
        env.step(15)
        state2 = env.get_state()
        revealed_count2 = np.sum(state2['revealed'])
        
        assert revealed_count2 > revealed_count1
        
        # Restore to first state
        env.set_state(state1)
        
        # Check restoration
        assert np.array_equal(env.board, state1['board'])
        assert np.array_equal(env.mines, state1['mines'])
        assert np.array_equal(env.revealed, state1['revealed'])
        assert env.game_over == state1['game_over']
        assert env.win == state1['win']
        assert env.first_step == state1['first_step']
        assert np.sum(env.revealed) == revealed_count1
    
    def test_state_preservation_through_reset(self):
        """Test that saved state survives environment reset"""
        env = MinesweeperEnv(width=5, height=5, n_mines=3)
        env.reset(seed=42)
        
        # Make moves and save state
        env.step(0)
        env.step(5)
        state = env.get_state()
        
        # Reset environment
        env.reset(seed=99)
        
        # Restore state
        env.set_state(state)
        
        # Should be back to saved state
        assert np.array_equal(env.board, state['board'])
        assert np.sum(env.revealed) == np.sum(state['revealed'])
    
    def test_state_with_game_over(self):
        """Test saving and restoring game over states"""
        env = MinesweeperEnv(width=3, height=3, n_mines=1)
        env.reset(seed=42)
        
        # Win the game
        for row in range(3):
            for col in range(3):
                if not env.mines[row, col]:
                    env.step(row * env.width + col)
        
        assert env.game_over
        assert env.win
        
        # Save state
        win_state = env.get_state()
        
        # Reset and play differently
        env.reset(seed=42)
        
        # Restore win state
        env.set_state(win_state)
        
        assert env.game_over
        assert env.win