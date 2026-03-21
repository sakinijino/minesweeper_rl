"""
Tests for compute_safe_mask() - constraint propagation safe-cell detection.

Strategy:
- Use explicit mine placement helpers to test specific board configurations.
- Tests are organized into: shape/dtype, pass-1 rules, pass-2 rules, obs integration.
"""

import numpy as np
import pytest

from src.env.minesweeper_env import MinesweeperEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(width, height, mine_positions, reveal_positions=()):
    """
    Create an env with manually placed mines and revealed cells (no cascade).
    reveal_positions: list of (r, c) cells to mark as revealed.
    """
    env = MinesweeperEnv(
        width=width, height=height, n_mines=len(mine_positions),
        render_mode=None, obs_channels=2,
    )
    env.reset(seed=0)
    env.mines[:] = False
    for (r, c) in mine_positions:
        env.mines[r, c] = True
    env._calculate_neighbors()
    for (r, c) in reveal_positions:
        env.revealed[r, c] = True
        env.board[r, c] = env.neighbor_counts[r, c]
    env.first_step = False
    return env


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestComputeSafeMaskShape:
    def test_returns_float32_array(self):
        env = make_env(5, 4, mine_positions=[(0, 0)])
        mask = env.compute_safe_mask()
        assert mask.dtype == np.float32

    def test_shape_matches_board(self):
        env = make_env(5, 4, mine_positions=[(0, 0)])
        mask = env.compute_safe_mask()
        assert mask.shape == (4, 5)

    def test_values_are_binary(self):
        env = make_env(5, 5, mine_positions=[(0, 0)])
        env.step(12)
        mask = env.compute_safe_mask()
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_initial_state_all_zeros(self):
        """No constraints visible → mask is all zeros."""
        env = make_env(5, 5, mine_positions=[(0, 0)])
        mask = env.compute_safe_mask()
        assert np.all(mask == 0.0)

    def test_revealed_cells_always_zero(self):
        """Revealed cells are not clickable → never marked safe."""
        env = make_env(5, 5, mine_positions=[(2, 2)])
        env.reset(seed=0)
        env.step(0)  # click top-left (first-click protection)
        mask = env.compute_safe_mask()
        assert np.all(mask[env.revealed] == 0.0)


# ---------------------------------------------------------------------------
# Pass-1: zero-count rule (all unrevealed neighbors are safe)
# ---------------------------------------------------------------------------

class TestPass1ZeroCount:
    def test_zero_count_cell_marks_all_unrevealed_neighbors_safe(self):
        """
        4×4 board, 1 mine at far corner (0,0).
        Reveal bottom-right corner (3,3): neighbor_count = 0.
        Its unrevealed neighbors (2,2),(2,3),(3,2) must all be safe.
        """
        env = make_env(4, 4, mine_positions=[(0, 0)], reveal_positions=[(3, 3)])
        assert env.neighbor_counts[3, 3] == 0
        mask = env.compute_safe_mask()
        assert mask[2, 2] == 1.0
        assert mask[2, 3] == 1.0
        assert mask[3, 2] == 1.0

    def test_zero_count_does_not_mark_revealed_cells(self):
        """Revealed neighbors of a zero-count cell stay 0 in mask."""
        # Reveal two adjacent cells: (3,3) and (3,2). Both have count=0.
        env = make_env(4, 4, mine_positions=[(0, 0)],
                       reveal_positions=[(3, 3), (3, 2)])
        mask = env.compute_safe_mask()
        assert mask[3, 3] == 0.0
        assert mask[3, 2] == 0.0

    def test_zero_count_does_not_affect_non_neighbor_cells(self):
        """
        Cells far from the zero-count revealed cell remain 0.
        """
        env = make_env(5, 5, mine_positions=[(0, 0)], reveal_positions=[(4, 4)])
        assert env.neighbor_counts[4, 4] == 0
        mask = env.compute_safe_mask()
        # (0, 0) is not adjacent to (4,4) on 5×5 → should be 0
        assert mask[0, 0] == 0.0
        assert mask[0, 1] == 0.0


# ---------------------------------------------------------------------------
# Pass-1: count-equals-unrevealed rule (all unrevealed neighbors are mines)
# ---------------------------------------------------------------------------

class TestPass1AllMines:
    def test_count_equals_unrevealed_count_does_not_mark_safe(self):
        """
        1×2 board, mine at (0,1). Reveal (0,0): count=1, 1 unrevealed neighbor (0,1).
        len(unrevealed)==count → (0,1) is certain mine → NOT in safe_mask.
        """
        env = make_env(2, 1, mine_positions=[(0, 1)], reveal_positions=[(0, 0)])
        assert env.neighbor_counts[0, 0] == 1
        mask = env.compute_safe_mask()
        assert mask[0, 1] == 0.0

    def test_certain_mine_cells_excluded_from_safe(self):
        """
        1×3 board, mines at (0,0) and (0,2). Reveal (0,1): count=2.
        Unrevealed = [(0,0),(0,2)], len=2=count → both certain mines → neither safe.
        """
        env = make_env(3, 1, mine_positions=[(0, 0), (0, 2)],
                       reveal_positions=[(0, 1)])
        assert env.neighbor_counts[0, 1] == 2
        mask = env.compute_safe_mask()
        assert mask[0, 0] == 0.0
        assert mask[0, 2] == 0.0


# ---------------------------------------------------------------------------
# Pass-2: certain-mine propagation
# ---------------------------------------------------------------------------

class TestPass2Propagation:
    def test_pass2_identifies_safe_cell_via_mine_inference(self):
        """
        1×5 board, mines at (0,0) and (0,2).
        Reveal (0,1): count=2, unrevealed=[(0,0),(0,2)], len=2=count → both certain mines (pass1).
        Reveal (0,3): count=1, unrevealed=[(0,2),(0,4)], len=2≠1 → no pass1 action.
          Pass2: certain_mine at (0,2) counts as 1 = count → (0,4) is safe!
        """
        env = make_env(5, 1, mine_positions=[(0, 0), (0, 2)],
                       reveal_positions=[(0, 1), (0, 3)])
        assert env.neighbor_counts[0, 1] == 2   # adjacent to (0,0) and (0,2)
        assert env.neighbor_counts[0, 3] == 1   # adjacent to (0,2) only
        mask = env.compute_safe_mask()
        # (0,4) is only identifiable as safe through pass2
        assert mask[0, 4] == 1.0
        # Certain mines are not safe
        assert mask[0, 0] == 0.0
        assert mask[0, 2] == 0.0

    def test_pass2_does_not_mark_safe_when_mines_not_accounted(self):
        """
        1×3 board, mine at (0,0). Reveal (0,1): count=1, unrevealed=[(0,0),(0,2)], len=2≠1.
        No certain mine from pass1. Pass2: certain_mine_here=0 ≠ count=1 → (0,2) NOT marked safe.
        """
        env = make_env(3, 1, mine_positions=[(0, 0)], reveal_positions=[(0, 1)])
        assert env.neighbor_counts[0, 1] == 1
        mask = env.compute_safe_mask()
        assert mask[0, 2] == 0.0


# ---------------------------------------------------------------------------
# obs_channels=3 integration
# ---------------------------------------------------------------------------

class TestObsChannels3:
    def test_obs_shape_three_channels(self):
        env = MinesweeperEnv(width=5, height=5, n_mines=3,
                             render_mode=None, obs_channels=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 5, 5)

    def test_obs_space_shape_three_channels(self):
        env = MinesweeperEnv(width=6, height=4, n_mines=3,
                             render_mode=None, obs_channels=3)
        assert env.observation_space.shape == (3, 4, 6)

    def test_ch2_all_zeros_at_start(self):
        """No constraints visible at start → ch2 = 0."""
        env = MinesweeperEnv(width=5, height=5, n_mines=3,
                             render_mode=None, obs_channels=3)
        obs, _ = env.reset(seed=0)
        assert np.all(obs[2] == 0.0)

    def test_ch0_ch1_unchanged_by_third_channel(self):
        """Ch0 and ch1 semantics must match obs_channels=2 output."""
        env2 = MinesweeperEnv(width=5, height=5, n_mines=3,
                              render_mode=None, obs_channels=2)
        env3 = MinesweeperEnv(width=5, height=5, n_mines=3,
                              render_mode=None, obs_channels=3)
        obs2, _ = env2.reset(seed=42)
        obs3, _ = env3.reset(seed=42)
        np.testing.assert_array_equal(obs2[0], obs3[0])  # ch0 same
        np.testing.assert_array_equal(obs2[1], obs3[1])  # ch1 same

    def test_ch2_range_in_zero_one(self):
        """Ch2 values must stay in [0.0, 1.0] after several steps."""
        env = MinesweeperEnv(width=5, height=5, n_mines=3,
                             render_mode=None, obs_channels=3)
        env.reset(seed=0)
        for action in range(25):
            if env.game_over:
                break
            if not env.revealed.flatten()[action]:
                obs, _, terminated, _, _ = env.step(action)
                assert float(obs[2].min()) >= 0.0
                assert float(obs[2].max()) <= 1.0
                if terminated:
                    break

    def test_obs_matches_observation_space(self):
        env = MinesweeperEnv(width=5, height=5, n_mines=3,
                             render_mode=None, obs_channels=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
