"""
Tests for MinesweeperEnv observation encoding.

Covers:
- Single-channel backward compatibility (obs_channels=1)
- Two-channel observation (obs_channels=2)
- observation_space shape consistency
"""

import numpy as np
import pytest

from src.env.minesweeper_env import MinesweeperEnv


@pytest.fixture
def env_single():
    """Single-channel environment (default)."""
    env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=1)
    env.reset(seed=0)
    return env


@pytest.fixture
def env_double():
    """Two-channel environment."""
    env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=2)
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# Step 1: Single-channel backward compatibility
# ---------------------------------------------------------------------------

class TestSingleChannel:
    def test_single_channel_obs_shape(self, env_single):
        obs, _ = env_single.reset(seed=0)
        assert obs.shape == (1, 5, 5)

    def test_single_channel_obs_range(self, env_single):
        obs, _ = env_single.reset(seed=0)
        assert obs.dtype == np.float32
        assert float(obs.min()) >= 0.0
        assert float(obs.max()) <= 1.0

    def test_single_channel_unrevealed_value(self, env_single):
        """Initial state: all cells unrevealed, mapped from -2 → 0.0."""
        obs, _ = env_single.reset(seed=0)
        assert np.all(obs == 0.0)


# ---------------------------------------------------------------------------
# Step 2: Two-channel observation
# ---------------------------------------------------------------------------

class TestTwoChannel:
    def test_two_channel_obs_shape(self, env_double):
        obs, _ = env_double.reset(seed=0)
        assert obs.shape == (2, 5, 5)

    def test_two_channel_ch0_initial(self, env_double):
        """ch0 = is_unrevealed: initially all 1.0."""
        obs, _ = env_double.reset(seed=0)
        ch0 = obs[0]
        assert np.all(ch0 == 1.0)

    def test_two_channel_ch1_initial(self, env_double):
        """ch1 = revealed_number: initially all 0.0 (nothing revealed)."""
        obs, _ = env_double.reset(seed=0)
        ch1 = obs[1]
        assert np.all(ch1 == 0.0)

    def test_two_channel_ch0_after_reveal(self):
        """After revealing a cell, ch0 at that position should become 0.0."""
        env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=2)
        # Use seed that guarantees first action is safe (first-click protection handles it anyway)
        env.reset(seed=0)
        # Find the first action that is safe: first_step protection ensures no mine on first click
        action = 12  # center cell (row=2, col=2)
        obs, reward, terminated, truncated, info = env.step(action)
        row, col = np.unravel_index(action, (5, 5))
        # The clicked cell should now be revealed → ch0 = 0.0
        assert obs[0, row, col] == 0.0

    def test_two_channel_ch1_after_reveal(self):
        """After revealing a numbered cell, ch1 at that position = neighbor_count/8."""
        env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=2)
        env.reset(seed=0)
        # Step through actions until we find a revealed numbered cell
        # We'll take action 12 (center) and inspect revealed cells
        action = 12
        obs, _, terminated, _, _ = env.step(action)
        # For all revealed cells, verify ch1 = neighbor_counts / 8
        for r in range(5):
            for c in range(5):
                if env.revealed[r, c] and not env.mines[r, c]:
                    expected = env.neighbor_counts[r, c] / 8.0
                    np.testing.assert_almost_equal(obs[1, r, c], expected, decimal=5)

    def test_two_channel_obs_range(self, env_double):
        """Both channels must be in [0.0, 1.0]."""
        obs, _ = env_double.reset(seed=0)
        assert obs.dtype == np.float32
        assert float(obs.min()) >= 0.0
        assert float(obs.max()) <= 1.0

    def test_two_channel_obs_range_after_steps(self):
        """Values stay in [0, 1] after multiple steps."""
        env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=2)
        obs, _ = env.reset(seed=42)
        for action in range(25):
            if env.game_over:
                break
            if not env.revealed.flatten()[action]:
                obs, _, terminated, _, _ = env.step(action)
                assert float(obs.min()) >= 0.0, f"obs min below 0 after action {action}"
                assert float(obs.max()) <= 1.0, f"obs max above 1 after action {action}"
                if terminated:
                    break


# ---------------------------------------------------------------------------
# Step 3: observation_space shape matches actual obs
# ---------------------------------------------------------------------------

class TestObsSpaceShape:
    def test_obs_space_shape_single(self):
        env = MinesweeperEnv(width=5, height=4, n_mines=2, render_mode=None, obs_channels=1)
        assert env.observation_space.shape == (1, 4, 5)

    def test_obs_space_shape_double(self):
        env = MinesweeperEnv(width=5, height=4, n_mines=2, render_mode=None, obs_channels=2)
        assert env.observation_space.shape == (2, 4, 5)

    def test_obs_matches_obs_space_single(self):
        env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=1)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape

    def test_obs_matches_obs_space_double(self):
        env = MinesweeperEnv(width=5, height=5, n_mines=3, render_mode=None, obs_channels=2)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
