"""
Tests for transfer_compatible_weights() function.

Tests that CNN weights can be transferred between models of different board sizes
(e.g., 5x5 → 8x8) while correctly skipping layers with incompatible shapes.
"""

import os
import io
import zipfile
import tempfile
import shutil

import pytest
import torch as th
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from src.env.minesweeper_env import MinesweeperEnv
from src.env.custom_cnn import CustomCNN
from src.factories.model_factory import transfer_compatible_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINI_POLICY_KWARGS = {
    'features_extractor_class': CustomCNN,
    'features_extractor_kwargs': {'features_dim': 8},
    'net_arch': {'pi': [8], 'vf': [8]},
}


def make_mini_env(width, height, n_mines, obs_channels=2):
    """Create a minimal VecNormalize environment for fast tests."""
    env_fn = lambda: MinesweeperEnv(
        width=width, height=height, n_mines=n_mines,
        obs_channels=obs_channels, render_mode=None,
    )
    return VecNormalize(DummyVecEnv([env_fn]))


def make_mini_model(env):
    """Create a minimal MaskablePPO model on CPU for fast tests."""
    return MaskablePPO(
        "CnnPolicy", env,
        policy_kwargs=MINI_POLICY_KWARGS,
        device='cpu', seed=42,
        n_steps=64, batch_size=32,
    )


def save_model_to_zip(model, save_dir, name="source_model"):
    """Save model via SB3 and return the resulting .zip path."""
    save_path = os.path.join(save_dir, name)
    model.save(save_path)
    return save_path + ".zip"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def source_zip_and_target(tmp_path_factory):
    """
    Module-scoped fixture: creates source 5x5 model zip and target 8x8 model.
    Reused across tests to avoid redundant model construction overhead.
    """
    tmp_dir = str(tmp_path_factory.mktemp("transfer_test"))

    # Source: 5x5 board
    src_env = make_mini_env(width=5, height=5, n_mines=3)
    src_model = make_mini_model(src_env)
    zip_path = save_model_to_zip(src_model, tmp_dir)
    src_env.close()

    # Target: 8x8 board (different board size → Linear weight shape differs)
    tgt_env = make_mini_env(width=8, height=8, n_mines=10)
    tgt_model = make_mini_model(tgt_env)

    yield zip_path, tgt_model, src_model

    tgt_env.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransferCompatibleWeights:

    def test_conv_weights_transferred(self, source_zip_and_target):
        """Conv layer weights from source are copied exactly to target."""
        zip_path, tgt_model, src_model = source_zip_and_target

        # Record source conv weights before transfer
        src_state = src_model.policy.state_dict()

        stats = transfer_compatible_weights(zip_path, tgt_model)

        tgt_state = tgt_model.policy.state_dict()

        # Conv2d weight/bias shapes are board-independent; they must be transferred
        for key in ('features_extractor.cnn.0.weight',
                    'features_extractor.cnn.0.bias',
                    'features_extractor.cnn.2.weight',
                    'features_extractor.cnn.2.bias'):
            assert key in stats['transferred'], f"Expected {key} in transferred"
            assert th.allclose(tgt_state[key], src_state[key]), \
                f"Conv weight {key} not transferred correctly"

    def test_mismatched_layers_skipped(self, source_zip_and_target):
        """Linear weight with different flatten dim is skipped; action head is skipped."""
        zip_path, tgt_model, src_model = source_zip_and_target

        stats = transfer_compatible_weights(zip_path, tgt_model)

        # features_extractor.linear.0.weight: (8, 64*5*5) vs (8, 64*8*8) → mismatch
        assert 'features_extractor.linear.0.weight' in stats['skipped'], \
            "Linear weight (board-size-dependent) should be skipped"

        # action_net weight: (25, 8) vs (64, 8) → mismatch
        # Find any action_net.weight key that was skipped
        skipped_keys = stats['skipped']
        action_keys = [k for k in skipped_keys if 'action_net' in k and 'weight' in k]
        assert len(action_keys) > 0, "action_net.weight should be skipped (action dim differs)"

    def test_transfer_stats_returned(self, source_zip_and_target):
        """Return value is a dict with 'transferred' and 'skipped' lists."""
        zip_path, tgt_model, _ = source_zip_and_target

        stats = transfer_compatible_weights(zip_path, tgt_model)

        assert isinstance(stats, dict), "Should return a dict"
        assert 'transferred' in stats, "Dict should have 'transferred' key"
        assert 'skipped' in stats, "Dict should have 'skipped' key"
        assert isinstance(stats['transferred'], list)
        assert isinstance(stats['skipped'], list)
        assert len(stats['transferred']) > 0, "At least some layers should be transferred"
        assert len(stats['skipped']) > 0, "At least some layers should be skipped"

    def test_forward_pass_on_new_board(self, source_zip_and_target):
        """After transfer, 8x8 model should produce valid actions without error."""
        zip_path, tgt_model, _ = source_zip_and_target

        transfer_compatible_weights(zip_path, tgt_model)

        # Create a valid 8x8 observation (2-channel, shape (1, 2, 8, 8))
        obs = np.zeros((1, 2, 8, 8), dtype=np.float32)
        action_masks = np.ones((1, 64), dtype=bool)

        # Should not raise any exception
        action, _ = tgt_model.predict(obs, action_masks=action_masks, deterministic=True)
        assert action is not None
        assert 0 <= int(action[0]) < 64
