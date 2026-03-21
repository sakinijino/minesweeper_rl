"""
Tests for run_training() in train.py.

Key behavior under test: when continue_training=True, reset_num_timesteps=True
so that total_timesteps in config means ADDITIONAL steps (intuitive behavior).
Both from-scratch and continue paths now pass reset_num_timesteps=True.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from tests.test_config_helper import create_test_config_manager


def make_mock_model():
    model = MagicMock()
    model.learn = MagicMock()
    return model


def make_mock_callback():
    return MagicMock()


class TestRunTrainingResetTimesteps:
    """Verify reset_num_timesteps behavior for from-scratch vs continue training."""

    def test_from_scratch_uses_reset_true(self):
        """From-scratch training passes reset_num_timesteps=True."""
        from train import run_training_loop as run_training

        config_manager = create_test_config_manager()
        model = make_mock_model()

        with patch("builtins.print"):
            run_training(
                model=model,
                config_manager=config_manager,
                checkpoint_callback=make_mock_callback(),
                continue_training=False,
                final_model_path="/tmp/test_final.zip",
                stats_path="/tmp/test_stats.pkl",
                train_env=MagicMock(),
            )

        model.learn.assert_called_once()
        _, kwargs = model.learn.call_args
        assert kwargs["reset_num_timesteps"] is True

    def test_continue_training_also_uses_reset_true(self):
        """Continue training also passes reset_num_timesteps=True (extra steps semantics)."""
        from train import run_training_loop as run_training

        config_manager = create_test_config_manager()
        model = make_mock_model()

        with patch("builtins.print"):
            run_training(
                model=model,
                config_manager=config_manager,
                checkpoint_callback=make_mock_callback(),
                continue_training=True,
                final_model_path="/tmp/test_final.zip",
                stats_path="/tmp/test_stats.pkl",
                train_env=MagicMock(),
            )

        model.learn.assert_called_once()
        _, kwargs = model.learn.call_args
        assert kwargs["reset_num_timesteps"] is True

    def test_total_timesteps_passed_from_config(self):
        """total_timesteps from config is passed to model.learn."""
        from train import run_training_loop as run_training

        config_manager = create_test_config_manager()
        expected_steps = config_manager.config.training_execution.total_timesteps
        model = make_mock_model()

        with patch("builtins.print"):
            run_training(
                model=model,
                config_manager=config_manager,
                checkpoint_callback=make_mock_callback(),
                continue_training=False,
                final_model_path="/tmp/test_final.zip",
                stats_path="/tmp/test_stats.pkl",
                train_env=MagicMock(),
            )

        _, kwargs = model.learn.call_args
        assert kwargs["total_timesteps"] == expected_steps

    def test_continue_training_total_timesteps_is_additional(self):
        """
        Regression test: continue training must NOT pass reset_num_timesteps=False.

        Old (broken) behavior: reset_num_timesteps=False made total_timesteps a
        cumulative cap, so a model already at 5M steps with total_timesteps=5M
        would exit immediately. The fix: always reset_num_timesteps=True so
        total_timesteps is always treated as additional steps.
        """
        from train import run_training_loop as run_training

        config_manager = create_test_config_manager()
        model = make_mock_model()

        with patch("builtins.print"):
            run_training(
                model=model,
                config_manager=config_manager,
                checkpoint_callback=make_mock_callback(),
                continue_training=True,
                final_model_path="/tmp/test_final.zip",
                stats_path="/tmp/test_stats.pkl",
                train_env=MagicMock(),
            )

        _, kwargs = model.learn.call_args
        # Must NOT be False — that's the old broken behavior
        assert kwargs["reset_num_timesteps"] is not False

    def test_keyboard_interrupt_handled_gracefully(self):
        """KeyboardInterrupt during training is caught without propagating."""
        from train import run_training_loop as run_training

        config_manager = create_test_config_manager()
        model = make_mock_model()
        model.learn.side_effect = KeyboardInterrupt()

        with patch("builtins.print"):
            # Should not raise
            run_training(
                model=model,
                config_manager=config_manager,
                checkpoint_callback=make_mock_callback(),
                continue_training=False,
                final_model_path="/tmp/test_final.zip",
                stats_path="/tmp/test_stats.pkl",
                train_env=MagicMock(),
            )

    def test_final_model_saved_even_on_interrupt(self):
        """Final model is saved in finally block even when training is interrupted."""
        from train import run_training_loop as run_training

        config_manager = create_test_config_manager()
        model = make_mock_model()
        model.learn.side_effect = KeyboardInterrupt()

        with patch("builtins.print"):
            run_training(
                model=model,
                config_manager=config_manager,
                checkpoint_callback=make_mock_callback(),
                continue_training=True,
                final_model_path="/tmp/test_final.zip",
                stats_path="/tmp/test_stats.pkl",
                train_env=MagicMock(),
            )

        model.save.assert_called_once_with("/tmp/test_final.zip")
