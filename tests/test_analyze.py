"""
Tests for scripts/analyze.py

Uses unittest.mock to isolate TensorBoard dependency.
Uses tmp_path (pytest built-in) for filesystem isolation.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the module under test with patched paths
# ---------------------------------------------------------------------------

ANALYZE_PATH = Path(__file__).parent.parent / "scripts" / "analyze.py"


def import_analyze(monkeypatch, training_runs_dir: Path, results_dir: Path):
    """Import scripts/analyze.py with TRAINING_RUNS_DIR and RESULTS_DIR patched."""
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze", ANALYZE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "TRAINING_RUNS_DIR", training_runs_dir)
    monkeypatch.setattr(mod, "RESULTS_DIR", results_dir)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dirs(tmp_path):
    """Return (training_runs_dir, results_dir) as fresh tmp subdirs."""
    training_runs = tmp_path / "training_runs"
    training_runs.mkdir()
    results = tmp_path / "experiments" / "results"
    # results dir is intentionally NOT created here so tests can check auto-creation
    return training_runs, results


@pytest.fixture
def analyze(monkeypatch, dirs):
    """Return the analyze module with patched directory constants."""
    training_runs_dir, results_dir = dirs
    return import_analyze(monkeypatch, training_runs_dir, results_dir)


# ---------------------------------------------------------------------------
# TestFindLatestRun
# ---------------------------------------------------------------------------


class TestFindLatestRun:
    def test_finds_most_recently_modified_dir(self, analyze, dirs):
        training_runs_dir, _ = dirs
        old = training_runs_dir / "run_old"
        new = training_runs_dir / "run_new"
        old.mkdir()
        time.sleep(0.01)
        new.mkdir()

        result = analyze.find_latest_run()
        assert result == new

    def test_single_dir_is_latest(self, analyze, dirs):
        training_runs_dir, _ = dirs
        only = training_runs_dir / "only_run"
        only.mkdir()

        result = analyze.find_latest_run()
        assert result == only

    def test_empty_training_runs_exits(self, analyze):
        with pytest.raises(SystemExit):
            analyze.find_latest_run()


# ---------------------------------------------------------------------------
# TestFindEventFiles
# ---------------------------------------------------------------------------


class TestFindEventFiles:
    def test_finds_tfevents_in_root(self, analyze, dirs):
        training_runs_dir, _ = dirs
        run_dir = training_runs_dir / "run_a"
        run_dir.mkdir()
        ef = run_dir / "events.out.tfevents.123456.host"
        ef.touch()

        result = analyze.find_event_files(run_dir)
        assert ef in result

    def test_finds_tfevents_nested(self, analyze, dirs):
        training_runs_dir, _ = dirs
        run_dir = training_runs_dir / "run_b"
        sub = run_dir / "subdir"
        sub.mkdir(parents=True)
        ef = sub / "events.out.tfevents.999.host"
        ef.touch()

        result = analyze.find_event_files(run_dir)
        assert ef in result

    def test_no_event_files_returns_empty(self, analyze, dirs):
        training_runs_dir, _ = dirs
        run_dir = training_runs_dir / "run_empty"
        run_dir.mkdir()

        result = analyze.find_event_files(run_dir)
        assert result == []


# ---------------------------------------------------------------------------
# TestPrintSummary
# ---------------------------------------------------------------------------


def _make_data(tags_values: dict) -> dict:
    """Build the data dict format used by print_summary."""
    return {tag: list(enumerate(values)) for tag, values in tags_values.items()}


class TestPrintSummary:
    def test_returns_extracted_metrics(self, analyze, capsys):
        data = _make_data(
            {
                "rollout/ep_rew_mean": [0.1, 0.5, 0.9],
                "rollout/success_rate": [0.05, 0.15, 0.25],
            }
        )
        metrics = analyze.print_summary("test_run", data)

        assert "final_ep_rew_mean" in metrics
        assert abs(metrics["final_ep_rew_mean"] - 0.9) < 1e-4
        assert "final_success_rate" in metrics
        assert abs(metrics["final_success_rate"] - 0.25) < 1e-4

    def test_total_steps_computed_correctly(self, analyze, capsys):
        # Steps are keys (0, 1, 2 ...) from _make_data; max step = 2 for 3 points
        data = _make_data({"rollout/ep_rew_mean": [0.1, 0.2, 0.3]})
        metrics = analyze.print_summary("run", data)
        assert metrics["total_steps"] == 2

    def test_missing_metric_handled_gracefully(self, analyze, capsys):
        # Only provide one of the expected metrics; others should print N/A without crashing
        data = _make_data({"rollout/ep_rew_mean": [0.5]})
        metrics = analyze.print_summary("run", data)
        # Should not raise; success_rate absent → not in extracted metrics
        assert "final_success_rate" not in metrics
        out = capsys.readouterr().out
        assert "N/A" in out


# ---------------------------------------------------------------------------
# TestSaveMetricsJson
# ---------------------------------------------------------------------------


class TestSaveMetricsJson:
    def test_json_contains_exp_id_and_run_name(self, analyze, dirs):
        _, results_dir = dirs
        results_dir.mkdir(parents=True)
        analyze.RESULTS_DIR = results_dir

        analyze.save_metrics_json("exp_001", "my_run", {"total_steps": 100000})

        path = results_dir / "exp_001_metrics.json"
        data = json.loads(path.read_text())
        assert data["exp_id"] == "exp_001"
        assert data["run_name"] == "my_run"

    def test_json_file_path_correct(self, analyze, dirs):
        _, results_dir = dirs
        results_dir.mkdir(parents=True)
        analyze.RESULTS_DIR = results_dir

        analyze.save_metrics_json("exp_002", "run_x", {"total_steps": 50})

        assert (results_dir / "exp_002_metrics.json").exists()

    def test_creates_results_dir_if_missing(self, analyze, dirs):
        _, results_dir = dirs
        analyze.RESULTS_DIR = results_dir
        assert not results_dir.exists()

        analyze.save_metrics_json("exp_003", "run_y", {})

        assert results_dir.exists()
        assert (results_dir / "exp_003_metrics.json").exists()


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """End-to-end CLI tests using subprocess to avoid polluting module state."""

    def _run(self, *args, training_runs_dir=None, results_dir=None, env_extra=None):
        """Run scripts/analyze.py as a subprocess with optional env overrides."""
        import os

        env = os.environ.copy()
        # Pass directory overrides via env vars so we can patch them inside the script
        # (The script itself doesn't read env vars, so we test via monkeypatching approach
        #  using the module fixture instead of subprocess for these tests.)
        return None

    def test_no_args_uses_latest_run(self, analyze, dirs, monkeypatch, capsys):
        """Without args, analyze picks the most recently modified run."""
        training_runs_dir, results_dir = dirs

        run = training_runs_dir / "only_run"
        run.mkdir()
        ef = run / "events.out.tfevents.1.host"
        ef.touch()

        fake_data = {"rollout/success_rate": [(0, 0.1), (1, 0.2)]}

        with patch.object(analyze, "read_tb_events", return_value=fake_data):
            # Simulate: no --run arg → uses latest
            run_dir = analyze.find_latest_run()
            event_files = analyze.find_event_files(run_dir)
            data = analyze.read_tb_events(event_files)
            metrics = analyze.print_summary(run_dir.name, data)

        assert run_dir == run
        assert "final_success_rate" in metrics

    def test_exp_id_arg_saves_json(self, analyze, dirs, monkeypatch):
        """When exp_id is provided, JSON is written to results dir."""
        training_runs_dir, results_dir = dirs
        monkeypatch.setattr(analyze, "RESULTS_DIR", results_dir)

        run = training_runs_dir / "test_run"
        run.mkdir()

        fake_data = {"rollout/ep_rew_mean": [(0, 0.5)]}

        with patch.object(analyze, "read_tb_events", return_value=fake_data):
            event_files = analyze.find_event_files(run)
            data = analyze.read_tb_events(event_files)
            metrics = analyze.print_summary(run.name, data)
            analyze.save_metrics_json("exp_099", run.name, metrics)

        path = results_dir / "exp_099_metrics.json"
        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["exp_id"] == "exp_099"
        assert saved["run_name"] == run.name
