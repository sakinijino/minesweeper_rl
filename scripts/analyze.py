#!/usr/bin/env python3
"""
Analyze TensorBoard logs from a training run.

Usage:
    python scripts/analyze.py [run_name] [--exp-id EXP_ID]

    run_name: name of the run directory under training_runs/ (default: latest)
    --exp-id: experiment ID for output JSON (e.g. exp_002). If omitted, no JSON is written.

Examples:
    python scripts/analyze.py
    python scripts/analyze.py mw_ppo_5x5x3_seed42_20260316104017
    python scripts/analyze.py mw_ppo_5x5x3_seed42_20260316104017 --exp-id exp_001
"""

import argparse
import json
import sys
from pathlib import Path

# Key metrics to display (TensorBoard tag -> display name)
METRICS = {
    "rollout/ep_rew_mean": "ep_rew_mean",
    "rollout/success_rate": "success_rate",
    "train/explained_variance": "explained_variance",
    "train/entropy_loss": "entropy_loss",
    "train/policy_gradient_loss": "policy_gradient_loss",
    "train/value_loss": "value_loss",
}

TRAINING_RUNS_DIR = Path(__file__).parent.parent / "training_runs"
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"


def find_latest_run() -> Path:
    runs = sorted(TRAINING_RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        print(f"ERROR: No runs found in {TRAINING_RUNS_DIR}", file=sys.stderr)
        sys.exit(1)
    return runs[0]


def find_event_files(run_dir: Path) -> list[Path]:
    return list(run_dir.rglob("events.out.tfevents.*"))


def read_tb_events(event_files: list[Path]) -> dict[str, list[tuple[int, float]]]:
    """Read TensorBoard events, return {tag: [(step, value), ...]}"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed. Run: pip install tensorboard", file=sys.stderr)
        sys.exit(1)

    data: dict[str, list[tuple[int, float]]] = {}
    for ef in event_files:
        ea = EventAccumulator(str(ef.parent))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if tag not in data:
                data[tag] = []
            data[tag].extend((e.step, e.value) for e in events)

    # Sort by step
    for tag in data:
        data[tag].sort(key=lambda x: x[0])

    return data


def print_summary(run_name: str, data: dict[str, list[tuple[int, float]]]) -> dict:
    """Print metrics summary table and return extracted key metrics."""
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"{'='*60}")

    # Determine total steps
    all_steps = [step for values in data.values() for step, _ in values]
    total_steps = max(all_steps) if all_steps else 0
    print(f"Total steps: {total_steps:,}")

    print(f"\n{'Metric':<25} {'First':>10} {'Last':>10} {'Min':>10} {'Max':>10} {'Points':>7}")
    print("-" * 75)

    extracted = {"total_steps": total_steps}

    for tag, display_name in METRICS.items():
        if tag in data:
            values = [v for _, v in data[tag]]
            steps = [s for s, _ in data[tag]]
            print(
                f"{display_name:<25} {values[0]:>10.4f} {values[-1]:>10.4f} "
                f"{min(values):>10.4f} {max(values):>10.4f} {len(values):>7}"
            )
            extracted[f"final_{display_name}"] = round(values[-1], 4)
            extracted[f"max_{display_name}"] = round(max(values), 4)
            extracted[f"min_{display_name}"] = round(min(values), 4)
        else:
            print(f"{display_name:<25} {'N/A':>10}")

    # Training quality assessment
    print(f"\n{'--- Quality Assessment ---'}")
    success_data = data.get("rollout/success_rate", [])
    if success_data:
        rates = [v for _, v in success_data]
        final_rate = rates[-1]
        max_rate = max(rates)
        improvement = max_rate - rates[0] if len(rates) > 1 else 0

        if max_rate < 0.1:
            print("  [!] success_rate never exceeded 10% — model barely learning")
        elif final_rate < 0.15 and max_rate > 0.2:
            print(f"  [!] success_rate volatile: peaked at {max_rate:.0%}, final {final_rate:.0%} — not converged")
        elif final_rate > 0.4:
            print(f"  [OK] Decent performance: final success_rate {final_rate:.0%}")
        else:
            print(f"  [~] Moderate: final success_rate {final_rate:.0%}, max {max_rate:.0%}")

        if improvement > 0.05:
            print(f"  [OK] Clear improvement: +{improvement:.0%} over training")
        else:
            print(f"  [!] Little improvement: only +{improvement:.0%}")

    ev_data = data.get("train/explained_variance", [])
    if ev_data:
        ev = ev_data[-1][1]
        if ev > 0.8:
            print(f"  [OK] explained_variance {ev:.2f} — value function well-fitted")
        elif ev > 0.5:
            print(f"  [~] explained_variance {ev:.2f} — value function partially fitted")
        else:
            print(f"  [!] explained_variance {ev:.2f} — value function poorly fitted")

    print(f"{'='*60}\n")
    return extracted


def save_metrics_json(exp_id: str, run_name: str, metrics: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {"exp_id": exp_id, "run_name": run_name, **metrics}
    path = RESULTS_DIR / f"{exp_id}_metrics.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"Metrics saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze TensorBoard training logs")
    parser.add_argument("run", nargs="?", help="Run directory name under training_runs/ (default: latest)")
    parser.add_argument("--exp-id", help="Experiment ID for output JSON (e.g. exp_002)")
    args = parser.parse_args()

    if args.run:
        run_dir = TRAINING_RUNS_DIR / args.run
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = find_latest_run()
        print(f"Using latest run: {run_dir.name}")

    event_files = find_event_files(run_dir)
    if not event_files:
        print(f"ERROR: No TensorBoard event files found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    data = read_tb_events(event_files)
    metrics = print_summary(run_dir.name, data)

    if args.exp_id:
        save_metrics_json(args.exp_id, run_dir.name, metrics)


if __name__ == "__main__":
    main()
