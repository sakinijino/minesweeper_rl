import modal
from pathlib import Path

app = modal.App("minesweeper-rl")

# Container image with all ML dependencies needed for training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libglib2.0-0",  # required by opencv
        "libgl1",        # required by opencv
    )
    .pip_install(
        "torch==2.7.0",
        "stable-baselines3==2.6.0",
        "sb3-contrib==2.6.0",
        "gymnasium==1.1.1",
        "pygame==2.6.1",
        "numpy==2.2.5",
        "PyYAML==6.0.1",
        "tensorboard==2.19.0",
    )
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("experiments/configs", remote_path="/app/experiments/configs")
    .add_local_file("train.py", remote_path="/app/train.py")
)

# Persistent volume: training results (checkpoints, logs) survive across runs
volume = modal.Volume.from_name("minesweeper-runs", create_if_missing=True)
RUNS_DIR = "/runs"


@app.function(
    gpu="T4",
    image=image,
    volumes={RUNS_DIR: volume},
    timeout=7200,  # 2 hours max
)
def train(config: str = "configs/colab_config.yaml"):
    import subprocess
    import sys
    import os

    os.chdir("/app")

    cmd = [
        sys.executable, "train.py",
        "--config", config,
        "--experiment_base_dir", RUNS_DIR,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Flush writes to volume so results are visible immediately
    volume.commit()
    print(f"Training complete. Results saved to volume '{RUNS_DIR}'.")


@app.local_entrypoint()
def main(config: str = "configs/colab_config.yaml"):
    print(f"Launching training on Modal (GPU: T4) with config: {config}")
    train.remote(config=config)
