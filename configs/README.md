# Minesweeper RL Configuration Files

This directory contains example configuration files for training minesweeper RL agents with various board size configurations.

## Configuration Types

### 1. Traditional Fixed-Size Configuration
**File:** `backward_compatible_config.yaml`, `local_config.yaml`

Uses the traditional `environment_config` section with fixed board dimensions:
```yaml
environment_config:
  width: 8
  height: 8
  n_mines: 10
  reward_win: 1.0
  reward_lose: -0.1
  reward_reveal: 0.02
  reward_invalid: -0.05
```

### 2. Curriculum Learning Configuration
**File:** `curriculum_learning_config.yaml`, `exponential_curriculum_config.yaml`

Uses `dynamic_environment_config` with curriculum learning that progressively increases board size:
```yaml
dynamic_environment_config:
  curriculum:
    enabled: true
    progression_type: "linear"  # or "exponential"
    start_size:
      width: 5
      height: 5
      n_mines: 3
    end_size:
      width: 10
      height: 10
      n_mines: 15
    progression_steps: 20
    step_duration: 25000
    success_threshold: 0.6
    evaluation_episodes: 50
```

### 3. Multi-Size Sampling Configuration
**File:** `multisize_sampling_config.yaml`

Uses `dynamic_environment_config` with random sampling from multiple board sizes:
```yaml
dynamic_environment_config:
  board_sizes:
    - width: 5
      height: 5
      n_mines: 3
    - width: 8
      height: 8
      n_mines: 12
  random_sampling: true
  sampling_weights: [0.6, 0.4]  # Optional weights for sampling
```

### 4. Hybrid Configuration
**File:** `hybrid_dynamic_config.yaml`

Uses `dynamic_environment_config` with fallback to fixed configuration:
```yaml
dynamic_environment_config:
  fixed_config:
    width: 7
    height: 7
    n_mines: 8
  curriculum:
    enabled: false  # Can be enabled when needed
```

## Usage

### Basic Training
```bash
python train.py --config configs/curriculum_learning_config.yaml
```

### Override Configuration Parameters
```bash
python train.py --config configs/curriculum_learning_config.yaml --total_timesteps 1000000 --n_envs 16
```

### Continue Training
```bash
python train.py --config configs/curriculum_learning_config.yaml --continue_from ./training_runs/previous_run
```

## Configuration Parameters

### Curriculum Learning Parameters
- `progression_type`: `"linear"`, `"exponential"`, or `"manual"`
- `start_size`/`end_size`: Board dimensions at start/end of curriculum
- `progression_steps`: Number of curriculum steps
- `step_duration`: Timesteps per curriculum step
- `success_threshold`: Win rate threshold to advance (0.0-1.0)
- `evaluation_episodes`: Episodes to evaluate success rate

### Multi-Size Sampling Parameters
- `board_sizes`: List of board size configurations
- `random_sampling`: Enable random sampling
- `sampling_weights`: Optional probability weights for each board size

### Reward Configuration
All configurations support shared reward settings:
- `reward_win`: Reward for winning
- `reward_lose`: Penalty for losing
- `reward_reveal`: Reward per revealed safe cell
- `reward_invalid`: Penalty for invalid moves
- `max_reward_per_step`: Maximum reward per step

## Key Features

### Adaptive CNN Architecture
The CNN automatically adapts to different board sizes using adaptive pooling, allowing:
- Training on variable board sizes
- Seamless curriculum learning progression
- Random sampling from multiple sizes

### Backward Compatibility
All existing configuration files continue to work without modification. The system automatically detects traditional vs. dynamic configurations.

### Progress Tracking
Curriculum learning configurations provide detailed progress reporting:
- Current board size
- Win rate tracking
- Curriculum progression status
- Episode completion metrics

## Tips for Configuration

1. **Start Small**: Begin curriculum learning with small boards (4x4 or 5x5)
2. **Gradual Progression**: Use 10-20 curriculum steps for smooth learning
3. **Success Thresholds**: Set realistic win rate thresholds (0.5-0.7)
4. **Evaluation Episodes**: Use 50-100 episodes for stable performance evaluation
5. **Step Duration**: Allow 20k-50k timesteps per curriculum step

## Troubleshooting

- **Memory Issues**: Reduce `n_envs` for large board sizes
- **Slow Convergence**: Increase `step_duration` in curriculum learning
- **Unstable Learning**: Lower `success_threshold` for faster progression
- **GPU Memory**: Reduce `batch_size` for large networks