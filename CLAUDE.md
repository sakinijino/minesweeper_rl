# Minesweeper RL Agent — Project Context

## 项目简介

用 MaskablePPO（sb3-contrib）训练扫雷 AI。
- 自定义 Gymnasium 环境，内置动作掩码（屏蔽已揭开格子）
- 自定义 CNN 特征提取器，针对网格输入优化
- 支持任意棋盘尺寸和地雷数（默认实验棋盘：5×5，3 个地雷）
- 支持 checkpoint 续训、多模型对比、人机对战

## 关键文件

| 文件/目录 | 说明 |
|-----------|------|
| `train.py` | 训练入口，接受 `--config` / `--continue_from` 等参数 |
| `play.py` | 评估/对战入口，支持 agent/batch/human/compare 模式 |
| `train_modal.py` | Modal 包装层，GPU 云端训练 |
| `src/env/` | 环境和 CNN 实现 |
| `src/config/` | 配置系统（YAML/JSON，参数优先级：CLI > config > continue） |
| `src/factories/` | 环境和模型工厂 |
| `configs/` | 基础配置模板（local/colab，不随实验变化） |
| `experiments/configs/` | 每次实验的具体 yaml，命名 `exp_NNN_描述.yaml` |
| `experiments/log.md` | 人工撰写的实验分析和结论 |
| `experiments/results/` | `make analyze` 自动生成的指标摘要 JSON |
| `experiments/ideas.md` | 优化方向 backlog |
| `scripts/analyze.py` | 分析 TensorBoard logs，输出摘要 |
| `scripts/pull_run.sh` | 从 Modal Volume 拉取训练结果到本地 |
| `Makefile` | 统一命令入口 |

## 常用命令

### 本地训练
```bash
python train.py --config configs/local_config.yaml
python train.py --config configs/local_config.yaml --total_timesteps 50000 --learning_rate 0.0005
python train.py --continue_from training_runs/your_run/ --total_timesteps 3_000_000
```

### Modal 云端训练（标准工作流）
```bash
make train CONFIG=experiments/configs/exp_NNN_xxx.yaml           # 启动
make pull [RUN=run_name]                                         # 拉取结果
make analyze [RUN=run_name] [EXP_ID=exp_NNN]                    # 训练曲线诊断
make eval [RUN=run_name]                                         # 干净胜率评估（100局）
make play [RUN=run_name]                                         # 定性观察（可选）
make compare                                                     # 跨实验对比（可选）
make tensorboard                                                 # 看曲线
make list                                                        # 列出 Volume 上所有 run
```

### 评估和对战
```bash
make eval [RUN=run_name]                                         # 批量评估（100局干净胜率）
make play [RUN=run_name]                                         # AI 可视化演示
make compare                                                     # 多模型对比
python play.py --mode human --width 8 --height 8 --n_mines 12  # 人机对战
```

## Modal 配置

- **App 名称**：`minesweeper-rl`
- **Volume 名称**：`minesweeper-runs`
- **GPU**：T4
- **云端 runs 目录**：`/runs`
- **本地缓存目录**：`training_runs/`（.gitignore 排除）

## 实验约定

- 每次实验前：在 `experiments/configs/` 新建 `exp_NNN_描述.yaml`
- 每次实验后：在 `experiments/log.md` 记录 Run 名称 / 关键指标 / 分析结论
- `make analyze EXP_ID=exp_NNN` 自动输出 `experiments/results/exp_NNN_metrics.json`（git 跟踪）
- `training_runs/` 只是本地缓存，按需下载，不加入 git

## 实验编号规范

- 格式：`EXP-NNN`（三位数字，从 001 开始）
- config：`experiments/configs/exp_NNN_描述.yaml`
- 结果：`experiments/results/exp_NNN_metrics.json`
- log.md 中对应一节：`## EXP-NNN 描述 (YYYY-MM-DD)`
