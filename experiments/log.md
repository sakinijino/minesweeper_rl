# 实验日志

## EXP-001 基线验证 (2026-03-16)

- **配置**：experiments/configs/exp_001_baseline_100k.yaml
- **Run**：mw_ppo_5x5x3_seed42_20260316104017
- **指标 (TensorBoard)**：见 experiments/results/exp_001_metrics.json
- **指标 (Eval)**：eval_win_rate = 18%（100 局，seed=42，frozen final_model）
- **分析**：
  - success_rate 在 7%-21% 之间剧烈波动，没有明显收敛趋势
  - explained_variance 最终 0.74，说明 value function 有一定拟合能力
  - entropy_loss 约 -1.69，探索程度正常
  - 100k 步训练量严重不足，无法判断模型真实能力上限
- **结论**：100k 步只够验证流程可行性，需要 ≥500k 步才能看到收敛趋势
- **下一步**：EXP-002 跑 1M 步（用 colab_config.yaml 参数），看收敛上限
