# 实验日志

## EXP-002 1M 步规模验证 (2026-03-16)

- **配置**：experiments/configs/exp_002_1m_scale.yaml
- **Run**：mw_ppo_5x5x3_seed42_20260316121733
- **实际步数**：802,816（因本地断连提前终止，目标 1M）
- **指标 (TensorBoard)**：见 experiments/results/exp_002_metrics.json
- **指标 (Eval)**：eval_win_rate = 52%（100 局，seed=42，final_model）
- **关键指标**：
  - success_rate: 10% → 53%（+43%，明显收敛）
  - explained_variance: 最终 0.84（value function 拟合良好）
  - entropy_loss: -2.67 → -0.90（策略逐渐收敛，探索减少）
- **分析**：
  - 800k 步内胜率从 10% 稳定收敛到 53%，证明 agent 确实能学会扫雷
  - 训练曲线有明显上升趋势，还未完全 plateau，继续训练应有提升空间
  - eval 胜率 52% 与 TensorBoard success_rate 53% 吻合，结果可信
  - 本次因本地 Modal 客户端断连中断（应使用 `modal run --detach`）
- **结论**：agent 能学习，800k 步达到 52% 胜率。训练未完全收敛，下一步考虑：
  1. 续训到 1M+ 步看是否继续提升
  2. 调整超参（learning_rate 衰减、更大网络）
  3. 多通道观测改善特征表示

---

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
