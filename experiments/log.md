# 实验日志

## EXP-006 修复 vf_coef（待运行）

- **配置**：experiments/configs/exp_006_vf_coef_fix.yaml
- **假设**：EXP-004/005 的 explained_variance 长期卡在 0.44，是因为 vf_coef=0.5 在新 reward 量级（win=1.0）下 value gradient 不足；提升到 1.0 应能让 value function 充分拟合，进而带动 eval_win_rate 突破 84%
- **唯一变量**：vf_coef: 0.5 → 1.0
- **步数**：2M（从头训练，与 EXP-005 对称比较）
- **Run**：TBD
- **指标 (Eval)**：eval_win_rate = TBD
- **关键观测指标**：
  - explained_variance：是否回升至 0.7+（EXP-003 达到 0.84）
  - value_loss：是否显著下降（目标 <0.5，EXP-003 为 0.26）
  - eval_win_rate：是否超越 EXP-005 的 84%
- **后续规划**：
  - explained_variance >0.7 且 eval >88%：vf_coef 是关键，下一步叠加更大网络（features_dim 256）
  - explained_variance 回升但 eval 84-88%：value fitting 改善但策略上限在此，考虑 lr schedule
  - explained_variance 没有回升：问题不是 vf_coef，需要重新诊断

**启动命令**：
```bash
make train CONFIG=experiments/configs/exp_006_vf_coef_fix.yaml
```

---

## EXP-005 续训 EXP-004 到 2M 步 (2026-03-16)

- **配置**：experiments/configs/exp_005_continue_reward_2m.yaml
- **续训自**：mw_ppo_5x5x3_seed42_20260316134208（EXP-004，1M 步）
- **额外步数**：1,000,000（目标总步数约 2M）
- **Run**：mw_ppo_5x5x3_seed42_20260316134208_continue_20260316144019
- **假设**：EXP-004 @1M 已达 75%（超过 EXP-002 同期 52%），续训到 2M 步后期望超越 EXP-003 @2M 的 83%
- **唯一变量**：步数（EXP-004 基础上续训，所有超参不变）
- **指标 (TensorBoard)**：见 experiments/results/exp_005_metrics.json
- **指标 (Eval)**：eval_win_rate = **84%**（100 局，seed=42，@2M步）
- **关键指标**：
  - success_rate: 66% → 74%，最高 83%（本段新增 +8%）
  - value_loss: 1.55 → 0.97（下降 37%，value function 在追赶，但仍高于 EXP-003 的水平）
  - explained_variance: 0.44-0.46 区间震荡，**未能回升**（EXP-003 达到 0.84，差距巨大）
  - entropy_loss: -0.57 → -0.40（策略继续收敛，探索减少）
- **分析**：
  - eval 84% 略超 EXP-003（83%），但只差 1%，不能认为是显著突破
  - explained_variance 全程卡在 0.44 附近，续训 1M 步没有改善——这是结构性问题，不是步数问题
  - value_loss 从 1.55 下降到 0.97，说明 value function 在缓慢追赶，但最终仍未充分拟合
  - 结论符合预案中的 "≤85%" 场景：**explained_variance 是真正瓶颈**，需要修复 vf_coef
- **结论**：新 reward @2M = 84%，与旧 reward @2M（EXP-003 = 83%）几乎持平。续训没有带来明显突破。explained_variance 长期偏低（0.44 vs 0.84）是核心瓶颈，下一步必须修复 vf_coef（0.5 → 1.0）。
- **后续规划**：EXP-006 = 新 reward + vf_coef=1.0 + 从头训练 2M 步

---

## EXP-004 Reward Shaping 验证 (2026-03-16)

- **配置**：experiments/configs/exp_004_reward_shaping.yaml
- **Run**：TBD
- **假设**：EXP-002/003 中 reward_win=0.2 导致 win 信号仅占总奖励 8%，agent 更倾向"安全揭格"而非"追求胜利"
- **变量**：reward_win: 0.2 → 1.0，reward_lose: -0.05 → -1.0（win 信号占比 8% → 31%）
- **步数**：1M（够看趋势，节省 GPU），其余超参与 EXP-002/003 完全一致
- **指标 (TensorBoard)**：见 experiments/results/exp_004_metrics.json
- **指标 (Eval)**：eval_win_rate = **75%**（100 局，seed=42，@1M步）
- **关键指标**：
  - success_rate: 10% → 64%，最高 68%（vs EXP-002 同期 53%，+11%）
  - ep_rew_mean: 0.75 → 2.31（reward 绝对值变大，符合预期）
  - explained_variance: 最终 0.47（value function 拟合偏弱，相比 EXP-003 的 0.84 明显更差）
  - entropy_loss: -2.67 → -0.58（收敛速度与 EXP-003 相似）
- **分析**：
  - @1M 步 eval 75%，远超目标 65%，也接近 EXP-003 @2M 步的 83%
  - success_rate @1M 64% vs EXP-002 @800k 53%，同等步数下提升明显，说明 reward 确实是瓶颈
  - explained_variance 0.47 偏低（EXP-003 达 0.84），value function 拟合不好——reward 量级变大（win=1.0 vs 0.2）导致 value 估计更难，可能需要更大的 vf_coef 或更长训练
  - 单纯 1M 步已追上 EXP-003 2M 步 80% 的水平，效率提升显著
- **结论**：reward 是主要瓶颈，修复后 1M 步 eval 75%（vs EXP-002 @800k 52%）。下一步可叠加更大网络（features_dim 256）或继续训练到 2M 步看能否超越 EXP-003 83%。

---

## EXP-003 续训到 2M 步 (2026-03-16)

- **配置**：experiments/configs/exp_003_continue_2m.yaml
- **续训自**：mw_ppo_5x5x3_seed42_20260316121733（EXP-002，800k 步）
- **额外步数**：1,196,032（实际总步数 2,016,752）
- **Run**：mw_ppo_5x5x3_seed42_20260316121733_continue_20260316124704
- **指标 (TensorBoard)**：见 experiments/results/exp_003_metrics.json
- **指标 (Eval)**：eval_win_rate = **83%**（100 局，seed=42，final_model）
- **关键指标**：
  - success_rate: 47% → 68% 最高 77%（本段新增 +30%）
  - 累计成长：EXP-002 起点 10% → EXP-003 终点 77%
  - explained_variance: 0.84（稳定，value function 拟合良好）
  - entropy_loss: -0.92 → -0.51（策略进一步收敛，探索仍存在）
- **分析**：
  - 续训后 success_rate 从 47% 持续爬升到 77%，到 2M 步末尾曲线仍在 68-77% 间震荡，尚未完全 plateau
  - eval 胜率 83% 显著超过 TensorBoard success_rate（训练时 ~70%），说明 eval 环境更"幸运"或模型在确定性 seed 下表现更稳定
  - 相比 EXP-002，仅增加步数（唯一变量），胜率从 52% → 83%，涨幅 +31%，说明步数是当前的主要瓶颈
  - 2M 步后仍有轻微上升趋势，但增速明显放缓（从 +43% 降到 +30%），可能正在逼近当前超参下的收敛上限
- **结论**：步数从 800k → 2M 带来显著提升（52% → 83%）。当前超参下收敛上限约在 75-85%。建议下一步尝试超参调整（更大网络、learning rate 衰减）以突破瓶颈。

---

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
