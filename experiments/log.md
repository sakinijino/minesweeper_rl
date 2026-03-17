# 实验日志

## EXP-010 课程学习 5×5×3 → 8×8×10 (2026-03-17)

- **配置**：experiments/configs/exp_010_curriculum_5x5_to_8x8.yaml
- **假设**：Conv 权重（spatial pattern detector）在 5×5 和 8×8 棋盘间可直接迁移（shape 相同）；EXP-007 已学会的空间特征能加速 8×8×10 策略学习，让模型在 5M 步内突破 0% 胜率（EXP-009 全程 0%）
- **唯一变量**：vs EXP-009 ——增加 Conv 权重迁移（EXP-007 @1.75M steps），其余超参不变
- **对比基准**：EXP-009（8×8×10 from scratch 5M = 0%）；EXP-007（5×5×3 @1.75M = 87%，transfer 来源）
- **Source checkpoint**：mw_ppo_5x5x3_seed42_20260317041904，step 1750000
- **步数**：5M（从头计数，transfer 只迁移权重，不继承 timesteps）
- **Run**：TBD
- **指标 (TensorBoard)**：TBD → experiments/results/exp_010_metrics.json
- **指标 (Eval)**：TBD
- **关键观测**：
  - 前 1M 步 success_rate 是否 > 0%（vs EXP-009 全程 0%）
  - 最终 eval_win_rate 是否 > 0%
  - 学习起飞时间对比 EXP-009
- **分析**：TBD
- **结论**：TBD

---

## EXP-009 8×8×10 基准训练 (2026-03-17)

- **配置**：experiments/configs/exp_009_8x8x10_baseline.yaml
- **假设**：EXP-007 最优配置（双通道 + 全部超参）可直接迁移到更大棋盘；5M 步能建立有效基准胜率，验证方法可扩展性
- **唯一变量**：棋盘 5×5×3 → 8×8×10，步数 2M → 5M，checkpoint_freq 50k → 100k
- **对比基准**：EXP-007（5×5×3 @1.75M = 87%）
- **步数**：5M（从头训练）
- **Run**：mw_ppo_8x8x10_seed42_20260317092525
- **指标 (TensorBoard)**：见 experiments/results/exp_009_metrics.json
- **指标 (Eval)**：eval_win_rate = **0%**（100 局，seed=42，@5M步，0/100 wins）
- **关键指标**：
  - success_rate: **全程 0%**，max 仅 1%（8×8×10 完全未学会获胜）
  - explained_variance: -0.004 → 0.53，max 0.57（比 5×5×3 的 0.43 略好，value function 有在学习）
  - entropy_loss: -3.85 → -1.33（策略收敛，但收敛到失败策略）
  - value_loss: 0.63 → 0.46（持续下降，但对应的策略是 lose 策略）
  - fps: 稳定 ~1566，训练时长 ~53 分钟（3201 秒）
- **分析**：
  - 从头训练 5M 步无法让 8×8×10 模型获得任何胜利，与预期"eval < 30%"分支吻合
  - 策略确实在学习（entropy 降低、ep_len_mean 从 5 →6），但模型陷入"尽量多揭格但总踩雷"的局部最优
  - 8×8 棋盘有 54 个非地雷格 vs 5×5 的 22 个，随机探索完成一局的概率极低（约 (54/64)^10 ≈ 17%），导致 reward signal 稀疏，策略无法从偶然胜利中学习
  - explained_variance 0.53 说明 value function 在学习"什么格子更容易踩雷"，但策略层面无法转化为胜利
- **结论**：from scratch 5M 步对 8×8×10 不够。需要课程学习（EXP-010：从 EXP-007 5×5×3 最优权重迁移到 8×8×10）；或先验证更多步数（10M+）是否能突破。优先推荐课程学习，因为 5×5 已有 87% 的有效策略可以迁移。
- **后续规划（触发 <30% 分支）**：
  - **EXP-010**：课程学习 —— 加载 EXP-007 最优 checkpoint（1.75M 步），在 8×8×10 上续训；需要解决网络输入维度不兼容问题（5×5→8×8 输入 shape 变化）
  - 或先试 10M 步续训（简单，但可能仍无效）

---

## EXP-008 扩大网络容量 (2026-03-17)

- **配置**：experiments/configs/exp_008_larger_network.yaml
- **假设**：EXP-007 双通道 explained_variance 仍卡在 0.43，与 EXP-003（旧 reward）的 0.84 差距巨大；新 reward win=1.0（旧 0.2）量级增加 5×，features_dim=128 的 Linear 层输出瓶颈导致 value function 无法拟合更大方差；扩容至 256 应能让 explained_variance 回升至 0.6+
- **唯一变量**：features_dim: 128 → 256（不能续训 EXP-007，权重维度不兼容）
- **对比基准**：EXP-007（双通道 + features_dim=128 @2M = 86%，explained_var=0.43）
- **步数**：2M（从头训练）
- **Run**：mw_ppo_5x5x3_seed42_20260317075354
- **指标 (TensorBoard)**：见 experiments/results/exp_008_metrics.json
- **指标 (Eval)**：eval_win_rate = **86%**（100 局，seed=42，@2M步）——与 EXP-007 完全相同
- **关键指标**：
  - success_rate: 9% → 87%，最高 89%（略高于 EXP-007 的 88%）
  - value_loss: 0.948 → 0.808（比 EXP-007 的 0.96→0.78 下降幅度相近）
  - explained_variance: max 0.468，final 0.425（**仅微升，远未达到 0.6 目标**）
  - entropy_loss: -2.67 → -0.47（收敛节奏正常）
  - fps: 早期 ~1700，末期 ~1234（比 EXP-007 下降约 22%）
  - 训练时长：约 27 分钟
- **分析**：
  - eval 86% 与 EXP-007 完全持平——features_dim 加倍对最终胜率无任何改善
  - explained_variance max 仅从 0.43 升到 0.47，完全未达到 0.6 目标，**网络容量假设被否定**
  - value_loss 末期 0.81（vs EXP-007 的 0.78），下降幅度相近，说明更大网络只是在同等困难下做了类似的工作
  - 对比 EXP-003（旧 reward，features_dim=128，explained_var=0.84）：相同网络容量在旧 reward 下能完美拟合；新 reward 下无论加多少容量，explained_var 都卡在 0.43-0.47
  - **核心结论**：explained_variance 瓶颈不在网络容量，而在于 value function 本身面对的信息量/任务难度。可能的原因：(1) 新 reward（win=1.0）下单局结果方差极高（随机地雷布局），value网络无法准确预期回报；(2) pi/vf 共享 CNN 特征，策略更新持续破坏 value 拟合
- **结论**：features_dim 128→256 无效。网络容量不是 explained_variance 瓶颈的根本原因。需另辟蹊径：考虑延长训练步数（4M+）、lr schedule、分开 pi/vf 网络，或接受当前 ~0.45 的 explained_var 上限，转而专注提升 eval 胜率（如更大 n_steps、multi-seed 验证）。
- **后续规划**：
  - 接受 explained_variance ~0.45 是该 reward 设置下的结构性上限，转向提升 eval 胜率策略
  - 考虑延长训练到 4M 步（续训 EXP-008），观察 eval 是否能突破 86%
  - 或探索 pi/vf 层扩容（[128,128]）——与 features_dim 独立，可验证是否改善 value 拟合

---

## EXP-007 多通道观测 (2026-03-17)

- **配置**：experiments/configs/exp_007_multichannel_obs.yaml
- **假设**：单通道将"未揭开"(-2→0.0) 和"已揭数字"(0-8→0.2-1.0) 线性混合，value function 无法区分两类语义不同的状态；双通道分离编码（ch0=未揭开掩码，ch1=已揭数字）应能让 value function 更好拟合，explained_variance 从 0.44 回升至 0.6+
- **唯一变量**：obs_channels: 1 → 2（ch0=is_unrevealed, ch1=neighbor_counts/8.0）
- **对比基准**：EXP-006（单通道 + vf_coef=1.0 @2M = 80%，explained_var=0.46）
- **步数**：2M（从头训练）
- **Run**：mw_ppo_5x5x3_seed42_20260317041904
- **指标 (TensorBoard)**：见 experiments/results/exp_007_metrics.json
- **指标 (Eval)**：eval_win_rate = **86%**（100 局，seed=42，@2M步）
- **关键指标**：
  - success_rate: 10% → 83%，最高 88%（历史最高）
  - value_loss: 0.96 → 0.78（比 EXP-006 的 0.91 下降更多）
  - explained_variance: max 0.46，final 0.43（**与 EXP-005/006 几乎相同，未突破**）
  - entropy_loss: -2.66 → -0.42（收敛节奏正常）
- **分析**：
  - eval 86% 是历史最高，超越 EXP-005（84%）和 EXP-006（80%）——双通道观测带来实际胜率提升
  - 但 explained_variance 依然卡在 0.43，假设"双通道能突破 value function 拟合瓶颈"**未被验证**
  - 双通道早期收敛更快（50k 步 explained_variance 已达 0.35+，EXP-006 早期几乎为负），说明信息表示更清晰
  - 然而最终天花板没有突破——对比 EXP-003（旧 reward）explained_variance=0.84，差距仍巨大
  - value_loss 下降更多（0.78 vs 0.91）但 explained_variance 没有跟上，说明网络在拟合均值，但方差解释能力受限于**网络容量**
  - **核心结论**：explained_variance 瓶颈不在观测编码，而在网络容量（features_dim=128 不足）
- **结论**：双通道观测值得保留（胜率 +2~6%，早期收敛更快），但不是 explained_variance 瓶颈的根本原因。真正的瓶颈是网络容量不足。
- **后续规划**：EXP-008 = 双通道 + features_dim 128 → 256，验证扩大网络容量能否让 explained_variance 回升到 0.6+

---

## EXP-006 修复 vf_coef（2026-03-17）

- **配置**：experiments/configs/exp_006_vf_coef_fix.yaml
- **假设**：EXP-004/005 的 explained_variance 长期卡在 0.44，是因为 vf_coef=0.5 在新 reward 量级（win=1.0）下 value gradient 不足；提升到 1.0 应能让 value function 充分拟合，进而带动 eval_win_rate 突破 84%
- **唯一变量**：vf_coef: 0.5 → 1.0
- **步数**：2M（从头训练，与 EXP-005 对称比较）
- **Run**：mw_ppo_5x5x3_seed42_20260317031520
- **指标 (TensorBoard)**：见 experiments/results/exp_006_metrics.json
- **指标 (Eval)**：eval_win_rate = **80%**（100 局，seed=42，@2M步）
- **关键指标**：
  - success_rate: 10% → 75%，最高 79%（与 EXP-004/005 相近）
  - value_loss: 0.97 → 0.91（仅微降，远未到 EXP-003 的 0.26）
  - explained_variance: max 0.52，final 0.46（稍高于 EXP-005 的 0.44，但**远未回升到 0.7+**）
  - entropy_loss: -2.67 → -0.36（收敛节奏与 EXP-004/005 相同）
- **分析**：
  - eval 80% 比 EXP-005（84%）还低——vf_coef 加倍不仅没有帮助，反而可能挤压了策略梯度
  - explained_variance max 仅 0.52（EXP-005 max 0.46），提升极其微弱，说明 vf_coef **不是** explained_variance 偏低的根本原因
  - value_loss 基本没变（0.91 vs 0.97），说明 value function 的拟合能力受限于特征质量，而非优化力度
  - **核心结论**：瓶颈在于观测表示/网络容量，而非训练超参。当前单通道 [-1,0-8] 编码对 value function 信息量不足
- **结论**：vf_coef 修复无效，排除超参层面的解释。瓶颈是架构层面——需要更丰富的观测编码（多通道）或更大网络（features_dim 256）
- **后续规划**：EXP-007 = 多通道观测（分离未揭开/数字/地雷计数信道），保持其余超参不变

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
