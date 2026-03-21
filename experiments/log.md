# 实验日志

## EXP-019c A6 课程学习 Stage4 —— 8×8×10（EXP-018 reward 配置）(2026-03-20)

配置 `experiments/configs/exp_019c_a6_curriculum_8x8x10.yaml`，Run `mw_ppo_8x8x10_seed42_20260320110407`，8M 步（新棋盘从头计数），迁移自 EXP-019b 最优 checkpoint（`mw_ppo_7x7x7_seed42_20260320100042` @ step 4,500,000，eval 27%），`TRANSFER_FROM` + `TRANSFER_STEPS=4500000`，Transferred 25 layers，Skipped 5 layers。

**核心变量**（vs EXP-015c）：reward_win 1.0→0.1，reward_lose -1.0→-0.1；source 改为 EXP-019b 最优 checkpoint。

**指标**：
- TensorBoard：success_rate final 2% / max 3%，EV 0.54，entropy_loss -0.93，value_loss 0.41
- Eval：未做（success_rate 过低，无意义）

**分析**：8M 步 success_rate 全程 0-3%，与 EXP-015c（±1.0，0%）几乎一样。小幅 reward 对 8×8×10 无显著改善。EV 0.54 偏低，value function 未充分拟合，说明 8×8×10 对当前网络容量和训练量均不足。

**结论**：❌ 未达成目标（success_rate max 3%，接近随机）。8×8×10 仍是硬瓶颈，reward 幅度不是关键因素。A6 课程路径的瓶颈在从 7×7→8×8 的跳跃，需要更大网络或更多步数。

---

## EXP-019b A6 课程学习 Stage3 —— 7×7×7（EXP-018 reward 配置）(2026-03-20)

配置 `experiments/configs/exp_019b_a6_curriculum_7x7x7.yaml`，Run `mw_ppo_7x7x7_seed42_20260320100042`，迁移自 EXP-019a 最优 checkpoint（`mw_ppo_6x6x5_seed42_20260320072443_continue_20260320084707` @ step 8,263,504，eval 77%），`TRANSFER_FROM` + `TRANSFER_STEPS=8263504`，Transferred 25 layers，Skipped 5 layers。

**核心变量**（vs EXP-015b）：reward_win 1.0→0.1，reward_lose -1.0→-0.1；source 改为 EXP-019a 最优 checkpoint。

### Phase 1：5M 步

**指标**：
- TensorBoard：success_rate final 29% / max 29%，EV 0.62，entropy_loss -0.78，value_loss 0.37
- Eval：**27%**（step 4,500,000，500 局）；final model 31.67%（300 局，方差较大）
- 对比 EXP-015b（±1.0）：eval 16% → 本实验 **+11%**

**分析**：小幅 reward 在 7×7×7 上确有改善（16%→27%），EV 0.62 正常拟合。success_rate 整个训练过程持续上升（0→29%），未见明显 plateau，说明 5M 步可能不够充分。eval 结果方差较大（同一 checkpoint 200 局 35%、300 局 38%、500 局 27%），7×7×7 随机性高、样本量影响大。虽未达 40% 门槛，但相比 EXP-015b 有明显改善且曲线仍上升，取 step 4,500,000 作为 EXP-019c source 推进观察。

**结论**：⚠️ 未完全达成目标（27% < 40%），但相对 EXP-015b（16%）改善 +11%，正迁移效果确认，继续推进 019c。

### Phase 2：续训至 13M（019b-continue）

配置 `experiments/configs/exp_019b_continue_7x7x7.yaml`，Run `mw_ppo_7x7x7_seed42_20260320100042_continue_20260320110730`，从 5M 续训至 13,074,432 步。

**指标**：
- TensorBoard：success_rate final 45% / max 55%，EV 0.62，entropy_loss -0.53（持续下降，仍在收敛）
- Eval：final model **51.6%**（500 局）；各 ckpt 300 局约 51-55%，方差较大
- 对比 EXP-015b（±1.0）：eval 16% → 本阶段 **51.6%，+35.6%**

**分析**：5M 步时 27%，13M 步达到 51.6%，说明 7×7×7 需要大量步数才能收敛。entropy 持续下降（-0.78→-0.53）说明 policy 仍在细化，未完全 plateau。小幅 reward 在充足步数下效果显著。

**结论**：✅ 达成目标（51.6% > 40%）。7×7×7 真实上限约 50-55%，约需 10-13M 步才能充分收敛。final model 作为候选 source（若未来需要更大棋盘再次迁移）。

---

## EXP-019a A6 课程学习 Stage2 —— 6×6×5（EXP-018 reward 配置）(2026-03-20)

配置 `experiments/configs/exp_019a_a6_curriculum_6x6x5.yaml`，迁移自 EXP-018（`mw_ppo_5x5x3_seed42_20260320034128`，EV=0.84，eval 85.6%），`TRANSFER_FROM` CLI 参数传入（YAML 的 `continue_from` 字段仅作文档用途，不被 config 系统读取），Transferred 25 layers，Skipped 5 layers（shape mismatch，棋盘尺寸变化正常）。

**核心变量**（vs EXP-015a）：reward_win 1.0→0.1，reward_lose -1.0→-0.1；source 改为 EXP-018。

### Phase 1：5M 步

Run `mw_ppo_6x6x5_seed42_20260320072443`，5M 步。

**指标**：
- Eval：**68%**（300 局，5M final model，未达 70% 门槛）

**分析**：5M 步时 eval 68%，曲线仍有上升趋势，未达 70% 门槛，决定继续续训。

**结论**：⚠️ 未达成阶段目标（68% < 70%），续训观察。

### Phase 2：续训至 8.6M

Run `mw_ppo_6x6x5_seed42_20260320072443_continue_20260320084707`，续训至 8,263,504 步。

**指标**：
- TensorBoard（@8.6M）：success_rate final 74% / max 80%，EV 0.72，value_loss 0.384
- Eval：最优 checkpoint（step 8,263,504）**77%**（300 局）；final model 76%（300 局）
- 对比 EXP-015a（±1.0，5M 步）：eval 65% → 本实验 **+12%**

**分析**：小幅 reward（win=0.1/lose=-0.1）配合 EXP-018 迁移在 6×6×5 上显著提升，EV 稳定在 0.70~0.74，无 EV 卡死问题。曲线在 8M 附近趋于 plateau，继续训练收益边际递减。最优 checkpoint（step 8,263,504，77%）作为 EXP-019b source。

**结论**：✅ 达成目标（eval 77% > 70%）。正迁移效果明显验证，A6 阶段假设成立。

---

## 终止奖励幅度对 policy 质量的影响——三种 reward 哲学对比（EXP-016/017/018）(2026-03-20)

**背景**：EXP-007（win=1.0/lose=-1.0）的 explained_variance 长期卡在 0.43，怀疑是终止信号幅度过大导致 value function 方差过高、拟合困难。设计三组对照实验，在保持其他超参不变的条件下，将 win/lose 幅度缩小至 reveal（0.1）量级，同时对比三种不同的奖惩哲学：

- **EXP-016**（win > |lose|）：死亡轻罚，胜利小奖，「随便探索」
- **EXP-017**（|lose| > win）：死亡重罚，胜利极小，「首要避死」
- **EXP-018**（win = |lose|）：对称幅度，方差理论最小化

三组均在 5×5×3 标准棋盘、2M 步 from scratch 训练后，统一进行对比分析。

### 指标对比（TensorBoard final model）

| 实验 | reward 配置 | EV final/max | value_loss final | success_rate final/max |
|------|------------|-------------|-----------------|----------------------|
| EXP-016 | win=0.2/lose=-0.05 | 0.84/0.84 | 0.245 | 77%/88% |
| EXP-017 | win=0.05/lose=-0.2 | 0.83/0.85 | 0.277 | 74%/87% |
| EXP-018 | win=0.1/lose=-0.1  | 0.84/0.87 | **0.242** | 77%/86% |

### 500 局 eval 对比（seed=123，final model）

| 实验 | eval 胜率 | 95% CI |
|------|----------|--------|
| EXP-016 | **86.0%** | [82.8%, 89.2%] |
| EXP-018 | 85.6% | [82.4%, 88.8%] |
| EXP-017 | 85.0% | [81.7%, 88.3%] |

**统计显著性**：三组两两比较均 p>0.65（proportion z-test），差异在噪声范围内无统计意义。初始 100 局 eval（017=89%，018=86%，016=81%）的差异是小样本偶然性，不代表真实排序。

### 收敛速度对比

- **到达 50% success_rate**：三组相同，约 507k 步
- **到达 70%**：EXP-016 最快（~966k），EXP-018 次之，EXP-017 最慢
- **到达 80%**：EXP-017 最快（~1,311k），EXP-016 次之，EXP-018 最慢
- **late-plateau（最后 200k 步平均）**：016=79.6% > 017=78.4% > 018=78.2%（差距微小）

### 后期稳定性

- **EXP-017**：value_loss 在 ~1.5M 步达到最低 0.230 后回弹至 final 0.277（+0.047），entropy 过快收敛至 -0.60，2M 步时仍处于动态调整中
- **EXP-016/018**：value_loss 单调下降至 final，无回弹，训练曲线稳定

### B4 总结

- **EV 瓶颈根因确认**：三组 EV 均恢复至 0.83~0.87（vs EXP-007 的 0.43），100% 证实终止信号幅度（±1.0）是 EV 卡死的根因
- **eval 三组统计等价**：reward 哲学对 5×5×3 最终胜率无显著影响（需更多局数或更难棋盘才能区分）
- **迁移学习推荐 EXP-018**：EV 最稳定（0.84 final，无回弹）、value_loss 最低（0.242）、无后期不稳定风险，是用于大棋盘课程学习的最优起点

---

## EXP-018 B4-3 对称极小 reward（5×5×3）(2026-03-20)

配置 `experiments/configs/exp_018_b4_reward_symmetric.yaml`，Run `mw_ppo_5x5x3_seed42_20260320034128`，2,015,232 步（2M，from scratch）。

**核心变量**（vs EXP-007）：reward_win 1.0→0.1，reward_lose -1.0→-0.1（对称极小）。

**指标**：
- TensorBoard（@2M）：见 `experiments/results/exp_018_metrics.json`；success_rate 7%→77% / max 86%，EV -0.19→0.84 / max 0.87，entropy_loss -2.66→-0.70，value_loss 0.511→0.242
- Eval：**85.6%**（500 局，seed=123，final model）

**分析**：EV 从 -0.19 升至 0.84（max 0.87），三组中最高且无回弹，late-training 最稳定。value_loss 0.242 是三组最低（016=0.245，017=0.277），value function 拟合质量最优。收敛速度：到达 50% 与其他两组相同（~507k 步）；到达 80% 稍慢（比 017 晚约 100k），但 late-plateau（最后 200k 步）success_rate 稳定。500 局 eval（seed=123）：85.6%，与 EXP-016（86.0%）和 EXP-017（85.0%）统计上无显著差异。从迁移学习角度看，EV 最稳定、value_loss 最低、无 017 的后期不稳定问题。

**结论**：✅ 对称极小 reward 成功恢复 EV=0.84，eval 85.6%，三组统计等价。EXP-018 是最优迁移起点，推荐用于后续大棋盘课程学习实验。

---

## EXP-017 B4-2 方向反转 reward（5×5×3）(2026-03-20)

配置 `experiments/configs/exp_017_b4_reward_lose_gt_win.yaml`，Run `mw_ppo_5x5x3_seed42_20260320033936`，2,015,232 步（2M，from scratch）。

**核心变量**（vs EXP-007）：reward_win 1.0→0.05，reward_lose -1.0→-0.2（|lose|>win，强避死）。

**指标**：
- TensorBoard（@2M）：见 `experiments/results/exp_017_metrics.json`；success_rate 9%→74% / max 87%，EV -0.18→0.83 / max 0.85，entropy_loss -2.66→-0.60（收敛最快），value_loss 0.533→0.277（final，但中途最低达 0.230 后后期回弹）
- Eval：**85.0%**（500 局，seed=123，final model）

**分析**：EV 0.83（max 0.85），与 EXP-003/016/018 同等级，确认 EV 恢复与 win/lose 比值无关，只与幅度有关。后期不稳定：value_loss 从最低 0.230 回弹至 final 0.277（+0.047），entropy 收敛至 -0.60（过快），表明 2M 步时正处于过拟合/策略坍缩边界。lose=-0.2 确实使"快死"成负回报（2 步死=0.1-0.2=-0.1），agent 比 016 更愿意探索；但 late-stage 不稳定说明 2M 步并非 017 的最优停点。

**结论**：⚠️ B4-2（|lose|>win）EV=0.83，500 局 eval 85%，与三组统计等价。「强避死」哲学行为上更谨慎，但 final model 出现 value_loss 后期回弹，训练尚未完全稳定。作为 transfer 基础时需谨慎，推荐 EXP-018 替代。

---

## EXP-016 B4-1 还原 EXP-003 reward（5×5×3）(2026-03-20)

配置 `experiments/configs/exp_016_b4_reward_win_gt_lose.yaml`，Run `mw_ppo_5x5x3_seed42_20260320032723`，2,015,232 步（2M，from scratch）。

**核心变量**（vs EXP-007）：reward_win 1.0→0.2，reward_lose -1.0→-0.05（还原 EXP-003 旧 reward）。

**指标**：
- TensorBoard（@2M）：见 `experiments/results/exp_016_metrics.json`；success_rate 10%→77% / max 88%，EV -0.19→0.84 / max 0.84，entropy_loss -2.66→-0.72，value_loss 0.508→0.245
- Eval：**86.0%**（500 局，seed=123，final model）

**分析**：EV 从 -0.19 升至 0.84，与 EXP-003（旧 reward，EV=0.84）完全匹配，远超 EXP-007（EV=0.43）——B4 核心假设证实：缩小 win/lose 幅度确实能恢复 value function 可预测性。lose=-0.05 极轻，agent 踩雷无痛感，2 步死局 reward=0.1-0.05=+0.05（正值），存在鲁莽倾向。收敛速度：最快到达 70% success_rate（~966k 步），但后期斜率平缓，最终与 017/018 持平。

**结论**：✅ B4-1 EV=0.84，500 局 eval 86%，与三组统计等价。B4 核心假设完全证实：缩小 win/lose 幅度从根本上恢复了 value function 可预测性。win>|lose| 哲学导致部分鲁莽行为，但 500 局 eval 与其他两组无显著差异。

---

## EXP-015c 8×8×10 第四阶段 (2026-03-19)

配置 `experiments/configs/exp_015c_8x8x10_stage4.yaml`，Run `mw_ppo_8x8x10_seed42_20260319080930`，6,012,928 步（~6M，新棋盘从头计数），迁移自 EXP-015b checkpoint（7×7×7 @3.5M，Conv 权重迁移）。

**核心变量**（vs EXP-011b）：增加 7×7×7 中间台阶（EXP-015b），迁移更充分的 7×7 策略。

**指标**：
- TensorBoard（@6M）：见 `experiments/results/exp_015c_metrics.json`；success_rate 全程 0~2% / max 2% / final 1%，EV -0.98→0.51，entropy_loss -3.85→-1.04，value_loss 0.657→0.440
- Eval：**0%**（100 局，@6M final model）

**分析**：EV 从 -0.98 缓慢爬升至 0.51，说明 value function 确实在学习状态价值，7×7 迁移权重有正迁移。但 success_rate 全程平坦（0~2%），policy 根本没有产生过有效的赢法——与 EXP-011b（直接 6×6→8×8）几乎无差异。根因不变：win/lose=±1.0 方差太大，在极稀疏的 win signal 下 value gradient 无效；即使给 policy 更充分的先验（7×7 中间台阶），8×8×10 的稀疏奖励仍然无法突破。更渐进的课程路径（5×5→6×6→7×7→8×8）相比直接跳跃（5×5→6×6→8×8）无显著改善。

**结论**：❌ eval 0%，方向 A5 穷尽。更渐进课程无法突破 8×8×10 的稀疏奖励瓶颈，根因仍是终止信号幅度（±1.0）导致 value 梯度在极稀疏环境下无效。下一步转向 B4 reward 重标定。

---

## EXP-015b 7×7×7 中间台阶 (2026-03-19)

配置 `experiments/configs/exp_015b_7x7x7_stage3.yaml`，Run `mw_ppo_7x7x7_seed42_20260319072120`，3,506,176 步（~3.5M，新棋盘从头计数），迁移自 EXP-015a checkpoint（6×6×5 @5M，Conv 权重迁移）。

**核心变量**（vs EXP-011b 直接 6×6→8×8）：插入 7×7×7 中间台阶，缩小难度跳跃。

**指标**：
- TensorBoard（@3.5M）：见 `experiments/results/exp_015b_metrics.json`；success_rate 0%→11% / max 16% / final 11%，EV -0.49→0.44 / max 0.58，entropy_loss -3.50→-1.01，value_loss 0.686→0.536
- Eval：**16%**（100 局，@3.5M final model）

**分析**：16% eval 说明 7×7×7 比 8×8×10 容易，6×6 迁移权重有效，但距 40% 目标有较大差距。success_rate 曲线在 3.5M 末期仍有上升趋势（未 plateau），继续训练可能突破 20-25%，但距 40% 仍遥远。EV max 0.58 是课程路径中最好的，说明 7×7 比 8×8 的奖励密度更友好（7×7×7 随机完成率约 33%）。即便如此，16% 远低于目标，将此 checkpoint 迁移到 8×8×10 只是在迁移"半生不熟"的策略。

**结论**：⚠️ eval 16%，未达 40% 目标，但已是课程链中最高中间站。将 checkpoint 迁移至 015c（8×8×10）；7×7 证明更渐进路径确有边际改善（vs EXP-011b），但幅度不足以突破根本瓶颈。

---

## EXP-015a 6×6×5 续训收敛 (2026-03-19)

配置 `experiments/configs/exp_015a_6x6x5_continue.yaml`，Run `mw_ppo_6x6x5_seed42_20260318040414_continue_20260319064320`，续训自 EXP-011a checkpoint（@2M），累计约 5M 步在 6×6×5 上（续训 3M，共 5,029,888 步）。

**核心变量**（vs EXP-011a）：充分训练至 plateau（EXP-011a 只训 2M 步、38%，曲线未 plateau 就转移是 EXP-011b 失败根因）；同时缩小课程跳跃（6×6→7×7 而非直接 6×6→8×8）。

**指标**：
- TensorBoard（@5M）：见 `experiments/results/exp_015a_metrics.json`；success_rate 21%→58%（续训段）/ max 65% / final 58%，EV final 0.23 / max 0.32，entropy_loss -0.84→-0.53，value_loss 1.019→1.060（末期轻微上升）
- Eval：**65%**（100 局，@5M final model）

**分析**：65% eval 达到目标（≥60%），6×6×5 充分收敛，可作为 015b 迁移来源。success_rate 曲线在续训后段趋于 plateau，说明 5M 步已充分挖掘 6×6×5 的潜力。EV 0.23 偏低（win=1.0 的结构性问题），但对 policy 实际有效——65% eval 证明策略有实质改善。

**结论**：✅ eval 65%，达成 ≥60% 目标，6×6×5 充分收敛。将 final checkpoint 迁移至 015b（7×7×7）。

---

## EXP-014 学习率 Cosine 衰减 8×8×10 (2026-03-18)

配置 `experiments/configs/exp_014_cosine_lr.yaml`，Run `mw_ppo_8x8x10_seed42_20260318115101`，5M 步（from scratch）。

**核心变量**（vs EXP-009）：`lr_schedule: "cosine"`，`learning_rate: 0.001`（10× EXP-009），`lr_end: 0.0001`（EXP-009 基准值），其余超参完全相同。假设：先用高 LR 快速探索奖励空间，cosine 衰减到低 LR 精细收敛，避免固定小 LR 初期学不动。

**指标**：
- TensorBoard（@5M）：见 `experiments/results/exp_014_metrics.json`；success_rate 全程 0% / max 1%，EV max 0.55 / final 0.52，entropy_loss -3.84→-1.08，policy_gradient_loss 早期波动更大（min -0.062）
- Eval：无法加载（lambda 闭包序列化 bug，已修复但本次 checkpoint 无法复用）

**分析**：高 LR（0.001）早期产生更大梯度更新，但 success_rate 全程 0%，说明 8×8×10 的瓶颈不是探索量或 LR schedule 问题，而是稀疏奖励本身——模型根本得不到足够的胜利信号来学习策略，无论 LR 多高都无法突破。Bug 记录：lambda 闭包捕获 `math` module 导致 pickle 序列化失败，已修复为 `from math import cos, pi`。

**结论**：❌ cosine LR schedule 对 8×8×10 无效。超参调优方向（B1）已排除。稀疏奖励瓶颈需要从根本上改变奖励密度（如更小棋盘渐进 A5）或算法层面改进（如 HER、ICM 内在激励）。

---

## EXP-013 揭格时进度奖励 8×8×10 (2026-03-18)

配置 `experiments/configs/exp_013_reveal_progress_reward.yaml`，Run `mw_ppo_8x8x10_seed42_20260318102659`，3.4M 步（提前终止，中间结果已足够判断）。

**核心变量**（vs EXP-009）：`reward_progress_coef` 0.0→1.0（揭格时乘数：`reward *= (1 + coef * safe_revealed_ratio)`，踩雷惩罚保持 -1.0 不变），修复 EXP-012 设计缺陷——进度 bonus 移到安全揭格时而非踩雷时。

**指标**：
- TensorBoard（@3.4M）：见 `experiments/results/exp_013_metrics.json`；success_rate 全程 0% / max 1%，ep_rew_mean 2.39→3.75 / max 4.51（仍显著虚高），EV max 0.50 / final 0.48，entropy_loss -3.85→-1.40
- Eval（@3.4M）：**0%**（100 局）

**分析**：揭格乘数 `× (1 + coef * ratio)` 同样造成 reward hacking：模型学会多揭几格（获高乘数奖励）但无需赢——揭 10 格 × 乘数 >> 胜利奖励的期望值。ep_rew_mean 虚高模式与 EXP-012 几乎相同，说明奖励乘数和奖励加法都会导致同一问题：只要进度信号改变了不同动作的相对价值，模型就会找到绕过"真正赢"的捷径。explained_variance 进一步下降（0.48 < EXP-009 的 0.53）：进度乘数扭曲了奖励分布，value function 更难拟合。

**结论**：❌ 揭格时进度奖励乘数同样无效。任何形式的进度奖励都在隐式改变"赢"的相对激励——在 8×8×10 的稀疏环境下，模型总能找到通过最大化中间奖励而非胜利来获取更高期望回报的策略。

---

## EXP-012 完成进度奖励塑形 8×8×10 (2026-03-18)

配置 `experiments/configs/exp_012_progress_reward.yaml`，Run `mw_ppo_8x8x10_seed42_20260318083104`，5M 步（from scratch）。

**核心变量**（vs EXP-009）：`reward_progress_coef` 0.0→1.0（踩雷时叠加进度奖励：`reward_lose + coef * (safe_revealed / total_safe)`），其余超参完全相同。

**指标**：
- TensorBoard：见 `experiments/results/exp_012_metrics.json`；success_rate 全程 0% / max 2%，ep_rew_mean 1.68→2.93 / max 3.33（显著高于 EXP-009 的约 0.8，但系虚高），EV -0.009→0.42 / max 0.46，entropy_loss -3.85→-1.22，value_loss 0.76→0.56
- Eval：**0%**（100 局，seed=42，@5M 步 final model）

**分析**：ep_rew_mean 虚高：模型确实在最大化奖励，但学会的是靠少量 reveal（0.1×格）+ progress_bonus 的组合——eval 显示大量局只走 2-4 步即死，奖励却是正值（progress_bonus 把 -1.0 拉到 +0.5~+3.7）。`reward_progress_coef=1.0` 把死亡惩罚从 -1.0 软化到最低 -0.17，实际上降低了踩雷的代价，让模型更愿意"随便踩一颗雷结束游戏"。explained_variance 0.42（低于 EXP-009/011b 的 0.53/0.56）：value function 拟合更差，说明奖励分布被 progress_bonus 扭曲。

**结论**：❌ `reward_progress_coef=1.0` 的进度奖励塑形无效甚至有害。根本原因：踩雷惩罚被软化后，模型发现"快速死"是性价比最高的策略。需保持踩雷惩罚不变，仅在安全揭格时给额外奖励（而非踩雷时给补偿）。

---

## EXP-011 分阶段课程学习 5×5×3 → 6×6×5 → 8×8×10 (2026-03-18)

### Stage 2：6×6×5 中间棋盘

配置 `experiments/configs/exp_011a_stage2_6x6x5.yaml`，Run `mw_ppo_6x6x5_seed42_20260318040414`，2M 步（从头计数，transfer 只迁移 Conv 权重），迁移自 EXP-007 最优（`mw_ppo_5x5x3_seed42_20260317041904`，step 1,750,000）。

**核心变量**（vs EXP-010）：在 5×5→8×8 之间插入 6×6×5 中间台阶；6×6×5 密度 14%，随机完成率约 40%，远高于 8×8×10（17%），能提供足够的奖励信号。

**指标**：
- TensorBoard：见 `experiments/results/exp_011a_metrics.json`；success_rate 0%→38% / max 38%（单调上升，final = max，尚未收敛），EV max 0.49 / final 0.30（末期下降），entropy_loss -3.13→-0.83，value_loss 0.47→1.02（末期上升），fps 早期 ~1600 / 末期 ~970
- Eval：**38%**（100 局，seed=42，@2M 步 final model）

**分析**：38% eval 说明 6×6×5 确实比 8×8×10 容易学（EXP-009/010 全程 0%），中间台阶假设成立。但未达到 50% 目标：success_rate 曲线单调上升未 plateau，说明还在学，可能需要 3-4M 步。explained_variance 末期从 0.49 下降到 0.30，value_loss 同步上升：策略在后期快速进步但 value function 滞后。Conv 权重迁移确实起到预热作用：第 49k 步时 EV 已达 0.38（EXP-009 from scratch 早期几乎为 -0.004）。

**结论**：⚠️ Stage 2 eval 38%，低于 50% 成功线，但仍有实质学习。曲线未 plateau 说明更多步数可能突破 50%。当前 2M final checkpoint 作为 Stage 3 的迁移来源。

### Stage 3：8×8×10 目标棋盘

配置 `experiments/configs/exp_011b_stage3_8x8x10.yaml`，Run `mw_ppo_8x8x10_seed42_20260318050656`，5M 步（从头计数），迁移自 `mw_ppo_6x6x5_seed42_20260318040414` step 2,000,000（38%）。

**核心变量**（vs EXP-010）：迁移来源从 5×5×3 改为 6×6×5 最优 checkpoint。

**指标**：
- TensorBoard：见 `experiments/results/exp_011b_metrics.json`；success_rate 全程 ~0% / max 2%，EV -0.63→0.51 / max 0.56，entropy_loss -3.86→-1.09，value_loss 0.89→0.44，fps 稳定 ~1550
- Eval：**1%**（100 局，seed=42，@5M 步 final model，赢 1/100）

**分析**：eval 1% vs EXP-009/010 的 0%：成功标准形式上达成，但 1% 在统计上几乎无意义。success_rate max 2%，与 EXP-010（max 1%）几乎没有差异——6×6×5 中间台阶的 Conv 权重对 8×8×10 的帮助与 5×5×3 直接迁移相比没有显著改善。explained_variance 轨迹（0.51/0.56）与 EXP-009（0.53/0.57）和 EXP-010（0.51/0.56）几乎完全相同，说明中间台阶不影响 value function 的学习质量。8×8×10 的稀疏奖励瓶颈（随机完成率 ~17%）仍然是根本障碍。

**结论**：❌ EXP-011 分阶段课程学习 eval 1%，仅象征性突破，无实质提升。根本瓶颈确认是稀疏奖励本身而非特征迁移质量。要真正攻克 8×8×10，需要从根本改变奖励密度，或从更易任务渐进（7×7 等更小步长），或接受需要 10M+ 步数量级的训练。

---

## EXP-010 课程学习 5×5×3 → 8×8×10 (2026-03-17)

配置 `experiments/configs/exp_010_curriculum_5x5_to_8x8.yaml`，Run `mw_ppo_8x8x10_seed42_20260317152622`，5M 步（从头计数，transfer 只迁移权重，不继承 timesteps），迁移自 `mw_ppo_5x5x3_seed42_20260317041904` step 1,750,000（EXP-007 最优，87%）。

**核心变量**（vs EXP-009）：增加 Conv 权重迁移（EXP-007 @1.75M steps），其余超参不变。假设：Conv 权重（spatial pattern detector）在 5×5 和 8×8 棋盘间可直接迁移（shape 相同），能加速 8×8×10 策略学习，让模型在 5M 步内突破 0% 胜率。

**指标**：
- TensorBoard：见 `experiments/results/exp_010_metrics.json`；success_rate 全程 0% / max 1%，EV -0.52→0.51 / max 0.56，entropy_loss -3.85→-1.08（比 EXP-009 的 -1.33 保留了更多探索），value_loss 0.72→0.45，fps 稳定 ~1590
- Eval：**0%**（100 局，seed=42，@5M 步，0/100 wins）

**分析**：Conv 权重迁移对 8×8×10 学习无实质帮助——success_rate 全程 0%，与 EXP-009（from scratch）完全一致。explained_variance 轨迹几乎相同，说明 value function 拟合能力没有因迁移而提升。唯一差异：entropy 收敛更慢（-1.08 vs -1.33），说明策略"更困惑"——可能是 Conv 迁移权重与随机初始化的 Linear/action head 形成不一致。核心问题不在特征提取，而在奖励稀疏性本身：8×8×10 随机探索完成一局概率 ≈ 17%，5M 步内几乎不可能从偶然胜利中学习。

**结论**：❌ 仅迁移 Conv 权重的课程学习策略在 5M 步内无效。根本瓶颈是稀疏奖励，而非特征质量；单靠 Conv 层迁移无法绕过"首次胜利"的探索鸿沟。需要中间棋盘尺寸过渡，或奖励塑形，或更长步数（10M+）。

---

## EXP-009 8×8×10 基准训练 (2026-03-17)

配置 `experiments/configs/exp_009_8x8x10_baseline.yaml`，Run `mw_ppo_8x8x10_seed42_20260317092525`，5M 步（从头训练）。

**核心变量**（vs EXP-007）：棋盘 5×5×3 → 8×8×10，步数 2M → 5M，checkpoint_freq 50k → 100k。假设：EXP-007 最优配置可直接迁移到更大棋盘；5M 步能建立有效基准胜率。

**指标**：
- TensorBoard：见 `experiments/results/exp_009_metrics.json`；success_rate 全程 0% / max 1%，EV -0.004→0.53 / max 0.57，entropy_loss -3.85→-1.33，value_loss 0.63→0.46，fps 稳定 ~1566
- Eval：**0%**（100 局，seed=42，@5M 步，0/100 wins）

**分析**：从头训练 5M 步无法让 8×8×10 模型获得任何胜利。策略确实在学习（entropy 降低、ep_len_mean 从 5→6），但模型陷入"尽量多揭格但总踩雷"的局部最优。8×8 棋盘有 54 个非地雷格 vs 5×5 的 22 个，随机探索完成一局的概率极低（约 (54/64)^10 ≈ 17%），导致 reward signal 稀疏，策略无法从偶然胜利中学习。explained_variance 0.53 说明 value function 在学习"什么格子更容易踩雷"，但策略层面无法转化为胜利。

**结论**：❌ from scratch 5M 步对 8×8×10 不够。需要课程学习（从 EXP-007 5×5×3 最优权重迁移）或更多步数（10M+）。优先推荐课程学习，因为 5×5 已有 87% 的有效策略可以迁移。

---

## EXP-008 扩大网络容量 (2026-03-17)

配置 `experiments/configs/exp_008_larger_network.yaml`，Run `mw_ppo_5x5x3_seed42_20260317075354`，2M 步（从头训练）。

**核心变量**（vs EXP-007）：features_dim: 128 → 256（不能续训 EXP-007，权重维度不兼容）。假设：EXP-007 双通道 explained_variance 仍卡在 0.43，features_dim=128 的 Linear 层输出瓶颈导致 value function 无法拟合更大方差；扩容至 256 应能让 explained_variance 回升至 0.6+。

**指标**：
- TensorBoard：见 `experiments/results/exp_008_metrics.json`；success_rate 9%→87% / max 89%，EV max 0.468 / final 0.425（仅微升，远未达到 0.6 目标），entropy_loss -2.67→-0.47，value_loss 0.948→0.808，fps 早期 ~1700 / 末期 ~1234
- Eval：**86%**（100 局，seed=42，@2M 步）——与 EXP-007 完全相同

**分析**：eval 86% 与 EXP-007 完全持平——features_dim 加倍对最终胜率无任何改善。explained_variance max 仅从 0.43 升到 0.47，完全未达到 0.6 目标，网络容量假设被否定。对比 EXP-003（旧 reward，features_dim=128，explained_var=0.84）：相同网络容量在旧 reward 下能完美拟合；新 reward 下无论加多少容量，explained_var 都卡在 0.43-0.47。核心结论：explained_variance 瓶颈不在网络容量，而在于 value function 本身面对的信息量/任务难度（新 reward win=1.0 导致单局结果方差极高）。

**结论**：❌ features_dim 128→256 无效。网络容量不是 explained_variance 瓶颈的根本原因。需另辟蹊径：考虑延长训练步数、lr schedule、分开 pi/vf 网络，或接受当前 ~0.45 的 explained_var 上限，转而专注提升 eval 胜率。

---

## EXP-007 多通道观测 (2026-03-17)

配置 `experiments/configs/exp_007_multichannel_obs.yaml`，Run `mw_ppo_5x5x3_seed42_20260317041904`，2M 步（从头训练）。

**核心变量**（vs EXP-006）：obs_channels: 1 → 2（ch0=is_unrevealed, ch1=neighbor_counts/8.0）。假设：单通道将"未揭开"(-2→0.0) 和"已揭数字"(0-8→0.2-1.0) 线性混合，value function 无法区分两类语义不同的状态；双通道分离编码应能让 value function 更好拟合，explained_variance 从 0.44 回升至 0.6+。

**指标**：
- TensorBoard：见 `experiments/results/exp_007_metrics.json`；success_rate 10%→83% / max 88%（历史最高），EV max 0.46 / final 0.43（与 EXP-005/006 几乎相同，未突破），entropy_loss -2.66→-0.42，value_loss 0.96→0.78
- Eval：**86%**（100 局，seed=42，@2M 步）

**分析**：eval 86% 是历史最高，超越 EXP-005（84%）和 EXP-006（80%）——双通道观测带来实际胜率提升。但 explained_variance 依然卡在 0.43，假设"双通道能突破 value function 拟合瓶颈"未被验证。双通道早期收敛更快（50k 步 EV 已达 0.35+，EXP-006 早期几乎为负），说明信息表示更清晰。对比 EXP-003（旧 reward）explained_variance=0.84，差距仍巨大。value_loss 下降更多（0.78 vs 0.91）但 explained_variance 没有跟上，说明网络在拟合均值，但方差解释能力受限于网络容量。

**结论**：✅ 双通道观测值得保留（胜率 +2~6%，早期收敛更快），但不是 explained_variance 瓶颈的根本原因。真正的瓶颈是网络容量不足。下一步：EXP-008 = 双通道 + features_dim 128 → 256。

---

## EXP-006 修复 vf_coef（2026-03-17）

配置 `experiments/configs/exp_006_vf_coef_fix.yaml`，Run `mw_ppo_5x5x3_seed42_20260317031520`，2M 步（从头训练）。

**核心变量**（vs EXP-005）：vf_coef: 0.5 → 1.0。假设：EXP-004/005 的 explained_variance 长期卡在 0.44，是因为 vf_coef=0.5 在新 reward 量级（win=1.0）下 value gradient 不足；提升到 1.0 应能让 value function 充分拟合，进而带动 eval_win_rate 突破 84%。

**指标**：
- TensorBoard：见 `experiments/results/exp_006_metrics.json`；success_rate 10%→75% / max 79%，EV max 0.52 / final 0.46（稍高于 EXP-005 的 0.44，但远未回升到 0.7+），entropy_loss -2.67→-0.36，value_loss 0.97→0.91（仅微降）
- Eval：**80%**（100 局，seed=42，@2M 步）

**分析**：eval 80% 比 EXP-005（84%）还低——vf_coef 加倍不仅没有帮助，反而可能挤压了策略梯度。explained_variance max 仅 0.52，提升极其微弱，说明 vf_coef 不是 explained_variance 偏低的根本原因。value_loss 基本没变（0.91 vs 0.97），说明 value function 的拟合能力受限于特征质量，而非优化力度。核心结论：瓶颈在于观测表示/网络容量，而非训练超参。

**结论**：❌ vf_coef 修复无效，排除超参层面的解释。瓶颈是架构层面——需要更丰富的观测编码（多通道）或更大网络（features_dim 256）。下一步：EXP-007 = 多通道观测。

---

## EXP-005 续训 EXP-004 到 2M 步 (2026-03-16)

配置 `experiments/configs/exp_005_continue_reward_2m.yaml`，Run `mw_ppo_5x5x3_seed42_20260316134208_continue_20260316144019`，续训自 EXP-004（1M 步），额外 1,000,000 步（目标总步数约 2M）。

**核心变量**：步数（EXP-004 基础上续训，所有超参不变）。假设：EXP-004 @1M 已达 75%（超过 EXP-002 同期 52%），续训到 2M 步后期望超越 EXP-003 @2M 的 83%。

**指标**：
- TensorBoard：见 `experiments/results/exp_005_metrics.json`；success_rate 66%→74% / max 83%（本段新增 +8%），EV 0.44-0.46 区间震荡（未能回升，EXP-003 达到 0.84，差距巨大），entropy_loss -0.57→-0.40，value_loss 1.55→0.97（下降 37%）
- Eval：**84%**（100 局，seed=42，@2M 步）

**分析**：eval 84% 略超 EXP-003（83%），但只差 1%，不能认为是显著突破。explained_variance 全程卡在 0.44 附近，续训 1M 步没有改善——这是结构性问题，不是步数问题。value_loss 从 1.55 下降到 0.97，说明 value function 在缓慢追赶，但最终仍未充分拟合。

**结论**：⚠️ 新 reward @2M = 84%，与旧 reward @2M（EXP-003 = 83%）几乎持平。续训没有带来明显突破。explained_variance 长期偏低（0.44 vs 0.84）是核心瓶颈，下一步必须修复 vf_coef（0.5 → 1.0）。

---

## EXP-004 Reward Shaping 验证 (2026-03-16)

配置 `experiments/configs/exp_004_reward_shaping.yaml`，1M 步（够看趋势，节省 GPU），其余超参与 EXP-002/003 完全一致。

**核心变量**（vs EXP-002/003）：reward_win: 0.2 → 1.0，reward_lose: -0.05 → -1.0（win 信号占比 8% → 31%）。假设：EXP-002/003 中 reward_win=0.2 导致 win 信号仅占总奖励 8%，agent 更倾向"安全揭格"而非"追求胜利"。

**指标**：
- TensorBoard：见 `experiments/results/exp_004_metrics.json`；success_rate 10%→64% / max 68%（vs EXP-002 同期 53%，+11%），ep_rew_mean 0.75→2.31，EV final 0.47（相比 EXP-003 的 0.84 明显更差），entropy_loss -2.67→-0.58
- Eval：**75%**（100 局，seed=42，@1M 步）

**分析**：@1M 步 eval 75%，远超目标 65%，也接近 EXP-003 @2M 步的 83%。success_rate @1M 64% vs EXP-002 @800k 53%，同等步数下提升明显，说明 reward 确实是瓶颈。explained_variance 0.47 偏低（EXP-003 达 0.84），value function 拟合不好——reward 量级变大（win=1.0 vs 0.2）导致 value 估计更难，可能需要更大的 vf_coef 或更长训练。单纯 1M 步已追上 EXP-003 2M 步 80% 的水平，效率提升显著。

**结论**：✅ reward 是主要瓶颈，修复后 1M 步 eval 75%（vs EXP-002 @800k 52%）。下一步可叠加更大网络或继续训练到 2M 步看能否超越 EXP-003 83%。

---

## EXP-003 续训到 2M 步 (2026-03-16)

配置 `experiments/configs/exp_003_continue_2m.yaml`，Run `mw_ppo_5x5x3_seed42_20260316121733_continue_20260316124704`，续训自 EXP-002（800k 步），额外 1,196,032 步（实际总步数 2,016,752）。

**核心变量**：步数（EXP-002 基础上续训，所有超参不变）。

**指标**：
- TensorBoard：见 `experiments/results/exp_003_metrics.json`；success_rate 47%→68% / max 77%（本段新增 +30%），EV 0.84（稳定，value function 拟合良好），entropy_loss -0.92→-0.51
- Eval：**83%**（100 局，seed=42，final_model）

**分析**：续训后 success_rate 从 47% 持续爬升到 77%，到 2M 步末尾曲线仍在 68-77% 间震荡，尚未完全 plateau。eval 胜率 83% 显著超过 TensorBoard success_rate（训练时 ~70%），说明 eval 环境更"幸运"或模型在确定性 seed 下表现更稳定。相比 EXP-002，仅增加步数（唯一变量），胜率从 52% → 83%，涨幅 +31%，说明步数是当前的主要瓶颈。2M 步后仍有轻微上升趋势，但增速明显放缓，可能正在逼近当前超参下的收敛上限。

**结论**：✅ 步数从 800k → 2M 带来显著提升（52% → 83%）。当前超参下收敛上限约在 75-85%。建议下一步尝试超参调整（更大网络、learning rate 衰减）以突破瓶颈。

---

## EXP-002 1M 步规模验证 (2026-03-16)

配置 `experiments/configs/exp_002_1m_scale.yaml`，Run `mw_ppo_5x5x3_seed42_20260316121733`，实际步数 802,816（因本地断连提前终止，目标 1M）。

**指标**：
- TensorBoard：见 `experiments/results/exp_002_metrics.json`；success_rate 10%→53%（+43%，明显收敛），EV final 0.84（value function 拟合良好），entropy_loss -2.67→-0.90
- Eval：**52%**（100 局，seed=42，final_model）

**分析**：800k 步内胜率从 10% 稳定收敛到 53%，证明 agent 确实能学会扫雷。训练曲线有明显上升趋势，还未完全 plateau，继续训练应有提升空间。eval 胜率 52% 与 TensorBoard success_rate 53% 吻合，结果可信。本次因本地 Modal 客户端断连中断（应使用 `modal run --detach`）。

**结论**：✅ agent 能学习，800k 步达到 52% 胜率。训练未完全收敛，下一步考虑：（1）续训到 1M+ 步看是否继续提升；（2）调整超参（learning_rate 衰减、更大网络）；（3）多通道观测改善特征表示。

---

## EXP-001 基线验证 (2026-03-16)

配置 `experiments/configs/exp_001_baseline_100k.yaml`，Run `mw_ppo_5x5x3_seed42_20260316104017`，100k 步。

**指标**：
- TensorBoard：见 `experiments/results/exp_001_metrics.json`；success_rate 7%-21% 之间剧烈波动（无明显收敛趋势），EV final 0.74，entropy_loss 约 -1.69
- Eval：**18%**（100 局，seed=42，frozen final_model）

**分析**：100k 步训练量严重不足，无法判断模型真实能力上限。success_rate 剧烈波动说明策略尚未稳定。

**结论**：⚠️ 100k 步只够验证流程可行性，需要 ≥500k 步才能看到收敛趋势。下一步：EXP-002 跑 1M 步（用 colab_config.yaml 参数），看收敛上限。
