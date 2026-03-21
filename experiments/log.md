# 实验日志

## 实验汇总

| 实验 | 棋盘 | 训练步数 | 核心变量 | eval | 结论 |
|------|------|---------|---------|------|------|
| EXP-001 | 5×5×3 | 100k | 基线 | 18% | ➖ 步数严重不足，流程验证 |
| EXP-002 | 5×5×3 | 800k | 扩步数 | 52% | ➖ 步数是早期主要瓶颈 |
| EXP-003 | 5×5×3 | 2M | win=0.2/lose=-0.05 | 83% | ✅ EV=0.84，小幅奖励下的自然上限 |
| EXP-004 | 5×5×3 | 1M | reward_win 0.2→1.0 | 75% | ⚠️ 大幅奖励提升早期效率，但 EV 下降至 0.47 |
| EXP-005 | 5×5×3 | 2M | 续训 EXP-004 | 84% | ⚠️ EV 卡 0.44，±1.0 瓶颈首次出现 |
| EXP-006 | 5×5×3 | 2M | vf_coef 0.5→1.0 | 80% | ❌ vf_coef 非瓶颈，单独用反而 -4% |
| EXP-007 | 5×5×3 | 2M | obs_channels 1→2 | **87%** | ✅ 双通道最优配置，早期收敛更快 |
| EXP-008 | 5×5×3 | 2M | features_dim 128→256 | 86% | ❌ 容量非瓶颈，+29% 算力无收益 |
| EXP-009 | 8×8×10 | 5M | from scratch | **0%** | ❌ 稀疏奖励瓶颈，success_rate 全程 0% |
| EXP-010 | 8×8×10 | 5M | 5×5×3 Conv 直接迁移 | **0%** | ❌ 权重迁移无效，与 from scratch 无差异 |
| EXP-011a | 6×6×5 | 2M | 课程 Stage2（from 5×5×3） | 38% | ⚠️ 中间棋盘可学，但未 plateau |
| EXP-011b | 8×8×10 | 5M | 课程 Stage3（from 6×6×5） | **1%** | ❌ 象征性突破，±1.0 奖励下的上限 |
| EXP-012 | 8×8×10 | 5M | 进度奖励（踩雷时 bonus） | **0%** | ❌ reward hacking：模型学会快速死 |
| EXP-013 | 8×8×10 | 3.4M | 进度奖励（揭格乘数） | **0%** | ❌ reward hacking，提前终止 |
| EXP-014 | 8×8×10 | 5M | LR cosine 衰减（0.001→0.0001） | **0%** | ❌ 稀疏奖励瓶颈，LR 无关 |
| EXP-015a | 6×6×5 | 5M | 渐进课程 Stage2 | **65%** | ✅ 充分收敛，达成目标 |
| EXP-015b | 7×7×7 | 3.5M | 渐进课程 Stage3（from 015a） | **16%** | ⚠️ 有学习，未达 40% 目标，步数不足 |
| EXP-015c | 8×8×10 | 6M | 渐进课程 Stage4（from 015b） | **0%** | ❌ ±1.0 下课程路径穷尽，确认奖励是根因 |
| EXP-016 | 5×5×3 | 2M | win=0.2/lose=-0.05 | **86%** | ✅ EV=0.84，三组统计等价（500局） |
| EXP-017 | 5×5×3 | 2M | win=0.05/lose=-0.2 | **85%** | ⚠️ EV=0.83，后期 value_loss 回弹，不稳定 |
| EXP-018 | 5×5×3 | 2M | win=0.1/lose=-0.1 | **85.6%** | ✅ EV=0.84，最稳定 |
| EXP-019a | 6×6×5 | 8.6M | 课程 Stage2（from EXP-018） | **77%** | ✅ 小幅 reward 显著改善（+12% vs EXP-015a） |
| EXP-019b | 7×7×7 | 13M | 课程 Stage3（from 019a） | **51.6%** | ✅ +35.6% vs EXP-015b，需 10M+ 步收敛 |
| EXP-019c | 8×8×10 | 8M | 课程 Stage4（from 019b） | **~0%** | ❌ 8×8×10 仍是硬瓶颈，reward 幅度非关键因素 |
| EXP-020 | 6×6×5 | 6M | obs_ch=3，n_envs=16（from 019a） | **75.4%** | ⚠️ 加速早期收敛，精度提升不显著（p=0.77），后期下滑 |
| EXP-021 | 6×6×5 | 6M | obs_ch=3，n_envs=64（from 019a） | **73.0%** | ⚠️ 57min 达同等水平，精度不优于基线（p=0.25） |
| EXP-022 | 7×7×7 | 7M续训（共20M） | 延长至 20M 步 | **56%** | ✅ +4.4% vs 019b，entropy 未完全 plateau |
| EXP-023f | 5×5×3 | 500K | obs_ch=3，n_envs=64，LR=1e-3 | **85%** | ✅ 与 EXP-018（2M步）持平，速度 4x，**新基线** |
| EXP-024 | 5×5×3 | 500K | cosine LR（1e-3→1e-4，023f 配置不变） | **84.7%** | ⚠️ EV 小幅提升（0.74→0.78），eval 持平，短训练无必要 |
| EXP-025 | 6×6×5 | 2M | TRANSFER 023f + constant LR=1e-3，n_envs=64，obs_ch=3 | **75%** | ⚠️ 2M步达75%，但振荡明显，LR过高无法收敛，不如 cosine |
| EXP-026 | 6×6×5 | 2M | TRANSFER 023f + cosine LR（1e-3→1e-4），n_envs=64，obs_ch=3 | **77.7%** | ✅ 2M步持平 EXP-019a（77%@8.6M），步数减少77%，**新665基线** |

---

## EXP-025 / EXP-026 6×6×5 TRANSFER + 高 LR（2026-03-21）

两组实验共用设计：TRANSFER_FROM EXP-023f（`mw_ppo_5x5x3_seed42_20260321115151`，5×5×3 eval 85%），迁移 Conv 25 层，LR 起点均为 1e-3，n_envs=64，obs_channels=3，2M steps。唯一变量：LR 调度方式。

**EXP-025**（`exp_025_665_transfer_lr1e3.yaml`）：Run `mw_ppo_6x6x5_seed42_20260321124422`（第一次 run `20260321123538` 被 SIGKILL —— approx_kl 持续飙升至 0.10 引发 OOM，于 490K 步中断）
**EXP-026**（`exp_026_665_transfer_cosine_lr.yaml`）：Run `mw_ppo_6x6x5_seed42_20260321123727`（一次完成，无中断）

注：EXP-025 constant LR=1e-3 在 665 场景下 approx_kl 无法收敛，OOM 后重启；EXP-026 cosine LR 自然压制了 KL 发散，全程稳定运行。

**指标对比**：

| 指标 | EXP-025（constant LR） | EXP-026（cosine） | EXP-019a（旧基线） |
|------|------------------------|-------------------|-------------------|
| 步数 | 2M | 2M | **8.6M** |
| Max SR | 76% | **82%** | — |
| Final SR | 70% | **82%** | — |
| EV（最终）| 0.697 | 0.693 | — |
| entropy（最终）| -0.919 | **-0.892** | — |
| clean eval（300局）| 75.0% | **77.7%** | **77%**（500局）|

**训练曲线特征**：
- EXP-025：approx_kl 全程 0.07-0.10（偏高），SR 振荡剧烈（0.13→0.42→0.28→0.42→0.33→0.42），说明 LR=1e-3 在 665 迁移场景下步子迈太大，无法稳定收敛
- EXP-026：cosine 从 1e-3 衰减到 1e-4，KL 随之平稳下降，SR 曲线从 400K 步起单调爬升至 82%

**分析**：
1. **665 比 553 更需要 cosine**：553 短训练中 constant LR 胜出，但 665 的 action space 更大（36 格），高 LR 导致 KL 持续偏高、策略更新不稳定，cosine 衰减正好解决了这个问题
2. **TRANSFER + cosine 的组合是关键**：迁移 Conv 权重给了好的特征起点，cosine 让策略在此基础上平稳精调，二者缺一不可
3. **2M 步达到 EXP-019a 水平（77% @ 8.6M）**：步数减少 **77%**（8.6M→2M），wallclock 大幅降低
4. **EXP-025 不是失败**：constant LR 的 75% 也已超过 EXP-015a（65%），但 cosine 版本在同等步数下明显更优

**结论**：✅ EXP-026 确立新 6×6×5 基线：**TRANSFER_FROM 023f + cosine LR（1e-3→1e-4）+ n_envs=64 + obs_ch=3 + 2M steps → eval 77.7%**。后续 7×7×7 课程实验应以此为 Stage2 起点。

---

## EXP-024 5×5×3 Cosine LR（2026-03-21）

配置 `experiments/configs/exp_024_553_cosine_lr.yaml`，Run `mw_ppo_5x5x3_seed42_20260321124701`（第一次 run `20260321123339` 被 Modal SIGKILL 后自动重启，两次训练曲线完全一致）。

**核心变量**（vs EXP-023f）：`lr_schedule=cosine`，`lr_end=0.0001`，其余参数完全相同。

**指标**：
- TensorBoard：success_rate max **83%**（@491K），final **77%**，EV **0.784**，entropy_loss -1.40
- Eval（@499K，300 局）：**84.7%**
- 训练时间：~4.5 min（500K steps，fps≈1762-3006，重启后更快）

**与 EXP-023f 对比**：

| 指标 | EXP-023f（constant LR=1e-3） | EXP-024（cosine 1e-3→1e-4） |
|------|------------------------------|------------------------------|
| Max SR | **84%** | 83% |
| Final SR | **81%** | 77% |
| EV（最终）| 0.741 | **0.784** |
| entropy（最终）| -1.429 | **-1.400** |
| clean eval | **85.0%**（300局） | 84.7%（300局） |

**训练曲线特征**：收敛轨迹与 023f 完全重合至约 230K 步（LR 仍接近 1e-3），此后随 LR 衰减开始分叉——SR 继续爬升但速度减慢，final SR 从峰值 83% 回落至 77%，说明 LR 过早降到低值切掉了后期探索能力。EV 提升是真实的（0.74→0.78），但不影响 eval 结果。

**发现的 Bug**：cosine schedule 的 lambda 被 cloudpickle 序列化进 checkpoint 后，在本地 eval 时因 `math.cos` 作用域问题反序列化失败。已在 `src/factories/model_factory.py:load_model_from_checkpoint` 中加 `TypeError` fallback（`custom_objects={"learning_rate": 0.0001}`）修复。

**分析**：
1. **cosine 衰减在 500K 短训练中适得其反**：LR 在后 1/4 阶段（375K-500K）已降至 3e-4 以下，过快压制了探索，导致 final SR 比 constant LR 低 4%
2. **EV 提升（+0.04）是真实信号但不实用**：value function 确实拟合更好，但在这个阶段不是瓶颈
3. **cosine 更适合更长训练**：2M+ 步的 665 实验（EXP-026）才是 cosine 能发挥价值的场景

**结论**：⚠️ EXP-024 eval（84.7%）与 EXP-023f（85.0%）统计持平，**5×5×3 短训练（500K步）维持 constant LR=1e-3 即可**，无需引入 cosine 调度。

---

## EXP-023 超参扫描 —— 5×5×3 新基线（2026-03-21）

7 组 Phase 1 扫描实验（各 500K steps），目标：找到 obs_channels=3 + reward ±0.1 下最快收敛的 n_envs / LR 组合。

**Phase 1 扫描配置与结果**：

| Sub-exp | Run | n_envs | LR | Max SR | Final SR | EV | GPU 训练时间 |
|---------|-----|--------|----|--------|----------|----|-------------|
| 023a | `20260321110321` | 16 | 1e-4 | 37% | 30% | 0.83 | — |
| 023b | `20260321110324` | 32 | 1e-4 | 40% | 38% | 0.83 | — |
| 023c | `20260321111328` | 64 | 1e-4 | 30% | 30% | 0.78 | — |
| 023d | `20260321110320` | 64 | 3e-4 | 74% | 63% | 0.75 | — |
| 023e | `20260321113242` | 128 | 3e-4 | 62% | 62% | 0.69 | 5.3 min |
| **023f** | `20260321115151` | **64** | **1e-3** | **84%** | **81%** | 0.74 | **3.5 min** |
| 023g | `20260321113243` | 128 | 1e-3 | 84% | 63% | 0.69 | 5.3 min |

固定参数（全部）：obs_channels=3，reward ±0.1，n_steps=512，batch_size=128，n_epochs=4，features_dim=128，total_timesteps=500K。

**最优配置**：023f（`exp_023f_553_n64_lr1e3.yaml`），Run `mw_ppo_5x5x3_seed42_20260321115151`。

**核心指标（023f）**：
- TensorBoard：success_rate max **84%**（final 81%），EV 0.741，entropy_loss -1.429
- 训练时间：**3.5 min**（GPU，500K steps @ avg fps≈2951）
- Eval：**85.0%**（300 局 clean eval）

**与 EXP-018 对比**（obs_channels=2，n_envs=16，LR=1e-4，2M steps，85.6% eval）：

| | EXP-018 | **EXP-023f** |
|--|---------|-------------|
| 步数 | 2,015,232 | **524,288（1/4）** |
| GPU 训练时间 | ~16 min | **3.5 min（4.5x 更快）** |
| clean eval | 85.6%（500局）| **85.0%（300局）** |
| EV（最终）| 0.844 | 0.741 |
| entropy（最终）| -0.696 | -1.429 |

**训练曲线特征（023f）**：收敛极快，前 200K 步 success_rate 即过 50%；EV 0.74 低于 EXP-018 的 0.84，entropy -1.43 明显高于 EXP-018 的 -0.70，说明策略未完全收敛——但 eval 已与旧 2M 步基线持平。

**分析**：
1. **LR 是核心驱动**：LR=1e-4 时 n_envs 64/128 均无法超越基线；LR=3e-4 时 n_envs=64 跳至 74%；LR=1e-3 时到达 84%
2. **n_envs 需要配合 LR 调整**：n_envs=64 vs 128 在 LR=1e-3 下，最终精度相同（84%），但 023f（64 envs）final SR 更稳定（81% vs 63%），且速度更快（3.5 vs 5.3 min）；n_envs=128 在 T4 上 subproc 开销导致 FPS 反更低
3. **obs_channels=3 from scratch 完全有效**：EV 能在 500K 步内达到 0.74，early value fitting 无明显障碍
4. **500K 步即已充分**：eval 85% 已与 EXP-018 的 2M 步相当，entropy 虽高但不影响实用性

**结论**：✅ 确立新 5×5×3 基线：**obs_channels=3，n_envs=64，LR=1e-3，500K steps，eval 85%**。相比旧基线步数减少 75%，wallclock 减少约 80%。后续课程学习实验以此配置为起点。

---

## EXP-022 A9 7×7×7 延长训练至 20M 步（2026-03-21）

配置 `experiments/configs/exp_022_a9_extend_7x7x7.yaml`，Run `mw_ppo_7x7x7_seed42_20260320100042_continue_20260320110730_continue_20260321090124`（因 Modal 抢占第一次中断后重启，`_continue_20260321085216` 仅完成 278k 步），7M 额外步数，从 EXP-019b-continue（`mw_ppo_7x7x7_seed42_20260320100042_continue_20260320110730`，13M 步，51.6%）`CONTINUE_FROM` 续训。

**核心变量**（A9 延长训练）：步数 13M → 20M（额外 7M 步），其余配置与 EXP-019b 完全一致（obs_channels=2，reward ±0.1，n_envs=16）。

**指标**：
- TensorBoard（`continue_20260321090124`）：success_rate max **61%**（@3.7M / 5.4M additional steps），final 44%（高方差），EV 0.60-0.67，entropy_loss -0.466（从 -0.534 缓降，未 plateau）
- 训练时间：约 80 分钟（7M 步，fps≈1463）
- Eval（@5.35M additional steps）：**56.0%**（500 局）；@3.7M additional steps：55.2%（500 局）

**训练曲线特征**：success_rate 在 3.5-5.5M additional steps 多次达到 0.59-0.61，但振荡剧烈（标准差约 ±0.05）。后期未见明显单调上升，可能接近短期局部最优，但 entropy 仍在缓慢下降说明未完全收敛。

**分析**：
1. 从 51.6%（13M 步）提升至 56.0%（20M 步），增量约 +4.4%，改善持续但趋于平缓
2. 7×7×7 学习曲线很平坦，方差极大，需要大量步数才能确认趋势
3. Entropy -0.466（vs 开始时 -0.534）还有下降空间，说明策略仍在收敛中
4. EV 0.62 意味着 value function 仍未完美拟合，7×7×7 的随机性导致估值困难

**结论**：✅ 延长训练有效，56% 是目前 7×7×7 的最优 eval；但天花板尚未确认（entropy 未 plateau）。若要确认真实上限，需要再续训 10M+ 步（到 30M 步左右）。当前 56% checkpoint 是 8×8×10 挑战的最强起始点。

---

## EXP-021 B3 n_envs=64 吞吐量验证 + A10 安全通道 —— 6×6×5（2026-03-21）

配置 `experiments/configs/exp_021_b3_n_envs64_6x6x5.yaml`，Run `mw_ppo_6x6x5_seed42_20260321085207`，6M 步，从 EXP-019a 最优 checkpoint（`mw_ppo_6x6x5_seed42_20260320072443_continue_20260320084707` @ step 8,263,504）权重迁移，`TRANSFER_FROM` + `TRANSFER_STEPS=8263504`，Transferred 27 layers，Skipped 3 layers（Conv1 因 obs_channels 2→3 shape mismatch）。

**核心变量**（vs EXP-020）：n_envs 16→64；（vs EXP-019a）：obs_channels 2→3，n_envs 16→64。

**指标**：
- TensorBoard：success_rate final 81% / max 81%（@6M 步），EV 0.62，entropy_loss -0.758，value_loss 0.358
- 训练时间：**57 分钟**（vs EXP-020 的 64 分钟，快约 11%）
- 吞吐量：fps≈5903（rollout 阶段），约为 EXP-020 n_envs=16 的 3-4x
- Eval：final model **73.0%**（500 局）；best checkpoint（@5.998M）71.8%（500 局）——final 即最优

**统计显著性**（与 EXP-019a 76.2% 对比，n=500 两比例 z 检验）：z=1.16，p=0.245，**不显著**。95% CI [69.1%, 76.9%] 与 EXP-019a [72.5%, 79.9%] 高度重叠。

**分析**：n_envs=64 的吞吐量提升显著（rollout fps 约 4x），但最终精度 73% 低于 EXP-020（75.4%）和 EXP-019a（76.2%）基线。原因分析：
1. n_envs=64 每次更新样本 65k（vs n_envs=16 的 16k），更新次数约为 1/4，在相同 wall-clock 时间内梯度更新步数更少
2. 大 batch 在稀疏奖励环境下需要更大 LR 或更长训练才能充分利用样本
3. 精度差异统计不显著（p=0.245），不能排除随机波动

**结论**：⚠️ n_envs=64 在**速度**上有优势（57 min 内达到 73%），但**精度**未超过 n_envs=16 基线，差异不显著。若目标是快速迭代实验，n_envs=64 是合理选择；若追求最终精度，n_envs=16 配合更多步数更稳妥。不推荐单独使用 n_envs=64 作为最终训练配置，建议配合 LR 调整或更多步数。

---

## EXP-020 A10 约束传播安全通道验证 —— 6×6×5（2026-03-21）

配置 `experiments/configs/exp_020_a10_safe_channel_6x6x5.yaml`，Run `mw_ppo_6x6x5_seed42_20260321085208`，6M 步，从 EXP-019a 最优 checkpoint（`mw_ppo_6x6x5_seed42_20260320072443_continue_20260320084707` @ step 8,263,504）权重迁移，`TRANSFER_FROM` + `TRANSFER_STEPS=8263504`，Transferred 27 layers，Skipped 3 layers（Conv1 因 obs_channels 2→3 shape mismatch）。

**核心变量**（vs EXP-019a）：obs_channels 2→3（新增约束传播安全掩码通道），n_envs 不变（16）。

**指标**：
- TensorBoard：success_rate final 65% / max **82%**（@4.55M 步），EV 0.70，entropy_loss -0.536，value_loss 0.303
- 训练时间：**64 分钟**
- Eval（best checkpoint @4.55M）：**75.4%**（500 局）；final model（@6M）仅 65%（下滑）

**训练曲线特征**：约 4.5M 步达到峰值 82%，之后明显下滑至 65%。下滑说明：(1) 模型在后期 overfitting 到某种局部策略；(2) 6M 步对当前配置略多，4-5M 步是最优停止点。

**统计显著性**（best ckpt 75.4% vs EXP-019a 76.2%，n=500 两比例 z 检验）：z=0.30，p=0.768，**不显著**。95% CI [71.6%, 79.2%] 与 EXP-019a [72.5%, 79.9%] 完全重叠。

**分析**：安全通道（obs_channels=3）在 4.55M 步达到 82%（TensorBoard），eval 75.4%，与 EXP-019a（8.6M 步，76.2%）**步数减少近一半**达到同等水平，说明安全通道**加速了早期收敛**。然而：
1. 最终精度差异统计不显著（p=0.768），无法确认通道带来绝对精度提升
2. Conv1 因 shape mismatch 被跳过（obs_channels 2→3），特征提取层实际上近乎重新学习，迁移效益主要来自 Linear+head
3. 训练后期下滑可能与重新初始化的 Conv1 学习不稳定有关

**结论**：⚠️ 安全通道可能**加速收敛**（约 4.5M 步即达同等水平），但最终精度提升不显著。最重要的发现是 **obs_channels=3 配置可行**，CNN 能正确利用第三通道。下一步应在 from-scratch 训练下验证（避免 shape mismatch 导致的 Conv1 重新学习），或在更大棋盘（7×7×7 / 8×8×10）上测试是否有更明显收益。

---

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
