# 优化方向 Backlog

## 待探索

- [x] 多通道观察表示（分离未揭开/数字/地雷计数信道）→ EXP-007（86%，胜率+，但 explained_var 未突破）
- [ ] 邻居计算改为 scipy 卷积（性能优化）
- [ ] 课程学习（从小棋盘到大棋盘，或从少地雷到多地雷）
- [ ] 调大 n_envs（目前 8，T4 应该可以跑 16-32）
- [ ] 学习率 schedule（线性衰减）
- [x] 更大网络（features_dim 256）→ EXP-008（86%，explained_var max 0.47，容量非瓶颈，与 EXP-007 持平）
- [ ] 更深 pi/vf 层（[128,128] 替代 [64,64]）← 下一个候选，独立于 features_dim 验证
- [ ] 延长训练（续训 EXP-008 到 4M 步），验证能否突破 86% eval 天花板
- [ ] entropy coefficient 衰减（初期高探索 → 后期低熵）
- [x] 修复 vf_coef（0.5 → 1.0）+ 新 reward 从头跑 2M 步 → EXP-006（80%，vf_coef 非瓶颈）
- [ ] 多 seed 训练验证结果稳定性

## 已完成

- [x] EXP-001：基线验证，100k 步，确认训练流程可行
- [x] 更长训练（500k / 1M 步）确认收敛上限 → EXP-002
- [x] 续训到 2M 步确认是否 plateau → EXP-003
- [x] reward shaping 优化（reward_win 0.2→1.0，win 信号 8%→31%）→ EXP-004（@1M 步 75%，验证有效）
- [x] 续训 EXP-004 到 2M 步，获得 "new reward @2M" 基准 → EXP-005（84%，explained_var 卡在 0.44）
