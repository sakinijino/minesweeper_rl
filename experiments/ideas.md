# 优化方向 Backlog

## 待探索

- [x] 更长训练（500k / 1M 步）确认收敛上限 → EXP-002
- [x] 续训到 2M 步确认是否 plateau → EXP-003
- [ ] 多通道观察表示（分离未揭开/数字/地雷计数信道）
- [ ] 邻居计算改为 scipy 卷积（性能优化）
- [ ] 课程学习（从小棋盘到大棋盘，或从少地雷到多地雷）
- [ ] 调大 n_envs（目前 8，T4 应该可以跑 16-32）
- [ ] 学习率 schedule（线性衰减）
- [ ] 更大网络（features_dim 256，更深 pi/vf layers）
- [ ] entropy coefficient 衰减（初期高探索 → 后期低熵）
- [x] reward shaping 优化（reward_win 0.2→1.0，win 信号 8%→31%）→ EXP-004（@1M 步 75%，验证有效）
- [x] 续训 EXP-004 到 2M 步，获得 "new reward @2M" 基准 → EXP-005（进行中）
- [ ] 多 seed 训练验证结果稳定性

## 已完成

- [x] EXP-001：基线验证，100k 步，确认训练流程可行
