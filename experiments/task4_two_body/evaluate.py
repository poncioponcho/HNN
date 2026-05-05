"""
Task 4: Evaluate HNN vs baseline on two-body problem.

Corresponds to: 论文 Section 4.4, Figure 4.
长期 rollout + 能量守恒评估 + 轨道可视化。

评估重点:
  - 8 维相空间中的长期预测精度
  - 能量守恒性（关键指标）
  - 轨道形状是否保持近圆

可视化:
  - 两个物体的运动轨道 (Figure 4a)
  - 相空间投影 (Figure 4b)
  - 能量 vs 时间 (Figure 4c)

预期结果 (论文):
  - HNN 在高维空间中仍能近似守恒能量
  - Baseline 能量发散更快（高维空间中误差累积更严重）
  - HNN 预测的轨道形状更稳定
"""
pass
