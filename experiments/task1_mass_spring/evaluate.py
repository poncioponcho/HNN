"""
Task 1: Evaluate HNN vs baseline on mass-spring system.

Corresponds to: 论文 Section 4.1, Figure 1.
长期 rollout + 能量守恒评估。

评估流程:
  1. 从测试集选取初始条件
  2. 使用 RK45 积分器 rollout 200 步
     - HNN: 动力学 = (∂H/∂p, -∂H/∂q)
     - Baseline: 动力学 = model(q, p)
  3. 计算评估指标:
     - L2 trajectory loss
     - MSE energy deviation (关键指标)
     - Coordinate MSE
  4. 生成可视化:
     - 相空间轨道 (Figure 1a)
     - 能量 vs 时间 (Figure 1c)

预期结果 (论文):
  - HNN 总能量在 rollout 期间近似恒定
  - Baseline 总能量随时间发散
  - 相空间中 HNN 轨道闭合，Baseline 螺旋

陷阱:
  - 积分器误差容差必须设为 1e-9
  - 能量偏差应归一化: ΔH/H(0)
"""
pass
