"""
Analytical Hamiltonian for the ideal mass-spring system.

Corresponds to: 论文 Section 4.1, Task 1.
哈密顿量: H = 0.5 * k * q^2 + p^2 / (2 * m)

输入: q (位移), p (动量)
输出: H (标量能量)

哈密顿方程:
  dq/dt = ∂H/∂p = p / m
  dp/dt = -∂H/∂q = -k * q

参数默认值 (论文):
  k = 1.0 (弹簧常数)
  m = 1.0 (质量)

数据生成:
  - 总能量在 [0.2, 1.0] 范围内均匀采样
  - 25 条轨迹，每条 30 个观测点
  - 添加高斯噪声 σ = 0.1

陷阱:
  - 这是唯一有解析解的线性系统，HNN 和基线差距最小
  - 但 HNN 仍能在长期 rollout 中保持能量守恒
"""
pass
