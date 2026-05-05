"""
Task 2: Generate ideal pendulum training data.

Corresponds to: 论文 Section 4.2, Task 2.
生成含噪的理想单摆系统轨迹数据。

数据生成流程:
  1. 从能量范围 [1.3, 2.3] 均匀采样总能量 E
  2. 根据 E 和 q0=0 生成初始条件 (q0, p0)
     - p0 = sqrt(2*m*(E - 2*m*g*l*(1-cos(q0)))) / l
  3. 使用 RK45 积分器生成轨迹
  4. 添加高斯噪声 N(0, σ²), σ=0.1
  5. 通过有限差分估计 (dq/dt, dp/dt)

输出: processed/pendulum_ideal_data.pt

参数 (论文):
  - m=l=1, g=3, energy [1.3, 2.3], 25 trajectories, 30 observations

陷阱:
  - g=3 是论文使用的非标准值，不是 9.81
  - 能量范围 [1.3, 2.3] 覆盖大角度运动
  - 当 E 接近 2*m*g*l=6 时，摆接近倒立点，数值不稳定
"""
pass
