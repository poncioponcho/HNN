"""
Task 1: Generate mass-spring training data.

Corresponds to: 论文 Section 4.1, Task 1.
生成含噪的质量-弹簧系统轨迹数据。

数据生成流程:
  1. 从能量范围 [0.2, 1.0] 均匀采样总能量 E
  2. 根据 E 随机生成初始条件 (q0, p0)
     - q0 = sqrt(2*E/k) * random_sign
     - p0 = sqrt(2*m*E) * random_sign (确保 H(q0,p0) ≈ E)
  3. 使用 RK45 积分器生成轨迹
  4. 添加高斯噪声 N(0, σ²), σ=0.1
  5. 通过有限差分估计 (dq/dt, dp/dt) 作为训练标签

输出: processed/mass_spring_data.pt
  - x: (N, 2) 相空间坐标
  - dx_dt: (N, 2) 时间导数
  - trajectories: list of arrays, 原始轨迹

参数 (论文):
  - k=1, m=1, noise=0.1, 25 trajectories, 30 observations per trajectory

陷阱:
  - 有限差分 (x_{t+1} - x_t) / dt 会放大噪声
  - 初始条件必须满足 H(q0, p0) = E，否则能量标签不准确
"""
pass
