"""
Task 4: Generate two-body problem training data.

Corresponds to: 论文 Section 4.4, Task 4.
生成近圆轨道的双体问题轨迹数据。

数据生成流程:
  1. 在半径 [0.5, 1.5] 范围内均匀采样轨道半径 r
  2. 计算近圆轨道初始条件:
     - q1 = [r, 0], q2 = [-r, 0] (对称放置)
     - v = sqrt(g * M / r), M = m1 + m2
     - p1 = [0, m1*v], p2 = [0, -m2*v]
  3. 对速度添加高斯噪声 N(0, σ²), σ=0.05
  4. 使用 RK45 积分器生成轨迹
  5. 通过有限差分估计时间导数
  6. 80/20 划分训练/测试集

输出: processed/two_body_data.pt

参数 (论文):
  - m1=m2=g=1, 1000 trajectories, 50 observations, noise σ=0.05

陷阱:
  - 8 维相空间是论文中最高维的测试
  - 初始速度必须精确计算 v=sqrt(g*M/r)，否则轨道不是近圆的
  - 噪声 σ=0.05 相对较小，但足以使轨道偏离圆形
  - 双体问题有 3 个守恒量（能量、角动量、质心动量），HNN 只约束能量
"""
pass
