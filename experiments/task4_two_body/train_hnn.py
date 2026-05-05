"""
Task 4: Train HNN on two-body problem data.

Corresponds to: 论文 Section 4.4, Figure 4.
训练 HNN 学习双体问题动力学。

模型架构 (论文):
  - 3 层 MLP, 200 隐单元, tanh 激活
  - 输入: (q1x,q1y,q2x,q2y,p1x,p1y,p2x,p2y) → 输出: 标量 H

训练配置:
  - optimizer: Adam, lr=1e-3
  - 训练 10k steps (论文指定)
  - 从 configs/two_body.yaml 读取超参数

输出: checkpoints/task4_hnn.pt

陷阱:
  - 8 维输入比 Task 1-2 的 2 维复杂得多
  - 需要更多训练数据 (1000 条轨迹) 和更长训练 (10k steps)
  - HNN 的 ∂H/∂x 梯度计算在 8 维下更耗时
"""
pass
