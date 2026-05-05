"""
Task 1: Train baseline NN on mass-spring data.

Corresponds to: 论文 Section 4.1, Figure 1 (baseline comparison).
训练基线神经网络直接预测 (dq/dt, dp/dt)。

模型架构 (论文):
  - 3 层 MLP, 200 隐单元, tanh 激活
  - 输入: (q, p) → 输出: (dq/dt, dp/dt)
  - 与 HNN 的 MLP 部分完全相同，保证公平比较

训练配置:
  - optimizer: Adam, lr=1e-3
  - loss: MSE loss

输出: checkpoints/task1_baseline.pt

陷阱:
  - 基线网络在短期预测上可能和 HNN 接近
  - 关键差异体现在长期 rollout 的能量守恒性上
"""
pass
