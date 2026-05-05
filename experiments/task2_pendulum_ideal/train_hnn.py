"""
Task 2: Train HNN on ideal pendulum data.

Corresponds to: 论文 Section 4.2, Figure 2.
训练哈密顿神经网络学习理想单摆动力学。

模型架构 (论文):
  - 3 层 MLP, 200 隐单元, tanh 激活
  - 输入: (q, p) → 输出: 标量 H

训练配置:
  - optimizer: Adam, lr=1e-3
  - 从 configs/pendulum_ideal.yaml 读取超参数

输出: checkpoints/task2_hnn.pt

陷阱:
  - 单摆是非线性系统，HNN 优势更明显
  - tanh 激活对周期性函数拟合效果优于 ReLU
"""
pass
