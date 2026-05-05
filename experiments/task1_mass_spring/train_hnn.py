"""
Task 1: Train HNN on mass-spring data.

Corresponds to: 论文 Section 4.1, Figure 1.
训练哈密顿神经网络学习质量-弹簧系统动力学。

模型架构 (论文):
  - 3 层 MLP, 200 隐单元, tanh 激活
  - 输入: (q, p) → 输出: 标量 H

训练配置:
  - optimizer: Adam, lr=1e-3
  - loss: HNN loss (见 src/training/losses.py)
  - 从 configs/mass_spring.yaml 读取超参数

输出: checkpoints/task1_hnn.pt

陷阱:
  - 输入 x 必须设置 requires_grad=True
  - HNN loss 中 ∂H/∂q 前有负号
  - 训练数据需打乱，避免同一轨迹连续点在同一个 batch
"""
pass
