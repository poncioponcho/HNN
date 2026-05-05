"""
Generic training loop for HNN and baseline models.

Corresponds to: 论文 Section 3-5, 所有实验的训练阶段。
使用 Adam 优化器，学习率 1e-3。

训练流程:
  1. 从数据中采样 mini-batch: (x, dx_dt)
     其中 x = (q, p), dx_dt = (dq/dt, dp/dt) 通过有限差分估计
  2. 前向传播:
     - HNN: 计算 H = model(x), 然后通过 autograd 得到 ∂H/∂x
     - Baseline: 直接计算 y_pred = model(x)
  3. 计算损失 (见 losses.py)
  4. 反向传播 + 参数更新

默认超参数 (论文):
  - optimizer: Adam
  - learning_rate: 1e-3
  - batch_size: 64
  - Task 1-3: 训练 ~2000 epochs
  - Task 4: 训练 10k steps
  - Task 5: AE 预训练 5k steps + 联合训练 5k steps

陷阱:
  - HNN 的损失计算需要 create_graph=True，否则无法反向传播
  - 有限差分估计导数会放大噪声，σ=0.1 的噪声可能导致导数估计不准
  - 训练数据需要打乱，避免同一轨迹的连续点在同一个 batch 中
"""
pass
