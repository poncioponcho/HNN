"""
Evaluation metrics for HNN experiments.

Corresponds to: 论文 Section 4-5, 所有实验的评估指标。

核心指标:
  1. L2 trajectory loss: ||x_pred - x_true||_2
     预测轨迹与真实轨迹的 L2 距离

  2. MSE energy deviation: MSE(H_pred(t), H_pred(0))
     预测轨迹的总能量随时间的均方偏差
     - HNN 应接近 0（能量守恒）
     - Baseline 会随时间增长（能量发散）

  3. Coordinate MSE: MSE(q_pred, q_true) + MSE(p_pred, p_true)
     各坐标分量的均方误差

输入:
  - trajectory_pred: (T, 2n) 预测轨迹
  - trajectory_true: (T, 2n) 真实轨迹
  - hamiltonian_fn: callable(x) -> H, 哈密顿量函数
输出:
  - metrics: dict, 包含各指标值

陷阱:
  - 能量偏差应相对于初始能量归一化: ΔH / H(0)
  - L2 loss 在长时间后会饱和（两个轨道可能趋向不同区域）
  - 论文主要关注能量守恒性，而非轨迹预测精度
"""
pass
