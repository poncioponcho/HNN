"""
Loss functions for HNN, baseline, and Autoencoder models.

Corresponds to: 论文 Section 3 (Eq. 5-6) 和 Section 5 (Eq. 9).

1. HNN Loss (论文 Eq. 5):
   L_HNN = ||∂H_θ/∂p - dq/dt||^2 + ||∂H_θ/∂q + dp/dt||^2

   输入:
     - model: HNN 模型
     - x: (batch_size, 2n) 相空间坐标, requires_grad=True
     - dx_dt: (batch_size, 2n) 时间导数标签
   输出:
     - loss: 标量

   计算步骤:
     H = model(x)                                    # (batch, 1)
     dH_dx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]  # (batch, 2n)
     n = x.shape[1] // 2
     dq_dt_pred = dH_dx[:, n:]                       # ∂H/∂p
     dp_dt_pred = -dH_dx[:, :n]                      # -∂H/∂q
     loss = MSE(dq_dt_pred, dx_dt[:, n:]) + MSE(dp_dt_pred, dx_dt[:, :n])

2. Baseline Loss:
   L_baseline = ||f_θ(q,p) - (dq/dt, dp/dt)||^2

3. AE Loss (Task 5):
   L_AE = MSE(x_reconstructed, x_input)

4. CC Loss (论文 Eq. 9, Task 5):
   L_CC = ||z_p - (z_q^t - z_q^{t+1})||^2

   陷阱:
     - z_q 和 z_p 是 latent 的前半和后半
     - z_q^t 和 z_q^{t+1} 是相邻两帧的 z_q
     - CC loss 确保 z_p 编码速度信息

5. Joint Loss (Task 5):
   L_total = L_AE + L_HNN + λ_CC * L_CC

关键提示:
  - HNN loss 中 ∂H/∂q 前面有负号 (哈密顿方程 dp/dt = -∂H/∂q)
  - 符号错误是最常见的实现 bug
  - create_graph=True 是必须的，否则无法对 loss 做反向传播
"""
pass
