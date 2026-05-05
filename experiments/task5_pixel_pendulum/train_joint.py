"""
Task 5: Joint training of Autoencoder + HNN.

Corresponds to: 论文 Section 5, Eq. 9, Figure 5.
联合训练自编码器和哈密顿神经网络。

训练流程:
  1. 输入两帧拼接像素 → Encoder → latent z = (z_q, z_p)
  2. 计算 HNN loss:
     - H = hnn(z)
     - dH_dz = autograd.grad(H, z)
     - L_HNN = ||∂H/∂z_p - dz_q/dt||^2 + ||∂H/∂z_q + dz_p/dt||^2
  3. 计算 AE loss:
     - L_AE = MSE(decoder(z), x_input)
  4. 计算 CC loss (辅助):
     - L_CC = ||z_p - (z_q^t - z_q^{t+1})||^2
     - 确保 z_p 编码速度信息
  5. 总损失:
     - L_total = L_AE + L_HNN + λ_CC * L_CC

联合损失 (论文 Eq. 9):
  L_CC = ||z_p - (z_q^t - z_q^{t+1})||^2

训练配置:
  - optimizer: Adam, lr=1e-3
  - 联合训练 5k steps
  - λ_CC = 0.1 (CC loss 权重)

输出: checkpoints/task5_joint.pt

陷阱:
  - z_q 和 z_p 是 latent 的前半和后半，索引不能搞错
  - CC loss 中 z_q^t 和 z_q^{t+1} 来自相邻两帧
  - 联合训练时 AE 和 HNN 的梯度会相互影响
  - 需要同时优化两组参数: AE 参数 + HNN 参数
  - 预训练 AE 有助于联合训练的稳定性
"""
pass
