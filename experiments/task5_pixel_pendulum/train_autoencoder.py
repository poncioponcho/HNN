"""
Task 5: Pretrain Autoencoder on pixel pendulum data.

Corresponds to: 论文 Section 5, Figure 5.
预训练自编码器将像素映射到低维 latent 空间。

训练流程:
  1. 输入: 两帧拼接的像素 (28*28*2=1568,)
  2. Encoder 编码到 latent (2,)
     - latent 前 1 维 = z_q (角度)
     - latent 后 1 维 = z_p (角速度)
  3. Decoder 重建像素 (1568,)
  4. Loss: MSE(x_reconstructed, x_input)

模型架构 (论文):
  - 4 层全连接, 200 隐单元, ReLU 激活 + 残差连接
  - latent_dim = 2 (必须为偶数)

训练配置:
  - optimizer: Adam, lr=1e-3
  - 预训练 5k steps

输出: checkpoints/task5_autoencoder.pt

陷阱:
  - 预训练阶段不训练 HNN，只训练 AE
  - latent 维度必须为偶数，前半 z_q 后半 z_p
  - 残差连接: output = relu(x + linear(x))
  - 输入需要展平: (batch, 2, 28, 28) → (batch, 1568)
"""
pass
