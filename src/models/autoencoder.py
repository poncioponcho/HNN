"""
Autoencoder for Pixel Pendulum (Task 5).

Corresponds to: 论文 Section 5, Figure 5.
核心思想: 将像素观测编码到低维 latent 空间，
latent 的前半 z_q 编码广义坐标（角度），后半 z_p 编码广义动量（角速度）。

架构: 4 层全连接 + ReLU 激活 + 残差连接
  - Encoder: 输入 (28*28*2=1568,) → 隐层 200 → latent (2,)
  - Decoder: latent (2,) → 隐层 200 → 输出 (1568,)

输入维度: (batch_size, 1568) — 两帧 28x28 像素拼接 (28*28*2)
  - 陷阱: 必须使用两帧拼接，单帧无法推断角速度
  - 论文原文: "we concatenate two adjacent frames along the channel dimension"

输出维度: (batch_size, 1568) — 重建的两帧像素

Latent 维度约束:
  - 必须为偶数，前一半是 z_q，后一半是 z_p
  - Task 5 中 latent_dim = 2，即 z_q 和 z_p 各 1 维

残差连接:
  - 论文提到使用残差连接提升训练稳定性
  - 实现: output = F.relu(x + linear(x))
"""
pass
