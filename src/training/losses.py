"""
Loss functions for HNN, baseline, and Autoencoder models.

Corresponds to: 论文 Section 3 (Eq. 5-6) 和 Section 5 (Eq. 9)。

核心思想:
  - HNN 的损失不是直接预测 H 本身，而是匹配 H 的偏导数与真实轨迹
  - 这种设计让网络学习「正确的能量函数结构」，而非记忆具体数值

数学原理:

1. HNN Loss (论文 Eq. 5):
   L_HNN = ||∂H_θ/∂p - dq/dt||² + ||-∂H_θ/∂q - dp/dt||²

   为什么这样设计？
     - 我们无法直接观测哈密顿量 H（它不是物理量）
     - 但我们可以观测 (q, p) 和它们的时间导数 (dq/dt, dp/dt)
     - 通过让 ∂H/∂p 匹配 dq/dt，间接学习 H 的结构
     - 这等价于学习满足哈密顿方程的能量函数

   维度说明:
     - coords: (batch, 2n)，前 n 列是 q，后 n 列是 p
     - dcoords_dt: (batch, 2n)，前 n 列是 dq/dt，后 n 列是 dp/dt
     - loss: 标量（对所有样本和维度求平均）

2. Baseline Loss:
   L_baseline = ||f_θ(q,p) - (dq/dt, dp/dt)||²

   直接回归向量场，无物理约束

3. AE Loss (Task 5):
   L_AE = MSE(x_reconstructed, x_input)

4. CC Loss (论文 Eq. 9, Task 5):
   L_CC = ||z_p - (z_q^t - z_q^{t+1})||²

关键提示:
  - ⚠️ HNN loss 中 dp/dt 前面有负号！（哈密顿方程 dp/dt = -∂H/∂q）
  - create_graph=True 是必须的，否则无法对 loss 做反向传播
  - 符号错误是最常见的 bug，会导致训练不收敛或学出错误的动力学
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def hnn_loss(
        model: nn.Module,
        coords: torch.Tensor,
        dcoords_dt: torch.Tensor
) -> torch.Tensor:
    """
    Hamiltonian Neural Network 损失函数

    这是整个 HNN 训练的核心！通过匹配哈密顿方程的右端项来学习能量函数。

    数学公式（论文 Eq. 5）:
        L = MSE(∂H_θ/∂p, dq/dt_true) + MSE(-∂H_θ/∂q, dp/dt_true)

    物理意义:
        第一项: 确保模型学习的 H 对 p 的偏导数等于真实的坐标速度
        第二项: 确保模型学习的 H 对 q 的偏导数的负值等于真实的动量变化率

    为什么不直接用 MSE(H_pred, H_true)？
      - 因为真实世界中我们无法直接测量哈密顿量 H
      - 我们只能测量位置 q、动量 p 以及它们的速度 (dq/dt, dp/dt)
      - 所以只能通过「间接」方式学习 H

    Args:
        model: HNN 模型实例（继承自 nn.Module）
               必须有 forward(coords) -> H 的接口
        coords: 相空间坐标张量，shape (batch_size, 2n)
                格式: [q₁, q₂, ..., qₙ, p₁, p₂, ..., pₙ]
                requires_grad=True（用于自动微分）
                例如质量弹簧系统: [[q₁, p₁], [q₂, p₂], ...]
        dcoords_dt: 真实的时间导数标签，shape (batch_size, 2n)
                    格式: [dq₁/dt, dq₂/dt, ..., dqₙ/dt, dp₁/dt, dp₂/dt, ..., dpₙ/dt]
                    通常通过有限差分估计: (x_{t+1} - x_t) / Δt

    Returns:
        loss: 标量损失值（MSE，已对 batch 和维度取平均）
              范围: [0, +∞)，越小表示模型越准确

    Example（简谐振子）:
        >>> model = HNN(input_dim=2)
        >>> coords = torch.tensor([[1.0, 0.5]], requires_grad=True)
        >>> dcoords_dt = torch.tensor([[0.5, -1.0]])  # 解析解的导数
        >>> loss = hnn_loss(model, coords, dcoords_dt)
        >>> loss.backward()  # 可以正常反向传播

    实现细节（逐行解释）:
    """
    # 确保 coords 可以计算梯度（这是 autograd 的前提）
    # 如果 coords 已经 requires_grad=True，这行不会改变任何东西
    coords = coords.clone().detach().requires_grad_(True)

    # Step 1: 通过模型计算当前状态的哈密顿量
    # H 是标量场：每个样本对应一个能量值
    # shape: (batch_size, 1)
    H = model(coords)

    # Step 2: 使用自动微分计算 H 对输入坐标的梯度
    # 这是 HNN 最核心的一步！实现了 ∇H = (∂H/∂q, ∂H/∂p)
    #
    # 参数解释:
    #   outputs=H: 要对其求导的张量（哈密顿量）
    #   inputs=coords: 求自变量（相空间坐标）
    #   grad_outputs=torch.ones_like(H):
    #       表示我们对 H 的每个元素都求偏导
    #       因为 H 是 (batch, 1)，所以这个也是 (batch, 1)
    #   create_graph=True:
    #       ⚠️ 关键参数！保留计算图以支持二阶微分
    #       训练时需要对 loss 再次求导（反向传播）
    #       如果不设置，二阶导数会丢失，导致无法更新模型参数
    #   retain_graph=True:
    #       保留计算图以便后续可能再次使用（可选优化）
    #
    # 返回值: 元组，[0] 取出梯度张量
    # shape: (batch_size, 2n) — 与 coords 同形状
    dH_dcoords = torch.autograd.grad(
        outputs=H,
        inputs=coords,
        grad_outputs=torch.ones_like(H),
        create_graph=True,
        retain_graph=True
    )[0]

    # Step 3: 将梯度拆分为坐标部分和动量部分
    # coords 的格式是 [q₁,...,qₙ, p₁,...,pₙ]
    # 所以梯度的前 n 列是 ∂H/∂q，后 n 列是 ∂H/∂p
    n_coords = coords.shape[1] // 2  # 自由度数量

    dH_dq = dH_dcoords[:, :n_coords]  # shape: (batch, n) — ∂H/∂q
    dH_dp = dH_dcoords[:, n_coords:]  # shape: (batch, n) — ∂H/∂p

    # Step 4: 应用哈密顿方程得到模型预测的时间导数
    # 论文 Eq. 6a: dq/dt =  ∂H/∂p （正则方程第一式）
    # 论文 Eq. 6b: dp/dt = -∂H/∂q （正则方程第二式，注意负号！）
    #
    # ⚠️⚠️⚠️ 第二式的负号是最常见的 bug 来源！！！⚠️⚠️⚠️
    # 如果漏掉负号，模型会学到错误的动力学（能量不守恒方向反了）
    dqdt_pred = dH_dp        # shape: (batch, n) — 预测的坐标速度
    dpdt_pred = -dH_dq       # shape: (batch, n) — 预测的动量变化率

    # Step 5: 从真实标签中提取对应的分量
    # dcoords_dt 的格式也是 [dq/dt, ..., dq/dt, dp/dt, ..., dp/dt]
    dqdt_true = dcoords_dt[:, :n_coords]  # shape: (batch, n) — 真实坐标速度
    dpdt_true = dcoords_dt[:, n_coords:]  # shape: (batch, n) — 真实动量变化率

    # Step 6: 计算 MSE loss（两部分的和）
    # 第一项: 匹配 dq/dt（坐标演化）
    # 第二项: 匹配 dp/dt（动量演化）
    #
    # 使用 F.mse_loss 自动对 batch 和维度取平均
    loss_q = F.mse_loss(dqdt_pred, dqdt_true)  # 坐标方向的误差
    loss_p = F.mse_loss(dpdt_pred, dpdt_true)  # 动量方向的误差

    total_loss = loss_q + loss_p  # 总损失（两项同等重要）

    return total_loss


def baseline_loss(
        model: nn.Module,
        coords: torch.Tensor,
        dcoords_dt: torch.Tensor
) -> torch.Tensor:
    """
    Baseline Neural Network 损失函数

    与 hnn_loss() 形成对比实验！

    数学公式:
        L_baseline = MSE(f_θ(q,p), (dq/dt, dp/dt))

    关键区别:
      - BaselineNN 直接输出向量场 f(q,p) → (dq/dt, dp/dt)
      - 不需要 autograd，因为模型直接预测导数
      - 无物理约束，纯数据驱动

    为什么需要这个 loss？
      - 作为对照组，证明 HNN 的物理先验是有价值的
      - 面试叙事：「Baseline 在训练集上 loss 可能更低，
         但长期 rollout 能量发散」

    Args:
        model: BaselineNN 模型实例
        coords: 相空间坐标，shape (batch, 2n)
        dcoords_dt: 真实时间导数，shape (batch, 2n)

    Returns:
        loss: 标量 MSE 损失

    Example:
        >>> model = BaselineNN(input_dim=2)
        >>> coords = torch.randn(10, 2)
        >>> dcoords_dt = torch.randn(10, 2)
        >>> loss = baseline_loss(model, coords, dcoords_dt)
    """
    # 直接前向传播得到预测的导数
    dcoords_pred = model(coords)  # shape: (batch, 2n)

    # 计算 MSE（简单直接，无 autograd）
    loss = F.mse_loss(dcoords_pred, dcoords_dt)

    return loss


def autoencoder_loss(
        reconstructed: torch.Tensor,
        original: torch.Tensor
) -> torch.Tensor:
    """
    Autoencoder 重建损失（Task 5）

    数学公式:
        L_AE = MSE(x_reconstructed, x_original)

    用于预训练像素→latent 的编码器。

    Args:
        reconstructed: 重建的像素数据，shape (batch, ...)
        original: 原始像素数据，shape 与 reconstructed 相同

    Returns:
        loss: 标量 MSE 损失
    """
    loss = F.mse_loss(reconstructed, original)
    return loss


def canonical_coordinates_loss(
        z_latent_current: torch.Tensor,
        z_latent_next: torch.Tensor
) -> torch.Tensor:
    """
    Canonical Coordinates (CC) 辅助损失（Task 5, 论文 Eq. 9）

    数学公式（论文 Eq. 9）:
        L_CC = ||z_p - (z_q^t - z_q^{t+1})||²

    物理意义:
      - z_latent 被拆分为前半部分 z_q（广义坐标）和后半部分 z_p（广义动量）
      - 对于相邻两帧 t 和 t+1，坐标的变化量 Δz_q ≈ z_p * Δt
      - 这个 loss 强制 z_p 编码速度信息（即广义动量）

    为什么需要 CC loss？
      - 单帧像素无法观测速度（只有位置信息）
      - 但哈密顿力学需要同时知道 q 和 p
      - CC loss 利用时间序列信息让 AE 学习到动量表示

    Args:
        z_latent_current: 当前帧的 latent 表示，shape (batch, 2d)
                          前 d 维是 z_q^t，后 d 维是 z_p^t
        z_latent_next: 下一帧的 latent 表示，shape (batch, 2d)
                        前 d 维是 z_q^{t+1}

    Returns:
        loss: 标量 MSE 损失

    实现细节:
    """
    latent_dim = z_latent_current.shape[1]  # 必须是偶数
    assert latent_dim % 2 == 0, \
        f"latent_dim 必须是偶数（分为 z_q 和 z_p），当前值: {latent_dim}"

    d = latent_dim // 2  # 每个 sub-space 的维度

    # 拆分 latent 为坐标和动量部分
    z_q_current = z_latent_current[:, :d]   # z_q^t: 当前帧的坐标
    z_p_current = z_latent_current[:, d:]   # z_p^t: 当前帧的动量
    z_q_next = z_latent_next[:, :d]         # z_q^{t+1}: 下一帧的坐标

    # 计算 CC loss: 让 z_p 近似等于坐标变化量
    # 物理上: Δq ≈ p * Δt（当 Δt 很小时）
    delta_z_q = z_q_next - z_q_current  # 坐标变化量
    loss = F.mse_loss(z_p_current, delta_z_q)

    return loss


def joint_hnn_ae_loss(
        ae_model: nn.Module,
        hnn_model: nn.Module,
        x_pixel_current: torch.Tensor,
        x_pixel_next: torch.Tensor,
        lambda_cc: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    联合损失函数（Task 5: Pixel Pendulum）

    将 AE、HNN 和 CC 三个 loss 组合为统一的目标函数。

    数学公式（论文 Section 5）:
        L_total = L_AE + L_HNN + λ_CC · L_CC

    Args:
        ae_model: Autoencoder 模型
        hnn_model: HNN 模型（在 latent space 上操作）
        x_pixel_current: 当前帧像素，shape (batch, C, H, W)
        x_pixel_next: 下一帧像素，shape (batch, C, H, W)
        lambda_cc: CC loss 的权重系数，默认 1.0

    Returns:
        total_loss: 联合标量损失
        loss_dict: 各个分量的字典 {'ae': ..., 'hnn': ..., 'cc': ...}
                   用于日志记录和分析
    """
    # Step 1: 编码两帧像素到 latent space
    z_current = ae_model.encode(x_pixel_current)  # (batch, 2d)
    z_next = ae_model.encode(x_pixel_next)        # (batch, 2d)

    # Step 2: 重建当前帧（AE loss）
    x_reconstructed = ae_model.decode(z_current)
    loss_ae = autoencoder_loss(x_reconstructed, x_pixel_current)

    # Step 3: 在 latent space 上计算 HNN loss
    # 需要 latent 的时间导数（通过差分估计）
    dz_dt = (z_next - z_current) / 1.0  # 假设 Δt=1
    loss_hnn = hnn_loss(hnn_model, z_current, dz_dt)

    # Step 4: CC loss（确保 z_p 编码动量）
    loss_cc = canonical_coordinates_loss(z_current, z_next)

    # Step 5: 加权组合
    total_loss = loss_ae + loss_hnn + lambda_cc * loss_cc

    # 返回总损失和各个分量（用于监控训练过程）
    loss_dict = {
        'total': total_loss.item(),
        'ae': loss_ae.item(),
        'hnn': loss_hnn.item(),
        'cc': loss_cc.item()
    }

    return total_loss, loss_dict
