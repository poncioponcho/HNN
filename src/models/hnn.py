"""
Hamiltonian Neural Network (HNN) model.

Corresponds to: 论文 Section 3, Eq. 5-6.
核心思想: 用 MLP 参数化哈密顿量 H(q,p)，输出标量，
通过 torch.autograd.grad 计算 ∂H/∂q 和 ∂H/∂p 得到哈密顿方程右端项。

输入维度: (batch_size, 2n)，其中 n 为自由度数
  - Task 1-2: input_dim = 2 (q, p)
  - Task 4: input_dim = 8 (q1x,q1y,q2x,q2y,p1x,p1y,p2x,p2y)
  - Task 5: input_dim = latent_dim (偶数)

输出维度: (batch_size, 1) — 标量哈密顿量 H

关键实现提示:
  - forward() 只返回标量 H，不计算梯度
  - 梯度计算在 loss 函数中通过 torch.autograd.grad 完成:
      dH_dx = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
      dq_dt = dH_dx[:, n:]      # ∂H/∂p
      dp_dt = -dH_dx[:, :n]     # -∂H/∂q
  - 必须设置 create_graph=True 以支持二阶反向传播
  - 陷阱: HNN 无法建模耗散/摩擦系统，因为哈密顿结构假设能量守恒
"""

import torch
import torch.nn as nn
from typing import Optional


class HNN(nn.Module):
    """
    哈密顿神经网络：学习保守系统的能量函数 H(q, p)

    数学基础（论文 Eq. 5-6）:
        dq/dt =  ∂H/∂p   （广义坐标的时间演化）
        dp/dt = -∂H/∂q   （广义动量的时间演化）

    与 BaselineNN 的本质区别：
        - BaselineNN 直接拟合向量场 f(q,p) → (dq/dt, dp/dt)
        - HNN 先学习标量场 H(q,p)，再通过 autograd 自动得到梯度
        - 这种「先学能量再求导」的结构天然保证辛对称性（近似）
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 200,
          num_hidden_layers: int = 3,
            nonlinearity: str = 'tanh'
    ):
        """
        初始化 HNN 模型

        Args:
            input_dim: 输入维度，必须是 2n（n 个广义坐标 + n 个广义动量）
                       例如：质量弹簧系统 input_dim=2 (q, p)
                             双体问题 input_dim=8 (4个坐标 + 4个动量)
            hidden_dim: 隐藏层宽度，推荐 200-500
                        论文默认使用 200，更大网络可能过拟合小数据集
            num_hidden_layers: 隐藏层数量，推荐 3-4 层
                               太浅无法拟合复杂哈密顿量，太深容易训练不稳定
            nonlinearity: 激活函数，推荐 tanh（光滑、有界、处处可微）
                          避免使用 ReLU（在 H 处不可微会导致 ∂H/∂q 不连续）
        """
        super(HNN, self).__init__()

        self.input_dim = input_dim
        self.n_coords = input_dim // 2  # 广义坐标数量 n

        assert input_dim % 2 == 0, \
            f"input_dim 必须是偶数（q 和 p 各 n 维），当前值: {input_dim}"

        # 选择激活函数
        if nonlinearity == 'tanh':
            activation = nn.Tanh
        elif nonlinearity == 'relu':
            activation = nn.ReLU
        else:
            raise ValueError(f"不支持的激活函数: {nonlinearity}")

        # 构建 MLP 网络：从 (q, p) 映射到标量 H
        # 网络结构: input_dim → [hidden_dim × num_hidden_layers] → 1
        layers = []

        # 第一层：输入层 → 第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # 中间隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # 输出层：最后一个隐藏层 → 标量输出（无激活函数！）
        # 注意：H 可以取任意实数值（动能+势能），不需要限制范围
        layers.append(nn.Linear(hidden_dim, 1))

        # 将所有层封装为 Sequential
        self.net = nn.Sequential(*layers)

        # 初始化权重（重要：影响训练收敛速度）
        self._init_weights()

    def _init_weights(self):
        """
        使用 Xavier 初始化权重

        原理：对于 tanh 激活函数，Xavier 初始化（对称激活函数）保证前向传播的方差稳定
              这对 HNN 很关键，因为我们需要通过多层反向传播计算 ∂H/∂q
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算哈密顿量 H(q, p)

        数学公式（论文 Eq. 5）:
            H = MLP(q, p)

        Args:
            coords: 相空间坐标张量，shape (batch_size, 2n)
                   前 n 列是广义坐标 q = [q1, q2, ..., qn]
                   后 n 列是广义动量 p = [p1, p2, ..., pn]

        Returns:
            H: 哈密顿量，shape (batch_size, 1)
               物理意义：系统的总能量（动能 + 势能）

        示例（质量弹簧系统）:
            输入: coords = [[1.0, 0.5]]  # q=1.0, p=0.5
            输出: H = [[2.25]]           # E = p²/(2m) + kq²/2 ≈ 2.25
        """
        return self.net(coords)

    @staticmethod
    def dynamics(
            t: torch.Tensor,
            coords: torch.Tensor,
            model: 'HNN'
    ) -> torch.Tensor:
        """
        计算哈密顿方程的右端项（时间导数）

        这是 HNN 最核心的函数！将学习的标量 H 转化为动力学方程。

        数学原理（论文 Eq. 6）:
            dq/dt =  ∂H/∂p   （Eq. 6a: 坐标演化方程）
            dp/dt = -∂H/∂q   （Eq. 6b: 动量演化方程）

        为什么用 torch.autograd.grad 而不是 backward()？
            - grad() 可以精确控制对哪个变量求导
            - 返回的是梯度张量，可以直接用于后续计算
            - create_graph=True 支持高阶格式: shape (batch_size, 2n)
                    

        Args:
        t: 标量时间，shape ()。保留接口兼容 torchdiffeq.odeint 的 func(t, y) 签名
           在自治系统中 H 不显含 t，故 t 不参与计算
        coords: 相空间状态，shape (batch_size, 2n)
                格式: [q1, q2, ..., qn, p1, p2, ..., pn]
        model: HNN 实例，forward 输出标量 H(coords)

        Returns:
            dcoords_dt: 时间导数，shape (batch_size, 2n)
                       格式: [dq1/dt, dq2/dt, ..., dqn/dt, dp1/dt, dp2/dt, ..., dpn/dt]

        实现细节:
            1. 计算 H = model(coords)  →  shape (batch, 1)
            2. 对 coords 求 ∂H/∂coords   →  shape (batch, 2n)
            3. 拆分梯度为 ∂H/∂q 和 ∂H/∂p
            4. 按照哈密顿方程组合成 dq/dt 和 dp/dt
        """
        # 单独这一行就是 HNN 的精髓：学习能量函数
        coords.requires_grad_(True)  # 确保可以计算梯度

        # Step 1: 通过 MLP 计算当前状态的哈密顿量
        # H 是标量场，每个样本对应一个能量值
        H = model(coords)  # shape: (batch, 1)

        # Step 2: 使用自动微分计算 H 对 coords 的梯度
        # 这一步实现了数学上的 ∇H = (∂H/∂q, ∂H/∂p)
        # create_graph=True: 保留计算图以支持二阶导数（训练时需要）
        # grad_outputs=torch.ones_like(H): 表示我们对 H 的每个元素求偏导
        dH_dcoords = torch.autograd.grad(
            outputs=H,
            inputs=coords,
            grad_outputs=torch.ones_like(H),
            create_graph=True,
            retain_graph=True
        )[0]  # shape: (batch, 2n)

        # Step 3: 将梯度拆分为坐标部分和动量部分
        # coords 的格式是 [q1, ..., qn, p1, ..., pn]
        # 所以梯度的前 n 列是 ∂H/∂q，后 n 列是 ∂H/∂p
        dH_dq = dH_dcoords[:, :model.n_coords]  # shape: (batch, n)
        dH_dp = dH_dcoords[:, model.n_coords:]  # shape: (batch, n)

        # Step 4: 应用哈密顿方程（论文 Eq. 6a, 6b）
        # 这是整个 HNN 的物理核心！
        # dq/dt =  ∂H/∂p  （正则方程的第一式）
        # dp/dt = -∂H/∂q  （正则方程的第二式，注意负号！）
        dqdt = dH_dp       # shape: (batch, n)
        dpdt = -dH_dq      # shape: (batch, n)

        # Step 5: 重新拼接为完整的状态导数向量
        # 格式与输入一致: [dq1/dt, ..., dqn/dt, dp1/dt, ..., dpn/dt]
        dcoords_dt = torch.cat([dqdt, dpdt], dim=1)  # shape: (batch, 2n)

        return dcoords_dt

    def compute_time_derivative(self, coords: torch.Tensor) -> torch.Tensor:
        """
        便捷方法：给定坐标，直接计算时间导数

        封装 dynamics() 方法，提供更简洁的接口

        Args:
            coords: 相空间坐标，shape (batch, 2n)

        Returns:
            dcoords_dt: 时间导数，shape (batch, 2n)
        """
        # 创建虚拟时间张量（dynamics 函数需要但不使用）
        dummy_t = torch.zeros(coords.shape[0], device=coords.device)
        return self.dynamics(dummy_t, coords, self)
