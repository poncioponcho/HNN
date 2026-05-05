"""
Baseline Neural Network for dynamics prediction.

Corresponds to: 论文 Section 3, 作为 HNN 的对照基线。
核心思想: 标准 MLP 直接预测 (dq/dt, dp/dt)，不经过哈密顿结构约束。

输入维度: (batch_size, 2n)，与 HNN 相同
输出维度: (batch_size, 2n) — 直接输出时间导数向量 (dq/dt, dp/dt)

关键区别:
  - HNN: 输出标量 H → 通过 autograd 得到梯度 → 哈密顿结构约束
  - Baseline: 直接输出向量 (dq/dt, dp/dt) → 无物理约束

陷阱:
  - 基线网络在短期预测上可能和 HNN 表现接近
  - 但在长期 rollout 中，基线的能量会发散，而 HNN 近似守恒
  - 基线网络架构应与 HNN 的 MLP 部分完全相同（层数、隐单元数、激活函数）
    以保证公平比较
"""

import torch
import torch.nn as nn


class BaselineNN(nn.Module):
    """
    基线神经网络：直接拟合向量场 f(q,p) → (dq/dt, dp/dt)

    与 HNN 的本质区别：
        - HNN 学习标量能量函数 H(q,p)，通过 autograd 得到梯度
        - BaselineNN 直接学习向量值函数 f(q,p)
        - BaselineNN 不包含任何物理先验，是「纯数据驱动」方法

    面试关键点：
        「BaselineNN 在训练集上的 loss 可能比 HNN 更低，
         因为它没有物理约束。但在测试集的长期积分中，
         BaselineNN 的误差会指数级增长，而 HNN 保持稳定。」
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 200,
            num_hidden_layers: int = 3,
            nonlinearity: str = 'tanh'
    ):
        """
        初始化 BaselineNN 模型

        Args:
            input_dim: 输入维度，必须是 2n（n 个广义坐标 + n 个广义动量）
                       注意：此参数必须与对应的 HNN 完全一致！
            hidden_dim: 隐藏层宽度，推荐 200-500
                        必须与 HNN 的 hidden_dim 相同以保证公平对比
            num_hidden_layers: 隐藏层数量，推荐 3-4 层
                               必须与 HNN 的 num_hidden_layers 相同
            nonlinearity: 激活函数，推荐 tanh
                          必须与 HNN 的 nonlinearity 相同

        重要提示：
            为了保证实验的公平性，BaselineNN 的网络架构（层数、宽度、激活函数）
            必须与 HNN 的 MLP 部分完全一致。唯一的区别在于输出层维度。
        """
        super(BaselineNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim  # 输出维度 = 输入维度（q,p → dq/dt,dp/dt）

        assert input_dim % 2 == 0, \
            f"input_dim 必须是偶数（q 和 p 各 n 维），当前值: {input_dim}"

        # 选择激活函数（与 HNN 保持一致）
        if nonlinearity == 'tanh':
            activation = nn.Tanh
        elif nonlinearity == 'relu':
            activation = nn.ReLU
        else:
            raise ValueError(f"不支持的激活函数: {nonlinearity}")

        # 构建 MLP 网络：从 (q, p) 直接映射到 (dq/dt, dp/dt)
        # 网络结构: input_dim → [hidden_dim × num_hidden_layers] → output_dim(=input_dim)
        #
        # ⚠️ 关键区别：HNN 的最后一层输出维度是 1（标量 H）
        #             而这里输出维度是 2n（向量场）
        layers = []

        # 第一层：输入层 → 第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # 中间隐藏层（与 HNN 完全一致）
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # 输出层：最后一个隐藏层 → 向量输出（无激活函数！）
        # 原因：(dq/dt, dp/dt) 可以取任意实数值
        # 例如：简谐振子的速度可以是正也可以是负
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        # 将所有层封装为 Sequential
        self.net = nn.Sequential(*layers)

        # 初始化权重（与 HNN 使用相同的初始化策略）
        self._init_weights()

    def _init_weights(self):
        """
        使用 Xavier 初始化权重

        与 HNN 使用完全相同的初始化方法，确保公平对比
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        前向传播：直接预测时间导数向量

        数学公式:
            (dq/dt, dp/dt) = MLP(q, p)

        与 HNN forward() 的区别：
            - HNN.forward() 返回标量 H (batch, 1)
            - BaselineNN.forward() 返回向量 (dq/dt, dp/dt) (batch, 2n)

        Args:
            coords: 相空间坐标张量，shape (batch_size, 2n)
                   前 n 列是广义坐标 q = [q1, q2, ..., qn]
                   后 n 列是广义动量 p = [p1, p2, ..., pn]

        Returns:
            dcoords_dt: 预测的时间导数，shape (batch_size, 2n)
                       前 n 列是 dq/dt = [dq1/dt, dq2/dt, ..., dqn/dt]
                       后 n 列是 dp/dt = [dp1/dt, dp2/dt, ..., dpn/dt]

        示例（质量弹簧系统）:
            输入: coords = [[1.0, 0.5]]  # q=1.0, p=0.5
            输出: dcoords_dt = [[-0.5, -1.0]]  # dq/dt=-0.5, dp/dt=-1.0
                  （对应解析解: dq/dt=p/m=-0.5, dp/dt=-kq=-1.0）
        """
        return self.net(coords)

    @staticmethod
    def dynamics(
            t: torch.Tensor,
            coords: torch.Tensor,
            model: 'BaselineNN'
    ) -> torch.Tensor:
        """
        计算 BaselineNN 预测的时间导数（接口兼容 ODE 求解器）

        这个静态方法的签名与 HNN.dynamics() 完全一致，
        使得两个模型可以使用相同的积分器代码！

        数学原理：
            对于 BaselineNN，dynamics 就是简单的 forward() 调用
            不需要 autograd，因为模型直接输出导数

        为什么需要这个方法？
            - 统一接口：RK4 积分器期望 dynamics_fn(t, y, model) 签名
            - 可替换性：在实验中可以无缝切换 HNN ↔ BaselineNN

        Args:
            t: 时间（未使用，但保留接口以兼容 ODE 求解器签名）
            coords: 当前相空间状态，shape (batch_size, 2n)
            model: BaselineNN 模型实例

        Returns:
            dcoords_dt: 预测的时间导数，shape (batch_size, 2n)
        """
        return model(coords)
