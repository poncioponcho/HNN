"""
Numerical integrators for Hamiltonian dynamics.

Corresponds to: 论文 Section 3-5, 所有实验的 rollout 阶段。
手动实现 RK4（四阶龙格-库塔）积分器，用于从学习到的动力学函数生成轨迹。

核心功能:
  - 给定动力学函数 f(t, y) -> dy/dt
  - 从初始条件 y0 出发积分多步
  - 返回完整轨迹 [y0, y1, ..., yT]

为什么选择 RK4 而不是欧拉法？
  - 欧拉法：局部截断误差 O(h²)，全局误差 O(h)
           长期积分能量漂移严重，不适合保守系统
  - RK4：    局部截断误差 O(h⁵)，全局误差 O(h⁴)
           能量漂移小得多，适合哈密顿系统评估
  - 注意：RK4 不是真正的辛积分器（如 Störmer-Verlet），
          但对于 HNN 的 rollout 评估已经足够精确

数学原理（经典 RK4 公式）:
    k₁ = f(tₙ, yₙ)
    k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
    k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
    k₄ = f(tₙ + h,   yₙ + h·k₃)
    yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

输入参数:
  - dynamics_fn: callable(t, y) -> dy/dt, 动力学右端项
  - t: float, 当前时间
  - y: Tensor, shape (batch, dim), 当前状态向量
  - h: float, 时间步长

输出:
  - y_next: Tensor, shape (batch, dim), 下一步状态

陷阱:
  - 步长 h 不能太大，否则数值不稳定
  - 对于刚性系统（stiff systems），需要隐式方法
  - HNN 的 dynamics_fn 必须使用 autograd 版本（HNN.dynamics）
"""

import torch
from typing import Callable, Optional, Tuple, List


def rk4_step(
        dynamics_fn: Callable,
        t: float,
        y: torch.Tensor,
        h: float
) -> torch.Tensor:
    """
    单步 RK4 积分（纯函数式实现）

    这是整个积分器的核心！每一步调用 4 次动力学函数，
    通过加权平均得到高精度的状态更新。

    数学公式:
        k₁ = f(tₙ, yₙ)                              # 在当前点求斜率
        k₂ = f(tₙ + h/2, yₙ + (h/2)·k₁)            # 用 k₁ 预测中点斜率
        k₃ = f(tₙ + h/2, yₙ + (h/2)·k₂)            # 用 k₂ 修正中点斜率
        k₄ = f(tₙ + h,   yₙ + h·k₃)                # 用 k₃ 预测终点斜率
        yₙ₊₁ = yₙ + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)   # 加权平均更新

    为什么是这些权重 (1, 2, 2, 1)/6？
        这些权重来自 Simpson 积分规则的推导，
        保证对多项式 ≤ 3 次精确成立（即 4 阶精度）

    Args:
        dynamics_fn: 动力学函数，签名必须是 (t, y) -> dy/dt
                     其中 t 是标量时间，y 是状态张量
                     返回值是与 y 同形状的时间导数张量
        t: 当前时间点（浮点数）
           对于自治系统（不显含时间），此参数会被忽略但保留接口
        y: 当前状态向量，shape (batch_size, state_dim)
           例如质量弹簧系统: y = [q, p]，state_dim=2
        h: 时间步长（浮点数）
           推荐范围: 0.001 ~ 0.1（取决于系统时间尺度）
           太大 → 数值不稳定；太小 → 计算成本高

    Returns:
        y_next: 下一步的状态向量，shape 与 y 相同
               即 y(t+h) 的近似值

    Example（简谐振子单步）:
        >>> def harmonic_oscillator(t, y):
        ...     q, p = y[:, 0], y[:, 1]
        ...     dqdt = p
        ...     dpdt = -q  # 假设 m=k=1
        ...     return torch.stack([dqdt, dpdt], dim=1)
        >>> y0 = torch.tensor([[1.0, 0.0]])  # q=1, p=0
        >>> y1 = rk4_step(harmonic_oscillator, 0.0, y0, h=0.01)
        >>> # y1 ≈ [cos(0.01), -sin(0.01)] ≈ [0.99995, -0.0099998]
    """
    # Step 1: 计算 k₁ = f(tₙ, yₙ)
    # 这是在当前点的斜率估计
    k1 = dynamics_fn(t, y)  # shape: (batch, dim)

    # Step 2: 计算 k₂ = f(tₙ + h/2, yₙ + (h/2)·k₁)
    # 使用欧拉法预测的中点状态来计算斜率
    y_k2 = y + (h / 2) * k1  # 显式欧拉半步
    k2 = dynamics_fn(t + h / 2, y_k2)

    # Step 3: 计算 k₃ = f(tₙ + h/2, yₙ + (h/2)·k₂)
    # 使用 k₂ 修正后的中点状态重新计算斜率（更准确）
    y_k3 = y + (h / 2) * k2
    k3 = dynamics_fn(t + h / 2, y_k3)

    # Step 4: 计算 k₄ = f(tₙ + h, yₙ + h·k₃)
    # 使用 k₃ 预测的终点状态计算斜率
    y_k4 = y + h * k3
    k4 = dynamics_fn(t + h, y_k4)

    # Step 5: 加权组合四个斜率，得到最终的状态更新
    # 权重 (1, 2, 2, 1)/6 来自 Simpson 规则的推导
    # 这个加权平均使得 RK4 具有 4 阶精度
    y_next = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return y_next


def solve_ivp_rk4(
        dynamics_fn: Callable,
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        n_steps: int,
        return_trajectory: bool = True
) -> torch.Tensor:
    """
    批量 RK4 积分器：在时间区间内进行多步积分

    封装 rk4_step() 为完整的轨迹生成器，
    用于 HNN 的长期 rollout 评估。

    与 scipy.integrate.solve_ivp 的区别:
      - 本实现是纯 PyTorch，支持 GPU 加速和自动微分
      - 固定步长（非自适应），更适合批量处理
      - 返回 PyTorch 张量而非 numpy 数组

    Args:
        dynamics_fn: 动力学函数 (t, y) -> dy/dt
        y0: 初始条件，shape (batch_size, state_dim)
                   可以同时积分多个轨迹（不同初始条件）
        t_span: 时间区间元组 (t_start, t_end)
                 例如 (0.0, 10.0) 表示从 t=0 积分到 t=10
        n_steps: 积分步数（整数）
                 总时间 = t_end - t_start
                 步长 h = (t_end - t_start) / n_steps
                 推荐: 根据所需精度选择，通常 100-1000 步
        return_trajectory: 是否返回完整轨迹
                          True  → 返回所有中间状态 (n_steps+1, batch, dim)
                          False → 仅返回最终状态 (batch, dim)

    Returns:
        如果 return_trajectory=True:
            trajectory: shape (n_steps+1, batch_size, state_dim)
                       包含 t₀, t₁, ..., t_{n_steps} 的所有状态
                       trajectory[0] = y0（初始条件）
        如果 return_trajectory=False:
            y_final: shape (batch_size, state_dim)
                    仅返回最终时刻的状态

    Example（简谐振子轨道）:
        >>> y0 = torch.tensor([[1.0, 0.0]])  # 从 q=1, p=0 开始
        >>> traj = solve_ivp_rk4(harmonic_osc, y0, (0, 6.28), n_steps=628)
        >>> # traj.shape = (629, 1, 2)，约一个完整周期
        >>> final_q = traj[-1, 0, 0].item()  # 应该接近 1.0
    """
    t_start, t_end = t_span
    h = (t_end - t_start) / n_steps  # 计算固定步长

    if return_trajectory:
        # 初始化轨迹存储列表
        trajectory = [y0.unsqueeze(0)]  # (1, batch, dim)

        # 当前时间和状态
        t = t_start
        y = y0.clone()

        # 循环执行 n_steps 步 RK4
        for step in range(n_steps):
            y = rk4_step(dynamics_fn, t, y, h)  # 单步积分
            t = t + h  # 更新时间
            trajectory.append(y.unsqueeze(0))  # 记录状态

        # 拼接为完整轨迹张量
        trajectory = torch.cat(trajectory, dim=0)  # (n_steps+1, batch, dim)
        return trajectory
    else:
        # 不保存中间结果，仅计算最终状态（节省内存）
        t = t_start
        y = y0.clone()

        for _ in range(n_steps):
            y = rk4_step(dynamics_fn, t, y, h)
            t += h

        return y


class RK4Integrator:
    """
    RK4 积分器类（面向对象封装）

    提供更灵活的接口，支持：
      - 状态重置
      - 分步执行（step-by-step）
      - 中间结果访问

    适用场景:
      - 需要在积分过程中插入自定义操作（如记录能量）
      - 需要动态调整步长
      - 需要中断和恢复积分过程
    """

    def __init__(
            self,
            dynamics_fn: Callable,
            t_start: float = 0.0,
            h: float = 0.01
    ):
        """
        初始化 RK4 积分器实例

        Args:
            dynamics_fn: 动力学函数 (t, y) -> dy/dt
            t_start: 初始时间，默认 0.0
            h: 时间步长，默认 0.01
        """
        self.dynamics_fn = dynamics_fn
        self.t = t_start
        self.h = h
        self.y = None  # 当前状态（待设置）
        self.history_t = []  # 时间历史
        self.history_y = []  # 状态历史

    def set_state(self, y0: torch.Tensor, t0: Optional[float] = None):
        """
        设置初始状态并清空历史

        Args:
            y0: 初始状态，shape (batch, dim)
            t0: 初始时间（可选），默认保持原值
        """
        self.y = y0.clone()
        if t0 is not None:
            self.t = t0
        self.history_t = [self.t]
        self.history_y = [self.y.clone()]

    def step(self) -> torch.Tensor:
        """
        执行单步 RK4 积分

        Returns:
            y_new: 更新后的状态
        """
        assert self.y is not None, "请先调用 set_state() 设置初始条件"

        self.y = rk4_step(self.dynamics_fn, self.t, self.y, self.h)
        self.t += self.h

        # 记录历史
        self.history_t.append(self.t)
        self.history_y.append(self.y.clone())

        return self.y

    def integrate(
            self,
            n_steps: int,
            record_history: bool = True
    ) -> torch.Tensor:
        """
        连续执行多步积分

        Args:
            n_steps: 积分步数
            record_history: 是否记录中间结果

        Returns:
            最终状态或完整轨迹
        """
        for _ in range(n_steps):
            self.step()

        if record_history:
            return torch.stack(self.history_y, dim=0)
        else:
            return self.y

    def get_trajectory(self) -> Tuple[List[float], List[torch.Tensor]]:
        """
        获取已记录的轨迹

        Returns:
            (times, states): 时间列表和状态列表
        """
        return self.history_t, self.history_y


def compute_energy_error(
        trajectory: torch.Tensor,
        true_energy: float,
        energy_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算轨迹上的能量漂移（用于评估积分器精度）

    这是评估保守系统积分器性能的关键指标！

    数学定义:
        相对能量误差 = |E(t) - E(0)| / |E(0)| × 100%

    对于理想的辛积分器，这个误差应该保持在机器精度附近。
    对于普通 RK4，误差会随时间缓慢增长（但比欧拉法好得多）。

    Args:
        trajectory: 完整轨迹，shape (n_times, batch, dim)
        true_energy: 真实能量值（常数，用于保守系统）
        energy_fn: 能量函数 (y) -> E(y)
                  例如简谐振子: E = (p² + q²) / 2

    Returns:
        relative_errors: 每个时间点的相对能量误差，shape (n_times,)
        absolute_errors: 每个时间点的绝对能量误差，shape (n_times,)

    Example:
        >>> traj = solve_ivp_rk4(dynamics, y0, (0, 100), n_steps=10000)
        >>> rel_err, abs_err = compute_energy_error(traj, E0, energy_fn)
        >>> max_drift = rel_err.max().item()  # 最大能量漂移百分比
    """
    n_times = trajectory.shape[0]
    energies = []
    
    # 计算每个时间点的能量
    for i in range(n_times):
        y_i = trajectory[i]  # (batch, dim)
        E_i = energy_fn(y_i)  # (batch,) 或标量
        energies.append(E_i)
    
    energies = torch.stack(energies, dim=0)  # (n_times, batch)
    
    # 计算误差
    abs_errors = torch.abs(energies - true_energy)  # 绝对误差
    rel_errors = abs_errors / (abs(true_energy) + 1e-8)  # 相对误差
    
    return rel_errors, abs_errors
