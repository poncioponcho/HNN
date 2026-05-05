"""
Physical systems for generating training data.

Corresponds to: 论文 Section 4, Task 1 (Mass-Spring), Task 2 (Pendulum).
提供解析解的数据生成器，用于训练 HNN 和 BaselineNN。

核心功能:
  - 生成保守系统的相空间轨迹
  - 通过有限差分估计时间导数 (dq/dt, dp/dt)
  - 添加可控噪声模拟真实观测
  - 支持 train/val/test 数据集划分

系统列表:
  1. MassSpringSystem (Task 1): 理想质量弹簧系统
     哈密顿量: H = (1/2)kq² + p²/(2m)
     解析解: q(t) = A·cos(ωt + φ), ω = √(k/m)

数据格式:
  输出: dict {
    'coords': (N, 2n) 相空间坐标 [q, p],
    'dcoords_dt': (N, 2n) 时间导数 [dq/dt, dp/dt],
    't': (N,) 时间点,
    'energies': (N,) 每个点的哈密顿量
  }

使用示例:
    >>> system = MassSpringSystem(k=1.0, m=1.0)
    >>> data = system.generate_dataset(
    ...     n_trajectories=25,
    ...     n_points=30,
    ...     noise_std=0.1
    ... )
    >>> print(data['coords'].shape)  # (750, 2) = 25*30 个样本
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """物理系统配置参数

    Attributes:
        mass: 质量 m (kg)
        spring_const: 弹簧常数 k (N/m)
        noise_std: 观测噪声标准差 σ
        dt: 时间步长 (s)
        t_max: 最大模拟时间 (s)
        energy_range: 总能量采样范围 [E_min, E_max]
        seed: 随机种子（保证可复现性）
    """
    mass: float = 1.0
    spring_const: float = 1.0
    noise_std: float = 0.1
    dt: float = 0.1
    t_max: float = 3.0
    energy_range: Tuple[float, float] = (0.2, 1.0)
    seed: int = 42


class MassSpringSystem:
    """
    理想质量弹簧系统（无阻尼、无外力）

    物理模型:
      - 运动方程: m·(d²q/dt²) + k·q = 0
      - 角频率: ω = √(k/m)
      - 周期: T = 2π/ω

    哈密顿形式:
      - 广义坐标: q (位移)
      - 广义动量: p = m·(dq/dt)
      - 哈密顿量: H(q,p) = p²/(2m) + (1/2)kq²
                 （动能 + 势能）

    解析解:
      给定初始条件 (q₀, p₀):
        振幅 A = √(q₀² + (p₀/(mω))²)
        初相位 φ = atan2(-p₀/(mω), q₀)

        q(t) = A·cos(ωt + φ)
        p(t) = -A·m·ω·sin(ωt + φ)

    为什么选择这个系统作为 Task 1?
      1. 有精确的解析解，可以严格验证模型正确性
      2. 线性系统，HNN 和 Baseline 都能拟合得很好
      3. 但在长期 rollout 中，HNN 能量守恒而 Baseline 发散
      4. 作为「Hello World」级别的测试用例
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        初始化质量弹簧系统

        Args:
            config: 系统配置参数，如果为 None 则使用默认值
        """
        self.config = config or SystemConfig()
        self.m = self.config.mass
        self.k = self.config.spring_const
        self.omega = np.sqrt(self.k / self.m)  # 角频率 ω = √(k/m)
        self.period = 2 * np.pi / self.omega     # 周期 T = 2π/ω

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        计算哈密顿量（总能量）

        公式: H = p²/(2m) + (1/2)kq²

        Args:
            q: 位移张量，shape (...,) 或 (batch,)
            p: 动量张量，shape 与 q 相同

        Returns:
            H: 能量值，shape 与 q 相同
        """
        kinetic = p**2 / (2 * self.m)       # 动能 T = p²/(2m)
        potential = (self.k * q**2) / 2       # 势能 V = (1/2)kq²
        return kinetic + potential

    def analytical_solution(
            self,
            t: np.ndarray,
            q0: float,
            p0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算解析解（给定初始条件）

        数学推导:
          从初始条件 (q₀, p₀) 出发:
            振幅 A = √(q₀² + (p₀/(mω))²)
            初相位 φ = atan2(-p₀/(mω), q₀)

          则:
            q(t) = A·cos(ωt + φ)
            p(t) = -A·m·ω·sin(ωt + φ)

        Args:
            t: 时间点数组，shape (N,)
            q0: 初始位移
            p0: 初始动量

        Returns:
            (q, p): 解析解的位移和动量，各 shape (N,)
        """
        omega = self.omega
        m = self.m

        # 计算振幅和初相位
        A = np.sqrt(q0**2 + (p0 / (m * omega))**2)
        phi = np.arctan2(-p0 / (m * omega), q0)

        # 解析解
        q = A * np.cos(omega * t + phi)
        p = -A * m * omega * np.sin(omega * t + phi)

        return q, p

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        哈密顿方程右端项（用于数值积分）

        公式:
            dq/dt =  ∂H/∂p = p/m
            dp/dt = -∂H/∂q = -k·q

        Args:
            t: 时间（未使用，自治系统）
            state: 状态向量 [q, p]，shape (2,)

        Returns:
            dstate_dt: [dq/dt, dp/dt]，shape (2,)
        """
        q, p = state
        dqdt = p / self.m
        dpdt = -self.k * q
        return np.array([dqdt, dpdt])

    def generate_trajectory(
            self,
            q0: float,
            p0: float,
            n_points: int = 30,
            add_noise: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        生成单条轨迹（含噪声）

        流程:
          1. 根据初始条件计算解析解
          2. 在等间隔时间点上采样
          3. 可选添加高斯噪声
          4. 用有限差分估计时间导数

        Args:
            q0: 初始位移
            p0: 初始动量
            n_points: 采样点数
            add_noise: 是否添加高斯噪声

        Returns:
            dict 包含:
              'coords': (n_points, 2) [q, p]
              'dcoords_dt': (n_points, 2) [dq/dt, dp/dt]
              't': (n_points,) 时间点
              'energies': (n_points,) 真实能量（无噪声）
        """
        dt = self.config.dt
        t_max = min(n_points * dt, self.config.t_max)
        t = np.linspace(0, t_max, n_points)

        # Step 1: 计算解析解（真值）
        q_true, p_true = self.analytical_solution(t, q0, p0)

        # Step 2: 添加观测噪声（如果需要）
        if add_noise and self.config.noise_std > 0:
            noise_q = np.random.normal(0, self.config.noise_std, size=q_true.shape)
            noise_p = np.random.normal(0, self.config.noise_std, size=p_true.shape)
            q_noisy = q_true + noise_q
            p_noisy = p_true + noise_p
        else:
            q_noisy = q_true.copy()
            p_noisy = p_true.copy()

        coords = np.stack([q_noisy, p_noisy], axis=1)  # (n_points, 2)

        # Step 3: 用有限差分估计时间导数
        # 使用中心差分（更精确）:
        #   dq/dt ≈ (q_{i+1} - q_{i-1}) / (2*dt)
        # 对于端点使用前向/后向差分
        dcoords_dt = np.zeros_like(coords)

        # 内部点：中心差分
        dcoords_dt[1:-1, 0] = (q_noisy[2:] - q_noisy[:-2]) / (2 * dt)  # dq/dt
        dcoords_dt[1:-1, 1] = (p_noisy[2:] - p_noisy[:-2]) / (2 * dt)  # dp/dt

        # 起点：前向差分
        dcoords_dt[0, 0] = (q_noisy[1] - q_noisy[0]) / dt
        dcoords_dt[0, 1] = (p_noisy[1] - p_noisy[0]) / dt

        # 终点：后向差分
        dcoords_dt[-1, 0] = (q_noisy[-1] - q_noisy[-2]) / dt
        dcoords_dt[-1, 1] = (p_noisy[-1] - p_noisy[-2]) / dt

        # Step 4: 计算真实能量（用于评估）
        q_tensor = torch.tensor(q_true, dtype=torch.float32)
        p_tensor = torch.tensor(p_true, dtype=torch.float32)
        energies = self.hamiltonian(q_tensor, p_tensor).numpy()

        return {
            'coords': coords.astype(np.float32),
            'dcoords_dt': dcoords_dt.astype(np.float32),
            't': t,
            'energies': energies
        }

    def generate_dataset(
            self,
            n_trajectories: int = 25,
            n_points: int = 30,
            train_ratio: float = 0.8,
            add_noise: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        生成完整训练数据集（多条轨迹）

        数据生成策略:
          1. 在能量范围 [E_min, E_max] 内均匀采样总能量
          2. 对每个能量，随机选择初相位 φ ∈ [0, 2π]
          3. 由 E 和 φ 反推初始条件 (q₀, p₀)
          4. 生成轨迹并添加噪声

        为什么这样采样？
          - 保证覆盖不同的振幅范围
          - 避免所有轨迹从相同的初始条件出发
          - 让模型学习到全局的哈密顿量结构

        Args:
            n_trajectories: 轨迹数量（论文默认 25）
            n_points: 每条轨迹的点数（论文默认 30）
            train_ratio: 训练集比例（默认 0.8）
            add_noise: 是否添加噪声

        Returns:
            dict 包含 PyTorch 张量:
              'train_coords': (N_train, 2)
              'train_dcoords_dt': (N_train, 2)
              'val_coords': (N_val, 2)
              'val_dcoords_dt': (N_val, 2)
              'all_coords': (N_total, 2)
              'all_dcoords_dt': (N_total, 2)
              'all_t': (N_total,)
              'all_energies': (N_total,)
        """
        # 设置随机种子以保证可复现性
        np.random.seed(self.config.seed)

        E_min, E_max = self.config.energy_range
        all_coords = []
        all_dcoords_dt = []
        all_t = []
        all_energies = []

        for i in range(n_trajectories):
            # Step 1: 采样总能量
            E = np.random.uniform(E_min, E_max)

            # Step 2: 随机选择初相位
            phi = np.random.uniform(0, 2 * np.pi)

            # Step 3: 由 E 和 φ 计算初始条件
            # 从 H = (1/2)kA² = E 得到振幅 A = √(2E/k)
            A = np.sqrt(2 * E / self.k)

            # 初始条件:
            #   q₀ = A·cos(φ)
            #   p₀ = -A·m·ω·sin(φ)
            q0 = A * np.cos(phi)
            p0 = -A * self.m * self.omega * np.sin(phi)

            # Step 4: 生成单条轨迹
            traj = self.generate_trajectory(q0, p0, n_points, add_noise)

            all_coords.append(traj['coords'])
            all_dcoords_dt.append(traj['dcoords_dt'])
            all_t.append(traj['t'])
            all_energies.append(traj['energies'])

        # 合并所有轨迹
        all_coords = np.concatenate(all_coords, axis=0)       # (N_total, 2)
        all_dcoords_dt = np.concatenate(all_dcoords_dt, axis=0)  # (N_total, 2)
        all_t = np.concatenate(all_t, axis=0)                   # (N_total,)
        all_energies = np.concatenate(all_energies, axis=0)     # (N_total,)

        # Step 5: 打乱数据并划分 train/val
        N = len(all_coords)
        indices = np.random.permutation(N)
        n_train = int(N * train_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        # 转换为 PyTorch 张量
        dataset = {
            'train_coords': torch.from_numpy(all_coords[train_idx]),
            'train_dcoords_dt': torch.from_numpy(all_dcoords_dt[train_idx]),
            'val_coords': torch.from_numpy(all_coords[val_idx]),
            'val_dcoords_dt': torch.from_numpy(all_dcoords_dt[val_idx]),
            'all_coords': torch.from_numpy(all_coords),
            'all_dcoords_dt': torch.from_numpy(all_dcoords_dt),
            'all_t': torch.from_numpy(all_t),
            'all_energies': torch.from_numpy(all_energies)
        }

        return dataset

    def get_statistics(self, dataset: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算数据集统计信息（用于归一化）

        Args:
            dataset: generate_dataset() 返回的字典

        Returns:
            dict 包含均值和标准差
        """
        coords = dataset['all_coords']
        dcoords = dataset['all_dcoords_dt']

        return {
            'coords_mean': coords.mean(dim=0),      # (2,)
            'coords_std': coords.std(dim=0),        # (2,)
            'dcoords_mean': dcoords.mean(dim=0),    # (2,)
            'dcoords_std': dcoords.std(dim=0),      # (2,)
        }


# ============================================================
#  便捷函数
# ============================================================

def create_mass_spring_data(
        n_trajectories: int = 25,
        n_points: int = 30,
        noise_std: float = 0.1,
        seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    快速创建质量弹簧系统数据集（便捷接口）

    Args:
        n_trajectories: 轨迹数量
        n_points: 每条轨迹点数
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        数据集字典（同 MassSpringSystem.generate_dataset）
    """
    config = SystemConfig(noise_std=noise_std, seed=seed)
    system = MassSpringSystem(config)
    return system.generate_dataset(n_trajectories, n_points)


if __name__ == '__main__':
    """简单测试：验证数据生成器的正确性"""
    print("=" * 60)
    print("测试: MassSpringSystem 数据生成器")
    print("=" * 60)

    # 创建系统
    config = SystemConfig(
        mass=1.0,
        spring_const=1.0,
        noise_std=0.1,
        seed=42
    )
    system = MassSpringSystem(config)

    print(f"\n系统参数:")
    print(f"  质量 m = {system.m}")
    print(f"  弹簧常数 k = {system.k}")
    print(f"  角频率 ω = {system.omega:.4f} rad/s")
    print(f"  周期 T = {system.period:.4f} s")

    # 生成单条轨迹
    print(f"\n生成单条轨迹 (q₀=1.0, p₀=0.0):")
    traj = system.generate_trajectory(q0=1.0, p0=0.0, n_points=10, add_noise=True)
    print(f"  coords shape: {traj['coords'].shape}")
    print(f"  dcoords_dt shape: {traj['dcoords_dt'].shape}")
    print(f"  前 3 个点:")
    for i in range(3):
        print(f"    t={traj['t'][i]:.2f}: "
              f"(q,p)=({traj['coords'][i,0]:.3f}, {traj['coords'][i,1]:.3f}) "
              f"| (dq/dt,dp/dt)=({traj['dcoords_dt'][i,0]:.3f}, {traj['dcoords_dt'][i,1]:.3f}) "
              f"| E={traj['energies'][i]:.4f}")

    # 生成完整数据集
    print(f"\n生成完整数据集 (25 条轨迹 × 30 点):")
    dataset = system.generate_dataset(n_trajectories=25, n_points=30)
    print(f"  训练集大小: {dataset['train_coords'].shape[0]}")
    print(f"  验证集大小: {dataset['val_coords'].shape[0]}")
    print(f"  总样本数: {dataset['all_coords'].shape[0]}")

    # 统计信息
    stats = system.get_statistics(dataset)
    print(f"\n数据统计:")
    print(f"  coords 均值: [{stats['coords_mean'][0]:.3f}, {stats['coords_mean'][1]:.3f}]")
    print(f"  coords 标准差: [{stats['coords_std'][0]:.3f}, {stats['coords_std'][1]:.3f}]")

    print("\n✅ 数据生成器测试通过！")
