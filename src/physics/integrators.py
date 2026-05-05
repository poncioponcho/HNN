"""
Numerical integrators for Hamiltonian dynamics.

Corresponds to: 论文 Section 3-5, 所有实验的 rollout 阶段。
使用 scipy.integrate.solve_ivp 封装 RK4/RK45 积分器。

核心功能:
  - 给定学习到的动力学函数 f(q, p) -> (dq/dt, dp/dt)
  - 从初始条件 (q0, p0) 出发积分 T 步
  - 返回轨迹 [(q0,p0), (q1,p1), ..., (qT,pT)]

默认参数 (论文):
  - method: 'RK45' (Dormand-Prince 自适应步长)
  - rtol: 1e-9
  - atol: 1e-9

输入:
  - fun: callable(t, y) -> dy/dt, 动力学函数
  - t_span: (t0, t_end), 积分时间范围
  - y0: ndarray, 初始状态
输出:
  - sol: OdeResult, 包含 .t (时间点) 和 .y (状态轨迹)

陷阱:
  - 误差容差必须设得很小 (1e-9)，否则数值误差会掩盖模型误差
  - HNN 的动力学函数需要通过 autograd 计算，不能直接用 model.forward()
  - 在 rollout 时需要将模型设为 eval 模式，关闭 dropout/batchnorm
  - 积分器不保证能量守恒（除非使用辛积分器），但高精度 RK45 足够用于评估
"""
pass
