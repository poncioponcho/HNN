"""
Test RK4 integrator accuracy.

Corresponds to: 论文 Section 3-5, 所有实验的 rollout 阶段。
验证 scipy.integrate.solve_ivp 封装的 RK4/RK45 积分器精度。

测试用例:
  - test_harmonic_oscillator: 质量-弹簧系统的解析解对比
    - 解析解: q(t) = A*cos(ωt + φ), p(t) = -A*m*ω*sin(ωt + φ)
    - 数值解误差应 < 1e-6 (rtol=atol=1e-9)
  - test_pendulum_energy_conservation: 单摆能量守恒性
    - 积分 100 步后能量偏差应 < 1e-6
  - test_two_body_conservation: 双体问题守恒性
    - 能量和角动量偏差应 < 1e-5

积分器参数 (论文):
  - method: RK45
  - rtol: 1e-9
  - atol: 1e-9

陷阱:
  - 误差容差设为 1e-9 是论文标准，测试中应使用相同值
  - 质量弹簧系统有解析解，是验证积分器精度的最佳选择
  - 长时间积分 (T > 100) 可能累积误差
"""
pass
