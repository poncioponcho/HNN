"""
Test HNN loss function sign correctness.

Corresponds to: 论文 Section 3, Eq. 5.
验证 HNN 损失函数的符号和计算正确性。

哈密顿方程:
  dq/dt = +∂H/∂p   (正号)
  dp/dt = -∂H/∂q   (负号)

HNN Loss:
  L = ||∂H/∂p - dq/dt||^2 + ||-∂H/∂q - dp/dt||^2
    = ||∂H/∂p - dq/dt||^2 + ||∂H/∂q + dp/dt||^2

测试用例:
  - test_hnn_loss_zero_for_correct_hamiltonian:
    给定正确的哈密顿量 H = p^2/(2m) + k*q^2/2，
    验证 loss ≈ 0
  - test_hnn_loss_sign_dq:
    验证 ∂H/∂p 前面是正号 (dq/dt = +∂H/∂p)
  - test_hnn_loss_sign_dp:
    验证 ∂H/∂q 前面是负号 (dp/dt = -∂H/∂q)
  - test_hnn_loss_positive:
    验证 loss 始终非负
  - test_baseline_loss_format:
    验证基线 loss 是简单的 MSE

陷阱:
  - ∂H/∂q 前面的负号是最容易出错的地方
  - 论文 Eq. 5 写的是 ||∂H/∂q + dp/dt||^2，
    等价于 ||-∂H/∂q - dp/dt||^2
  - 梯度索引: x = [q, p], ∂H/∂x = [∂H/∂q, ∂H/∂p]
    - dq/dt_pred = ∂H/∂x[:, n:]  = ∂H/∂p
    - dp/dt_pred = -∂H/∂x[:, :n] = -∂H/∂q
"""
pass
