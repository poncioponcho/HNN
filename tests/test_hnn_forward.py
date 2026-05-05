"""
Test HNN forward pass and autograd gradient computation.

Corresponds to: 论文 Section 3, Eq. 5-6.
验证 HNN 模型的核心功能:
  1. forward 输出标量 H (shape: (batch, 1))
  2. autograd.grad 能正确计算 ∂H/∂x
  3. ∂H/∂p 和 -∂H/∂q 的维度正确

测试用例:
  - test_hnn_output_shape: 验证输出形状为 (batch, 1)
  - test_hnn_gradient_exists: 验证 autograd.grad 返回非 None
  - test_hnn_gradient_shape: 验证梯度形状与输入相同
  - test_hnn_symplectic_structure: 验证 ∂H/∂p 和 -∂H/∂q 的符号正确
    - 对于简单哈密顿量 H = p^2/2 + q^2/2:
      ∂H/∂p = p, -∂H/∂q = -q
  - test_hnn_create_graph: 验证 create_graph=True 允许二阶反向传播

陷阱:
  - 输入 x 必须设置 requires_grad=True
  - H.sum() 而非 H 用于 grad，因为 grad 要求标量输入
  - create_graph=True 是必须的，否则 loss 无法反向传播
"""
pass
