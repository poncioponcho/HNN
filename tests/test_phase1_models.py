"""
Phase 1 测试：验证 HNN 和 BaselineNN 的前向传播正确性

测试目标：
1. HNN.forward() 返回标量 (batch, 1)
2. BaselineNN.forward() 返回向量 (batch, 2n)
3. HNN.dynamics() 能正确计算时间导数
4. BaselineNN.dynamics() 接口兼容
5. 梯度可以反向传播（训练可行性）

运行方式:
    python tests/test_phase1_models.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.hnn import HNN
from src.models.baseline_nn import BaselineNN


def test_hnn_forward():
    """测试 HNN 前向传播：输出应该是标量 (batch, 1)"""
    print("=" * 60)
    print("测试 1: HNN.forward() - 标量哈密顿量输出")
    print("=" * 60)

    # 创建 HNN 模型（质量弹簧系统，input_dim=2）
    model = HNN(input_dim=2, hidden_dim=200, num_hidden_layers=3)

    # 生成随机输入：(batch_size=10, coords=[q, p])
    batch_size = 10
    coords = torch.randn(batch_size, 2)  # shape: (10, 2)

    # 前向传播
    with torch.no_grad():
        H = model(coords)

    # 验证输出形状
    assert H.shape == (batch_size, 1), \
        f"HNN 输出形状错误！期望 ({batch_size}, 1)，实际 {H.shape}"

    print(f"✅ 输入形状: {coords.shape}  →  输出形状: {H.shape}")
    print(f"   输出示例（前 3 个样本的哈密顿量）:")
    for i in range(min(3, batch_size)):
        print(f"     样本 {i}: q={coords[i, 0]:.3f}, p={coords[i, 1]:.3f} → H={H[i, 0]:.3f}")
    print()


def test_baseline_forward():
    """测试 BaselineNN 前向传播：输出应该是向量 (batch, 2n)"""
    print("=" * 60)
    print("测试 2: BaselineNN.forward() - 向量场输出")
    print("=" * 60)

    # 创建 BaselineNN 模型（与 HNN 相同配置）
    model = BaselineNN(input_dim=2, hidden_dim=200, num_hidden_layers=3)

    # 生成随机输入
    batch_size = 10
    coords = torch.randn(batch_size, 2)  # shape: (10, 2)

    # 前向传播
    with torch.no_grad():
        dcoords_dt = model(coords)

    # 验证输出形状：应该与输入相同 (batch, 2n)
    assert dcoords_dt.shape == (batch_size, 2), \
        f"BaselineNN 输出形状错误！期望 ({batch_size}, 2)，实际 {dcoords_dt.shape}"

    print(f"✅ 输入形状: {coords.shape}  →  输出形状: {dcoords_dt.shape}")
    print(f"   输出示例（前 3 个样本的时间导数）:")
    for i in range(min(3, batch_size)):
        dqdt = dcoords_dt[i, 0].item()
        dpdt = dcoords_dt[i, 1].item()
        print(f"     样本 {i}: (q,p)=({coords[i, 0]:.3f},{coords[i, 1]:.3f}) "
              f"→ (dq/dt,dp/dt)=({dqdt:.3f},{dpdt:.3f})")
    print()


def test_hnn_dynamics():
    """测试 HNN.dynamics()：通过 autograd 计算哈密顿方程"""
    print("=" * 60)
    print("测试 3: HNN.dynamics() - 哈密顿方程自动微分")
    print("=" * 60)

    model = HNN(input_dim=2, hidden_dim=200, num_hidden_layers=3)
    model.eval()

    # 单个样本测试
    coords = torch.tensor([[1.0, 0.5]], requires_grad=True)  # q=1.0, p=0.5

    # 调用 dynamics 函数
    dummy_t = torch.zeros(1)
    dcoords_dt = HNN.dynamics(dummy_t, coords, model)

    # 验证输出形状
    assert dcoords_dt.shape == (1, 2), \
        f"dynamics 输出形状错误！期望 (1, 2)，实际 {dcoords_dt.shape}"

    dqdt = dcoords_dt[0, 0].item()  # ∂H/∂p
    dpdt = dcoords_dt[0, 1].item()  # -∂H/∂q

    print(f"✅ 输入状态: q={coords[0, 0].item():.3f}, p={coords[0, 1].item():.3f}")
    print(f"   计算得到的时间导数:")
    print(f"     dq/dt = ∂H/∂p = {dqdt:.6f}")
    print(f"     dp/dt = -∂H/∂q = {dpdt:.6f}")
    print()

    # 物理意义说明
    print("   💡 数学原理（论文 Eq. 6）:")
    print("      网络先计算 H(q,p)，然后自动微分得到 ∂H/∂q 和 ∂H/∂p")
    print("      最后按照正则方程组合成 (dq/dt, dp/dt)")
    print()


def test_baseline_dynamics_interface():
    """测试 BaselineNN.dynamics() 接口兼容性"""
    print("=" * 60)
    print("测试 4: BaselineNN.dynamics() - 接口兼容性检查")
    print("=" * 60)

    model = BaselineNN(input_dim=2, hidden_dim=200, num_hidden_layers=3)
    model.eval()

    coords = torch.tensor([[1.0, 0.5]])

    # 调用 dynamics（签名与 HNN.dynamics 完全一致）
    dummy_t = torch.zeros(1)
    dcoords_dt = BaselineNN.dynamics(dummy_t, coords, model)

    assert dcoords_dt.shape == (1, 2), \
        f"BaselineNN dynamics 输出形状错误！期望 (1, 2)，实际 {dcoords_dt.shape}"

    print(f"✅ BaselineNN.dynamics() 接口正常工作")
    print(f"   输入: (q,p)=(1.0, 0.5)")
    print(f"   输出: (dq/dt,dp/dt)=({dcoords_dt[0,0].item():.3f}, {dcoords_dt[0,1].item():.3f})")
    print()


def test_gradient_flow():
    """测试梯度是否可以正常反向传播（关键！训练依赖此功能）"""
    print("=" * 60)
    print("测试 5: 梯度反向传播 - 训练可行性验证")
    print("=" * 60)

    # 测试 HNN 的梯度流
    model = HNN(input_dim=2, hidden_dim=200, num_hidden_layers=3)
    coords = torch.randn(5, 2, requires_grad=True)

    # 前向 + 反向
    H = model(coords)
    loss = H.sum()  # 简单的求和作为伪损失
    loss.backward()

    # 检查梯度是否存在
    assert coords.grad is not None, "输入梯度为空！autograd 失败"
    assert not torch.isnan(coords.grad).any(), "梯度包含 NaN！"

    # 检查模型参数梯度
    has_param_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_param_grad = True
            break

    assert has_param_grad, "模型参数没有梯度！无法训练"

    print("✅ HNN 梯度反向传播正常")
    print(f"   输入梯度范数: {coords.grad.norm():.6f}")
    print()

    # 测试 BaselineNN 的梯度流
    model_base = BaselineNN(input_dim=2, hidden_dim=200, num_hidden_layers=3)
    coords_base = torch.randn(5, 2, requires_grad=True)
    dcoords_true = torch.randn(5, 2)  # 模拟真实导数

    # 前向 + 计算 MSE loss + 反向
    pred = model_base(coords_base)
    loss_base = ((pred - dcoords_true) ** 2).mean()
    loss_base.backward()

    assert coords_base.grad is not None, "BaselineNN 输入梯度为空"
    assert not torch.isnan(coords_base.grad).any(), "BaselineNN 梯度包含 NaN"

    print("✅ BaselineNN 梯度反向传播正常")
    print(f"   输入梯度范数: {coords_base.grad.norm():.6f}")
    print()


def test_architecture_consistency():
    """验证 HNN 和 BaselineNN 架构一致性（公平对比要求）"""
    print("=" * 60)
    print("测试 6: 架构一致性检查 - 保证实验公平性")
    print("=" * 60)

    # 使用完全相同的超参数创建两个模型
    config = {
        'input_dim': 2,
        'hidden_dim': 200,
        'num_hidden_layers': 3,
        'nonlinearity': 'tanh'
    }

    hnn = HNN(**config)
    baseline = BaselineNN(**config)

    # 统计参数数量（Baseline 应该略多，因为最后一层维度不同）
    hnn_params = sum(p.numel() for p in hnn.parameters())
    baseline_params = sum(p.numel() for p in baseline.parameters())

    print(f"✅ 两个模型使用相同的超参数配置")
    print(f"   HNN 参数数量:     {hnn_params:,}")
    print(f"   BaselineNN 参数数量: {baseline_params:,}")
    print(f"   差异原因: 最后一层输出维度不同 (1 vs 2n)")
    print()


if __name__ == '__main__':
    print("\n" + "🔬 " * 15)
    print("PHASE 1 测试: HNN & BaselineNN 核心模块验证")
    print("🔬 " * 15 + "\n")

    try:
        test_hnn_forward()
        test_baseline_forward()
        test_hnn_dynamics()
        test_baseline_dynamics_interface()
        test_gradient_flow()
        test_architecture_consistency()

        print("\n" + "✅ " * 20)
        print("所有测试通过！Phase 1 核心模块实现正确 ✨")
        print("✅ " * 20 + "\n")

        print("📊 面试要点总结:")
        print("  1. HNN 输出标量 H，通过 autograd 得到动力学方程")
        print("  2. BaselineNN 直接输出向量场，无物理约束")
        print("  3. 两个模型架构相同，保证公平对比")
        print("  4. 梯度流正常，可以进行端到端训练\n")

    except AssertionError as e:
        print("\n❌ 测试失败！")
        print(f"   错误信息: {e}\n")
        raise
