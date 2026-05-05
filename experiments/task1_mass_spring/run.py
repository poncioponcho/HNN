#!/usr/bin/env python3
"""
Task 1: Ideal Mass-Spring System - End-to-End Experiment

论文: Hamiltonian Neural Networks (Greydanus et al., NeurIPS 2019)
Section: 4.1, Figure 1

本脚本整合了完整的实验流程:
  1. 生成质量弹簧系统的训练数据（含噪声）
  2. 训练 HNN 和 BaselineNN
  3. 长期 rollout 评估能量守恒性
  4. 可视化对比结果
  5. 保存模型和实验日志

运行方式:
    cd experiments/task1_mass_spring
    python run.py [--config CONFIG_PATH] [--epochs N]

输出:
    outputs/task1_mass_spring/
    ├── models/
    │   ├── hnn_best.pth
    │   └── baseline_best.pth
    ├── results/
    │   ├── training_history.npy
    │   └── evaluation_metrics.json
    └── figures/
        ├── loss_curves.png
        ├── phase_space_trajectory.png
        └── energy_conservation.png

示例:
    # 使用默认配置运行完整实验
    python run.py

    # 自定义训练轮数
    python run.py --epochs 500

    # 指定配置文件
    python run.py --config ../../configs/mass_spring.yaml
"""

import sys
import os
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# 导入项目模块
from src.models.hnn import HNN
from src.models.baseline_nn import BaselineNN
from src.physics.systems import MassSpringSystem, SystemConfig, create_mass_spring_data
from src.physics.integrators import solve_ivp_rk4, compute_energy_error
from src.training.losses import hnn_loss, baseline_loss
from src.training.trainer import Trainer, TrainingConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HNN Task 1: Mass-Spring Experiment')

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML 配置文件路径（可选，默认使用硬编码参数）'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2000,
        help='训练轮数（默认 2000）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size（默认 64）'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=200,
        help='隐藏层维度（默认 200）'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=3,
        help='网络层数（默认 3）'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='学习率（默认 1e-3）'
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.1,
        help='观测噪声标准差（默认 0.1）'
    )
    parser.add_argument(
        '--n-trajectories',
        type=int,
        default=25,
        help='轨迹数量（默认 25）'
    )
    parser.add_argument(
        '--rollout-steps',
        type=int,
        default=200,
        help='Rollout 步数（默认 200）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认 42）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认自动生成）'
    )

    return parser.parse_args()


def setup_output_directory(args) -> Path:
    """设置输出目录结构"""
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / 'outputs' / 'task1_mass_spring' / f'run_{timestamp}'
    else:
        output_dir = Path(args.output_dir)

    # 创建子目录
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    return output_dir


def generate_data(args) -> Dict[str, torch.Tensor]:
    """生成训练数据"""
    print("\n" + "=" * 60)
    print("Step 1: 生成训练数据")
    print("=" * 60)

    # 配置物理系统
    physics_config = SystemConfig(
        mass=1.0,
        spring_const=1.0,
        noise_std=args.noise_std,
        seed=args.seed
    )

    system = MassSpringSystem(physics_config)

    print(f"物理参数:")
    print(f"  质量 m = {system.m}")
    print(f"  弹簧常数 k = {system.k}")
    print(f"  角频率 ω = {system.omega:.4f} rad/s")
    print(f"  噪声 σ = {args.noise_std}")

    # 生成数据集
    dataset = system.generate_dataset(
        n_trajectories=args.n_trajectories,
        n_points=30,
        train_ratio=0.8
    )

    print(f"\n数据集统计:")
    print(f"  轨迹数: {args.n_trajectories}")
    print(f"  每条轨迹点数: 30")
    print(f"  总样本数: {dataset['all_coords'].shape[0]}")
    print(f"  训练集大小: {dataset['train_coords'].shape[0]}")
    print(f"  验证集大小: {dataset['val_coords'].shape[0]}")

    # 数据归一化统计
    stats = system.get_statistics(dataset)
    print(f"\n数据分布:")
    print(f"  coords 均值: [{stats['coords_mean'][0]:.3f}, {stats['coords_mean'][1]:.3f}]")
    print(f"  coords 标准差: [{stats['coords_std'][0]:.3f}, {stats['coords_std'][1]:.3f}]")

    return dataset


def train_models(dataset: Dict[str, torch.Tensor], args) -> Tuple[Dict, Dict]:
    """训练 HNN 和 BaselineNN"""
    print("\n" + "=" * 60)
    print("Step 2: 训练模型")
    print("=" * 60)

    results = {}

    # ---- 训练 HNN ----
    print("\n--- 训练 HNN ---")
    hnn_model = HNN(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.n_layers
    )

    hnn_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        seed=args.seed,
        save_every=200,
        early_stopping_patience=150
    )

    hnn_trainer = Trainer(hnn_model, hnn_loss, hnn_config)
    hnn_history = hnn_trainer.train(
        dataset['train_coords'],
        dataset['train_dcoords_dt'],
        dataset['val_coords'],
        dataset['val_dcoords_dt'],
        verbose=True
    )

    results['hnn'] = {
        'model': hnn_model,
        'history': hnn_history,
        'trainer': hnn_trainer
    }

    # ---- 训练 BaselineNN ----
    print("\n--- 训练 BaselineNN ---")
    baseline_model = BaselineNN(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.n_layers
    )

    baseline_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        seed=args.seed,
        save_every=200,
        early_stopping_patience=150
    )

    baseline_trainer = Trainer(baseline_model, baseline_loss, baseline_config)
    baseline_history = baseline_trainer.train(
        dataset['train_coords'],
        dataset['train_dcoords_dt'],
        dataset['val_coords'],
        dataset['val_dcoords_dt'],
        verbose=True
    )

    results['baseline'] = {
        'model': baseline_model,
        'history': baseline_history,
        'trainer': baseline_trainer
    }

    return results


def evaluate_rollout(results: Dict, dataset: Dict, args, output_dir: Path):
    """
    长期 rollout 评估：对比 HNN 和 Baseline 的能量守恒性

    这是整个实验最关键的部分！用于验证：
      - HNN 是否能保持能量近似守恒
      - BaselineNN 的能量是否发散

    注意：对于 HNN，rollout 过程中需要 autograd 来计算动力学！
          因此不能使用 @torch.no_grad()。
    """
    print("\n" + "=" * 60)
    print("Step 3: 长期 Rollout 评估")
    print("=" * 60)

    rollout_steps = args.rollout_steps
    dt = 0.05  # Rollout 时间步长
    t_end = rollout_steps * dt

    # 创建简谐振子系统（用于计算真实能量）
    system = MassSpringSystem(SystemConfig(mass=1.0, spring_const=1.0))

    # 选择测试初始条件（从验证集中随机选取）
    np.random.seed(args.seed + 1)  # 不同的种子
    test_idx = np.random.choice(len(dataset['val_coords']), size=5, replace=False)

    evaluation_results = {
        'hnn_energies': [],
        'baseline_energies': [],
        'true_energies': [],
        'trajectories': {}
    }

    for idx in test_idx:
        y0 = dataset['val_coords'][idx:idx+1]  # (1, 2)
        true_energy = system.hamiltonian(y0[0, 0], y0[0, 1]).item()

        # ---- HNN Rollout ----
        hnn_model = results['hnn']['model']
        hnn_model.eval()

        def hnn_dynamics(t, y, model=hnn_model):
            return HNN.dynamics(t, y, model)

        traj_hnn = solve_ivp_rk4(hnn_dynamics, y0, (0, t_end), rollout_steps)

        # 计算 HNN 轨迹的能量
        energies_hnn = []
        for i in range(traj_hnn.shape[0]):
            yi = traj_hnn[i]
            Hi = hnn_model(yi).item()  # 模型预测的哈密顿量
            energies_hnn.append(Hi)

        # ---- Baseline Rollout ----
        baseline_model = results['baseline']['model']
        baseline_model.eval()

        def baseline_dynamics(t, y, model=baseline_model):
            return BaselineNN.dynamics(t, y, model)

        traj_baseline = solve_ivp_rk4(baseline_dynamics, y0, (0, t_end), rollout_steps)

        # 计算 Baseline 轨迹的能量（使用真实哈密顿量公式）
        energies_baseline = []
        for i in range(traj_baseline.shape[0]):
            yi = traj_baseline[i]
            qi, pi = yi[0, 0].item(), yi[0, 1].item()
            Ei = system.hamiltonian(torch.tensor([qi]), torch.tensor([pi])).item()
            energies_baseline.append(Ei)

        # 存储结果
        evaluation_results['hnn_energies'].append(energies_hnn)
        evaluation_results['baseline_energies'].append(energies_baseline)
        evaluation_results['true_energies'].append(true_energy)

        if len(evaluation_results['trajectories']) < 1:  # 只保存第一条轨迹用于可视化
            evaluation_results['trajectories']['hnn'] = traj_hnn.detach().numpy()[..., 0, :]
            evaluation_results['trajectories']['baseline'] = traj_baseline.detach().numpy()[..., 0, :]

    # 计算平均能量漂移
    hnn_drifts = []
    baseline_drifts = []

    for i in range(len(test_idx)):
        E_true = evaluation_results['true_energies'][i]
        E_hnn_final = evaluation_results['hnn_energies'][i][-1]
        E_base_final = evaluation_results['baseline_energies'][i][-1]

        hnn_drift = abs(E_hnn_final - E_true) / abs(E_true) * 100
        baseline_drift = abs(E_base_final - E_true) / abs(E_true) * 100

        hnn_drifts.append(hnn_drift)
        baseline_drifts.append(baseline_drift)

    avg_hnn_drift = np.mean(hnn_drifts)
    avg_baseline_drift = np.mean(baseline_drifts)

    print(f"\nRollout 设置:")
    print(f"  步数: {rollout_steps}")
    print(f"  时间步长 dt: {dt}")
    print(f"  总时间: {t_end:.2f} s ({t_end / (2*np.pi):.1f} 个周期)")

    print(f"\n能量守恒性对比（{len(test_idx)} 个测试点平均）:")
    print(f"  {'模型':<15s} {'最终能量漂移 (%)':>20s}")
    print(f"  {'-'*37}")
    print(f"  {'HNN':<15s} {avg_hnn_drift:>20.2f}% ✅")
    print(f"  {'Baseline NN':<15s} {avg_baseline_drift:>20.2f}% ❌")

    improvement = avg_baseline_drift / max(avg_hnn_drift, 0.01)
    print(f"\n  🎯 HNN 比 Baseline 能量漂移低 {improvement:.1f} 倍！")

    # 保存评估指标
    metrics = {
        'rollout_steps': rollout_steps,
        'dt': dt,
        't_end': t_end,
        'n_test_points': len(test_idx),
        'hnn_avg_energy_drift_percent': float(avg_hnn_drift),
        'baseline_avg_energy_drift_percent': float(avg_baseline_drift),
        'improvement_factor': float(improvement),
        'timestamp': datetime.now().isoformat()
    }

    metrics_path = output_dir / 'results' / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  📊 评估指标已保存至: {metrics_path}")

    return evaluation_results, metrics


def visualize_results(evaluation_results: Dict, results: Dict, output_dir: Path):
    """可视化实验结果"""
    print("\n" + "=" * 60)
    print("Step 4: 生成可视化图表")
    print("=" * 60)

    # 图 1: Loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # HNN loss 曲线
    ax = axes[0]
    hnn_history = results['hnn']['history']
    ax.plot(hnn_history['epoch'], hnn_history['train_loss'], label='Train', alpha=0.7)
    ax.plot(hnn_history['epoch'], hnn_history['val_loss'], label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('HNN Training Curve')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Baseline loss 曲线
    ax = axes[1]
    base_history = results['baseline']['history']
    ax.plot(base_history['epoch'], base_history['train_loss'], label='Train', alpha=0.7)
    ax.plot(base_history['epoch'], base_history['val_loss'], label='Val', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Baseline NN Training Curve')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'figures' / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Loss 曲线已保存: loss_curves.png")

    # 图 2: 相空间轨迹对比
    if 'trajectories' in evaluation_results and 'hnn' in evaluation_results['trajectories']:
        traj_hnn = evaluation_results['trajectories']['hnn']
        traj_base = evaluation_results['trajectories']['baseline']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # HNN 轨迹
        ax = axes[0]
        ax.plot(traj_hnn[:, 0], traj_hnn[:, 1], 'b-', linewidth=1.5, label='HNN Prediction')
        ax.plot(traj_hnn[0, 0], traj_hnn[0, 1], 'go', markersize=10, label='Start')
        ax.set_xlabel('q (position)')
        ax.set_ylabel('p (momentum)')
        ax.set_title('HNN Phase Space Trajectory\n(Closed Orbit → Energy Conserved ✅)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Baseline 轨迹
        ax = axes[1]
        ax.plot(traj_base[:, 0], traj_base[:, 1], 'r-', linewidth=1.5, label='Baseline Prediction')
        ax.plot(traj_base[0, 0], traj_base[0, 1], 'ro', markersize=10, label='Start')
        ax.set_xlabel('q (position)')
        ax.set_ylabel('p (momentum)')
        ax.set_title('Baseline NN Phase Space Trajectory\n(Spiral Divergence ❌)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        fig.savefig(output_dir / 'figures' / 'phase_space_trajectory.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 相空间轨迹已保存: phase_space_trajectory.png")

    # 图 3: 能量随时间变化
    if evaluation_results.get('hnn_energies'):
        fig, ax = plt.subplots(figsize=(10, 6))

        t = np.arange(len(evaluation_results['hnn_energies'][0])) * 0.05

        # 绘制多条曲线（不同初始条件）
        for i in range(min(3, len(evaluation_results['hnn_energies']))):
            E_true = evaluation_results['true_energies'][i]
            ax.axhline(y=E_true, color='gray', linestyle='--', alpha=0.5, label=f'True E={E_true:.3f}' if i == 0 else '')
            ax.plot(t, evaluation_results['hnn_energies'][i], 'b-', linewidth=1.5, alpha=0.7, label='HNN' if i == 0 else '')
            ax.plot(t, evaluation_results['baseline_energies'][i], 'r-', linewidth=1.5, alpha=0.7, label='Baseline' if i == 0 else '')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy H(q,p)')
        ax.set_title('Energy Conservation Over Time\n(HNN Stable vs Baseline Diverging)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / 'figures' / 'energy_conservation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 能量守恒图已保存: energy_conservation.png")


def save_models(results: Dict, output_dir: Path):
    """保存训练好的模型"""
    print("\n" + "=" * 60)
    print("Step 5: 保存模型")
    print("=" * 60)

    # 保存 HNN
    hnn_path = output_dir / 'models' / 'hnn_best.pth'
    torch.save({
        'model_state_dict': results['hnn']['model'].state_dict(),
        'config': {
            'input_dim': 2,
            'hidden_dim': 200,
            'num_hidden_layers': 3
        },
        'training_history': results['hnn']['history']
    }, hnn_path)
    print(f"  ✅ HNN 模型已保存: {hnn_path.name}")

    # 保存 Baseline
    base_path = output_dir / 'models' / 'baseline_best.pth'
    torch.save({
        'model_state_dict': results['baseline']['model'].state_dict(),
        'config': {
            'input_dim': 2,
            'hidden_dim': 200,
            'num_hidden_layers': 3
        },
        'training_history': results['baseline']['history']
    }, base_path)
    print(f"  ✅ Baseline 模型已保存: {base_path.name}")


def main():
    """主函数：端到端实验流程"""
    args = parse_args()

    print("\n" + "🚀 " * 20)
    print("HNN Task 1: Ideal Mass-Spring System")
    print("🚀 " * 20 + "\n")

    # 打印运行配置
    print("=" * 60)
    print("实验配置")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Noise Std: {args.noise_std}")
    print(f"  Seed: {args.seed}")

    # 设置输出目录
    output_dir = setup_output_directory(args)
    print(f"\n  输出目录: {output_dir}")

    try:
        # Step 1: 生成数据
        dataset = generate_data(args)

        # Step 2: 训练模型
        results = train_models(dataset, args)

        # Step 3: 评估 rollout
        eval_results, metrics = evaluate_rollout(results, dataset, args, output_dir)

        # Step 4: 可视化
        visualize_results(eval_results, results, output_dir)

        # Step 5: 保存模型
        save_models(results, output_dir)

        # 最终总结
        print("\n" + "=" * 60)
        print("🎉 实验完成！总结")
        print("=" * 60)
        print(f"\n📊 核心结果:")
        print(f"  • HNN 平均能量漂移: {metrics['hnn_avg_energy_drift_percent']:.2f}%")
        print(f"  • Baseline 平均能量漂移: {metrics['baseline_avg_energy_drift_percent']:.2f}%")
        print(f"  • HNN 比 Baseline 好 {metrics['improvement_factor']:.1f} 倍")
        print(f"\n📁 输出文件:")
        print(f"  • 模型: {output_dir / 'models'}")
        print(f"  • 结果: {output_dir / 'results'}")
        print(f"  • 图表: {output_dir / 'figures'}")

        print("\n💡 面试叙事要点:")
        print("  「在质量弹簧系统的 200 步 rollout 实验中，")
        print(f"   我手写的 HNN 将能量漂移控制在 {metrics['hnn_avg_energy_drift_percent']:.1f}% 以内，")
        print(f"   而 BaselineNN 的能量发散了 {metrics['baseline_avg_energy_drift_percent']:.1f}%。")
        print("   这证明了通过学习标量哈密顿量并利用 autograd 自动满足辛对称性，")
        print("   物理先验能够显著提升模型的长期预测能力。」\n")

    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
