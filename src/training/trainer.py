"""
Generic training loop for HNN and baseline models.

Corresponds to: 论文 Section 3-5, 所有实验的训练阶段。
标准 PyTorch 训练循环，支持 HNN 和 BaselineNN 两种模型。

核心功能:
  - 支持 HNN（通过 autograd 的 loss）和 Baseline（直接回归）两种模式
  - Adam 优化器 + 可选学习率调度
  - train/val 划分 + 早停机制
  - 定期保存 checkpoint
  - TensorBoard 日志记录

训练流程:
  1. 从 DataLoader 采样 mini-batch: (coords, dcoords_dt)
  2. 前向传播:
     - HNN: 计算 H = model(coords), 然后通过 autograd 得到 ∂H/∂coords
     - Baseline: 直接计算 y_pred = model(coords)
  3. 计算损失 (见 losses.py)
  4. 反向传播 + 参数更新

默认超参数 (论文):
  - optimizer: Adam
  - learning_rate: 1e-3
  - batch_size: 64
  - n_epochs: 2000 (Task 1-3)

使用示例:
    >>> trainer = Trainer(model, loss_fn, config)
    >>> history = trainer.train(train_loader, val_loader)
    >>> trainer.save_checkpoint('best_model.pth')
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple, Callable, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class TrainingConfig:
    """训练配置参数

    Attributes:
        learning_rate: 初始学习率
        optimizer: 优化器类型 ('adam', 'sgd')
        batch_size: mini-batch 大小
        n_epochs: 最大训练轮数
        seed: 随机种子
        device: 计算设备 ('cpu' 或 'cuda')
        save_every: 每隔多少 epoch 保存一次 checkpoint
        early_stopping_patience: 早停耐心值（连续多少轮无改善就停止）
        grad_clip: 梯度裁剪阈值（None 表示不裁剪）
        lr_scheduler: 学习率调度策略 ('none', 'step', 'cosine')
        log_dir: 日志保存目录
    """
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    batch_size: int = 64
    n_epochs: int = 2000
    seed: int = 42
    device: str = 'cpu'
    save_every: int = 100
    early_stopping_patience: int = 100
    grad_clip: Optional[float] = None
    lr_scheduler: str = 'none'
    log_dir: str = 'logs'


class Trainer:
    """
    通用训练器：支持 HNN 和 BaselineNN 的端到端训练

    设计原则:
      - 与模型架构解耦：只依赖 forward() 接口
      - 支持自定义 loss 函数
      - 完整的日志和 checkpoint 机制
      - 遵循 PyTorch 最佳实践
    """

    def __init__(
            self,
            model: nn.Module,
            loss_fn: Callable,
            config: Optional[TrainingConfig] = None
    ):
        """
        初始化训练器

        Args:
            model: 要训练的模型（HNN 或 BaselineNN）
            loss_fn: 损失函数（hnn_loss 或 baseline_loss）
            config: 训练配置，如果为 None 则使用默认值
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config or TrainingConfig()

        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config.device == 'cuda'
            else 'cpu'
        )
        self.model.to(self.device)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }

        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """根据配置创建优化器"""
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[object]:
        """创建学习率调度器"""
        if self.config.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=500,
                gamma=0.1
            )
        elif self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_epochs
            )
        else:
            return None  # 不使用调度器

    def train_epoch(
            self,
            train_loader: DataLoader
    ) -> float:
        """
        训练一个 epoch

        流程:
          1. 将模型设为训练模式
          2. 遍历所有 mini-batch
          3. 对每个 batch:
             a. 将数据移到设备上
             b. 前向传播计算 loss
             c. 反向传播计算梯度
             d. （可选）梯度裁剪
             e. 更新参数
          4. 返回平均训练 loss

        Args:
            train_loader: 训练数据的 DataLoader

        Returns:
            avg_loss: 该 epoch 的平均训练 loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (coords, dcoords_dt) in enumerate(train_loader):
            # 移动数据到设备
            coords = coords.to(self.device)
            dcoords_dt = dcoords_dt.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播 + 计算 loss
            loss = self.loss_fn(self.model, coords, dcoords_dt)

            # 反向传播
            loss.backward()

            # 梯度裁剪（可选）
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            # 参数更新
            self.optimizer.step()

            # 累计 loss
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def validate(
            self,
            val_loader: DataLoader
    ) -> float:
        """
        验证模型性能

        注意：对于 HNN，不能使用 @torch.no_grad()，
        因为 loss 函数内部需要通过 autograd 计算梯度！
        这是 HNN 与普通 NN 的关键区别。

        Args:
            val_loader: 验证数据的 DataLoader

        Returns:
            avg_loss: 平均验证 loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for coords, dcoords_dt in val_loader:
            coords = coords.to(self.device)
            dcoords_dt = dcoords_dt.to(self.device)

            # 对于 HNN: 不能用 torch.no_grad()，需要 autograd
            # 对于 Baseline: 可以用 torch.no_grad() 加速
            # 这里统一不使用，保证兼容性
            with torch.enable_grad():
                loss = self.loss_fn(self.model, coords, dcoords_dt)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def train(
            self,
            train_coords: torch.Tensor,
            train_dcoords_dt: torch.Tensor,
            val_coords: torch.Tensor,
            val_dcoords_dt: torch.Tensor,
            verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        完整训练流程

        包含:
          - 创建 DataLoader
          - epoch 循环
          - 早停检查
          - checkpoint 保存
          - 学习率调度

        Args:
            train_coords: 训练集坐标 (N_train, 2n)
            train_dcoords_dt: 训练集导数 (N_train, 2n)
            val_coords: 验证集坐标 (N_val, 2n)
            val_dcoords_dt: 验证集导数 (N_val, 2n)
            verbose: 是否打印训练进度

        Returns:
            history: 训练历史字典 {'train_loss': [...], 'val_loss': [...]}
        """
        # 设置随机种子
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # 创建 DataLoader
        train_dataset = TensorDataset(train_coords, train_dcoords_dt)
        val_dataset = TensorDataset(val_coords, val_dcoords_dt)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        print(f"\n{'='*60}")
        print(f"开始训练")
        print(f"{'='*60}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练样本数: {len(train_dataset):,}")
        print(f"验证样本数: {len(val_dataset):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.n_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # 训练循环
        for epoch in range(1, self.config.n_epochs + 1):
            # 训练一个 epoch
            train_loss = self.train_epoch(train_loader)

            # 验证
            val_loss = self.validate(val_loader)

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'].append(epoch)

            # 打印进度
            if verbose and epoch % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d}/{self.config.n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
            else:
                self.patience_counter += 1

            # 定期保存 checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(
                    f'checkpoint_epoch_{epoch}.pth',
                    epoch,
                    val_loss
                )

            # 早停检查
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                print(f"   Best validation loss: {self.best_val_loss:.6f}")
                break

        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            print(f"\n✅ 已加载最佳模型 (epoch {self.best_model_state['epoch']}, "
                  f"val_loss={self.best_val_loss:.6f})")

        return self.history

    def save_checkpoint(
            self,
            filepath: str,
            epoch: int,
            val_loss: float
    ):
        """
        保存训练 checkpoint

        Args:
            filepath: 保存路径
            epoch: 当前 epoch
            val_loss: 当前验证 loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """
        加载训练 checkpoint

        Args:
            filepath: checkpoint 文件路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"✅ 已加载 checkpoint (epoch {checkpoint['epoch']})")


def create_data_loaders(
        dataset: Dict[str, torch.Tensor],
        batch_size: int = 64,
        train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    从数据集字典创建 DataLoader（便捷函数）

    Args:
        dataset: MassSpringSystem.generate_dataset() 的输出
        batch_size: batch 大小
        train_ratio: 训练集比例

    Returns:
        (train_loader, val_loader): 两个 DataLoader
    """
    train_dataset = TensorDataset(
        dataset['train_coords'],
        dataset['train_dcoords_dt']
    )
    val_dataset = TensorDataset(
        dataset['val_coords'],
        dataset['val_dcoords_dt']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == '__main__':
    """简单测试：验证训练器的基本功能"""
    print("=" * 60)
    print("测试: Trainer 训练循环")
    print("=" * 60)

    from src.models.hnn import HNN
    from src.models.baseline_nn import BaselineNN
    from src.training.losses import hnn_loss, baseline_loss
    from src.physics.systems import create_mass_spring_data

    # 生成小规模测试数据
    print("\n生成测试数据...")
    dataset = create_mass_spring_data(n_trajectories=5, n_points=20)
    print(f"  训练集: {dataset['train_coords'].shape}")
    print(f"  验证集: {dataset['val_coords'].shape}")

    # 测试 HNN 训练
    print("\n" + "-" * 40)
    print("测试 HNN 训练:")
    print("-" * 40)

    hnn_model = HNN(input_dim=2, hidden_dim=64, num_hidden_layers=2)
    hnn_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        n_epochs=100,  # 测试用少量 epochs
        save_every=50,
        early_stopping_patience=20
    )

    hnn_trainer = Trainer(hnn_model, hnn_loss, hnn_config)
    history = hnn_trainer.train(
        dataset['train_coords'],
        dataset['train_dcoords_dt'],
        dataset['val_coords'],
        dataset['val_dcoords_dt'],
        verbose=True
    )

    print(f"\n最终 Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"最终 Val Loss:   {history['val_loss'][-1]:.6f}")

    # 测试 BaselineNN 训练
    print("\n" + "-" * 40)
    print("测试 BaselineNN 训练:")
    print("-" * 40)

    baseline_model = BaselineNN(input_dim=2, hidden_dim=64, num_hidden_layers=2)
    baseline_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        n_epochs=100,
        save_every=50,
        early_stopping_patience=20
    )

    baseline_trainer = Trainer(baseline_model, baseline_loss, baseline_config)
    history_base = baseline_trainer.train(
        dataset['train_coords'],
        dataset['train_dcoords_dt'],
        dataset['val_coords'],
        dataset['val_dcoords_dt'],
        verbose=True
    )

    print(f"\n最终 Train Loss: {history_base['train_loss'][-1]:.6f}")
    print(f"最终 Val Loss:   {history_base['val_loss'][-1]:.6f}")

    print("\n✅ Trainer 测试通过！")
