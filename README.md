# Hamiltonian Neural Networks — 复现指南

> **论文**: Hamiltonian Neural Networks
> **作者**: Sam Greydanus, Misko Dzamba, Marc Finzi
> **arXiv**: https://arxiv.org/abs/1906.01563
> **NeurIPS 2019**

---

## 1. 核心思想

传统神经网络学习动力学时，无法保证能量守恒等物理不变量。HNN 的核心洞察是：

**用神经网络参数化哈密顿量 H(q, p)，再通过自动微分得到哈密顿方程的右端项，从而在结构上保证能量守恒。**

---

## 2. 核心公式对照

### 哈密顿方程

$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

### HNN 损失函数（论文 Section 3, Eq. 5）

$$\mathcal{L}_{HNN} = \left\| \frac{\partial H_\theta}{\partial p} - \dot{q} \right\|^2 + \left\| \frac{\partial H_\theta}{\partial q} + \dot{p} \right\|^2$$

其中 $\dot{q}$ 和 $\dot{p}$ 是训练数据中的时间导数（通过有限差分估计）。

### 基线 NN 损失（论文 Section 3）

$$\mathcal{L}_{baseline} = \left\| f_\theta(q, p) - (\dot{q}, \dot{p}) \right\|^2$$

基线网络直接输出向量 $(\dot{q}, \dot{p})$，不经过哈密顿结构。

### CC 辅助损失（Task 5, 论文 Section 5, Eq. 9）

$$\mathcal{L}_{CC} = \left\| z_p - (z_q^{t} - z_q^{t+1}) \right\|^2$$

其中 $z_q$ 和 $z_p$ 分别是 latent 的前半和后半部分。CC loss 确保 $z_p$ 编码速度信息。

### Task 5 联合损失

$$\mathcal{L}_{total} = \mathcal{L}_{AE} + \mathcal{L}_{HNN} + \mathcal{L}_{CC}$$

---

## 3. 环境安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 注意：Gym Pendulum-v0 仅在 gym==0.21.0 中可用
# 新版 gym 中已改为 Pendulum-v1
```

---

## 4. 复现路线图

### Step 1: Task 1 — Ideal Mass-Spring（验证 HNN 守恒能量）

```bash
cd experiments/task1_mass_spring
python generate_data.py    # 生成含噪轨迹
python train_hnn.py        # 训练 HNN
python train_baseline.py   # 训练基线
python evaluate.py         # 对比长期 rollout 的能量守恒
```

**预期结果**: HNN 的总能量在长时间 rollout 后保持近似恒定，基线 NN 的能量发散。

### Step 2: Task 2 — Ideal Pendulum（非线性系统）

```bash
cd experiments/task2_pendulum_ideal
python generate_data.py
python train_hnn.py
python train_baseline.py
python evaluate.py
```

**预期结果**: HNN 在相空间中产生闭合轨道，基线轨道逐渐螺旋。

### Step 3: Task 3 — Real Pendulum（真实数据验证）

```bash
cd experiments/task3_pendulum_real
python load_data.py        # 加载 Schmidt & Lipson 数据集
python train_hnn.py
python train_baseline.py
python evaluate.py
```

**注意**: 需要下载 Schmidt & Lipson 数据集，放置于 `data/raw/` 目录。

### Step 4: Task 4 — Two-Body Problem（高维扩展性）

```bash
cd experiments/task4_two_body
python generate_data.py
python train_hnn.py
python train_baseline.py
python evaluate.py
```

**预期结果**: HNN 在 8 维相空间中仍能守恒能量，基线发散更快。

### Step 5: Task 5 — Pixel Pendulum（像素输入 + AE + HNN）

```bash
cd experiments/task5_pixel_pendulum
python generate_data.py       # Gym Pendulum-v0 生成像素数据
python train_autoencoder.py   # 预训练 AE
python train_joint.py         # 联合训练 AE + HNN
python evaluate.py            # 像素重建 + 角度对比
```

---

## 5. 关键实现提示

### HNN 的 forward 输出标量

```python
# HNN forward: 输入 (q, p) -> 输出 标量 H
# loss 不在 forward 中计算，而是在外部通过 autograd 获取输入梯度
def forward(self, x):
    return self.mlp(x)  # 输出形状: (batch_size, 1)

# 获取哈密顿方程右端项:
H = model(x)                          # 标量
dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
dq_dt = dH[:, 1:]                     # ∂H/∂p
dp_dt = -dH[:, :1]                    # -∂H/∂q
```

### 基线 NN 直接输出向量

```python
# Baseline forward: 输入 (q, p) -> 输出 (dq/dt, dp/dt)
def forward(self, x):
    return self.mlp(x)  # 输出形状: (batch_size, 2)
```

### 积分器

```python
# 使用 scipy.integrate.solve_ivp (RK45)
from scipy.integrate import solve_ivp
sol = solve_ivp(fun, t_span, y0, method='RK45', rtol=1e-9, atol=1e-9)
```

### Task 5 的 latent 维度

- Autoencoder 的 latent 维度**必须为偶数**
- 前一半是 $z_q$（广义坐标），后一半是 $z_p$（广义动量）
- 需要两帧拼接作为输入（28×28×2），因为单帧无法观测速度

### 陷阱提醒

1. **HNN 无法建模摩擦/耗散**：哈密顿结构假设能量守恒，含摩擦的系统需要扩展（如 Dissipative HNN）
2. **有限差分估计导数**：训练标签 $\dot{q}, \dot{p}$ 通过 $(x_{t+1} - x_t) / \Delta t$ 估计，噪声会被放大
3. **Task 5 需要两帧输入**：单帧像素无法推断角速度，必须拼接相邻两帧
4. **数据归一化**：训练前需对 $(q, p)$ 做归一化，否则梯度不稳定

---

## 6. 论文超参数汇总

| 参数 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------|--------|--------|--------|--------|--------|
| 网络层数 | 3 | 3 | 3 | 3 | AE: 4层 |
| 隐单元数 | 200 | 200 | 200 | 200 | 200 |
| 激活函数 | tanh | tanh | tanh | tanh | ReLU (AE) |
| 学习率 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| 训练步数 | — | — | — | 10k | AE: 5k, Joint: 5k |
| 轨迹数 | 25 | 25 | — | 1000 | 200 |
| 每轨迹观测数 | 30 | 30 | — | 50 | 100 |
| 噪声 σ | 0.1 | 0.1 | — | 0.05 | — |
| 优化器 | Adam | Adam | Adam | Adam | Adam |

---

## 7. 项目结构

```
hamiltonian-nn/
├── configs/          # 各任务 YAML 超参配置
├── data/             # 数据目录 (gitignored)
├── src/
│   ├── models/       # HNN, Baseline NN, Autoencoder
│   ├── physics/      # 解析哈密顿量 & 积分器
│   ├── training/     # 训练循环 & 损失函数
│   ├── evaluation/   # 评估指标 & 可视化
│   └── utils/        # 工具函数
├── experiments/      # 5 个 Task 的独立实验脚本
├── notebooks/        # 交互式 Demo
└── tests/            # 单元测试
```
