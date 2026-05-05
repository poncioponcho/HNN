"""
Task 3: Train HNN on real pendulum data.

Corresponds to: 论文 Section 4.3, Figure 3.
在真实实验数据上训练 HNN。

模型架构: 与 Task 2 相同
  - 3 层 MLP, 200 隐单元, tanh 激活

训练配置:
  - optimizer: Adam, lr=1e-3
  - 从 configs/pendulum_real.yaml 读取超参数

输出: checkpoints/task3_hnn.pt

陷阱:
  - 真实数据含摩擦，HNN 假设守恒，可能有偏差
  - 论文指出 HNN 仍能学到近似的哈密顿量
  - 数据量可能较少，注意过拟合
"""
pass
