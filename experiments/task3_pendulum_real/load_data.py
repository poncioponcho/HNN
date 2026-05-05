"""
Task 3: Load Schmidt & Lipson real pendulum data.

Corresponds to: 论文 Section 4.3, Figure 3.
加载真实实验数据集，验证 HNN 在非合成数据上的表现。

数据来源:
  - Schmidt & Lipson (2009) 的真实单摆实验数据
  - 包含角度 theta 和角速度 theta_dot 的观测值
  - 下载地址: 需从论文补充材料获取，放置于 data/raw/

数据加载流程:
  1. 读取 CSV 文件 (theta, theta_dot)
  2. 归一化到 [-1, 1] 范围
  3. 通过有限差分计算时间导数
  4. 划分训练/测试集

输出: processed/pendulum_real_data.pt

陷阱:
  - 真实数据含测量噪声和可能的摩擦
  - HNN 无法建模摩擦，可能导致系统性偏差
  - 数据可能需要手动下载，不在仓库中
  - 归一化方式影响训练稳定性
"""
pass
