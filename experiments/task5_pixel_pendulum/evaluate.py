"""
Task 5: Evaluate pixel pendulum (AE + HNN joint model).

Corresponds to: 论文 Section 5, Figure 5 & 6.
像素重建 + 角度曲线对比评估。

评估流程:
  1. 像素重建质量:
     - 输入像素 → Encoder → latent → Decoder → 重建像素
     - 计算 MSE 重建误差
     - 可视化: 原始 vs 重建像素并排对比

  2. Latent 动力学:
     - 在 latent 空间中使用 HNN rollout
     - 将 latent 轨迹映射回角度空间
     - 与真实角度对比

  3. 角度曲线对比:
     - 真实角度 vs 预测角度随时间变化
     - HNN 预测 vs Baseline 预测

可视化:
  - 像素重建对比 (Figure 6)
  - 角度曲线对比
  - Latent 相空间轨道

预期结果 (论文):
  - AE 能较好地重建像素
  - HNN 在 latent 空间中守恒能量
  - 预测角度与真实角度趋势一致

陷阱:
  - latent → 角度的映射需要标定
  - z_q 不一定直接等于角度，可能需要线性变换
  - 像素重建质量受 AE 容量限制
"""
pass
