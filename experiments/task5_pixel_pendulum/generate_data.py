"""
Task 5: Generate pixel pendulum data using OpenAI Gym.

Corresponds to: 论文 Section 5, Task 5.
使用 Gym Pendulum-v0 环境生成像素观测数据。

数据生成流程:
  1. 初始化 Gym Pendulum-v0 环境
  2. 限制初始角度在 [-pi/6, pi/6] 范围内
  3. 渲染 28x28 灰度图像
  4. 拼接相邻两帧: (28, 28, 2)
  5. 记录真实角度和角速度作为评估标签
  6. 生成 200 条轨迹，每条 100 帧

输出: processed/pixel_pendulum_data.pt
  - frames: (N_traj, T, 2, 28, 28) 两帧拼接的像素
  - angles: (N_traj, T) 真实角度
  - angular_velocities: (N_traj, T) 真实角速度

参数 (论文):
  - Gym Pendulum-v0, 200 trajectories, 100 frames, max_angle=pi/6
  - 图像: 28x28x1 灰度, 两帧拼接为 28x28x2

陷阱:
  - Pendulum-v0 仅在 gym==0.21.0 中可用
  - 新版 gym 改为 Pendulum-v1，API 略有不同
  - 渲染图像需要调用 env.render(mode='rgb_array') 并转换为灰度
  - 两帧拼接是必须的，单帧无法推断角速度
  - max_angle=pi/6 限制了运动范围，使问题更易学习
"""
pass
