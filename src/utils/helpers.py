"""
Helper utilities: random seed, device selection, logging.

Corresponds to: 项目基础设施，非论文特定内容。

功能:
  1. set_seed(seed): 设置全局随机种子
     - torch.manual_seed(seed)
     - torch.cuda.manual_seed_all(seed)
     - numpy.random.seed(seed)
     - random.seed(seed)
     - torch.backends.cudnn.deterministic = True

  2. get_device(): 自动选择计算设备
     - 优先 CUDA, 其次 MPS (Apple Silicon), 最后 CPU

  3. setup_logging(level): 配置日志格式
     - 格式: [%(asctime)s] %(levelname)s - %(message)s

  4. load_config(yaml_path): 加载 YAML 配置文件

  5. save_checkpoint(model, optimizer, epoch, path): 保存模型检查点

  6. load_checkpoint(model, optimizer, path): 加载模型检查点

陷阱:
  - 设置 deterministic=True 会降低 GPU 训练速度
  - MPS 后端在某些 PyTorch 版本中可能不支持所有操作
  - 保存 checkpoint 时应同时保存模型结构和权重
"""
pass
