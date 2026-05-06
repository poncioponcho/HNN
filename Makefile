# Hamiltonian Neural Networks - Makefile
# 
# 一键执行命令，用于简化实验流程
# 
# 使用方法:
#   make task1           # 运行 Task 1 完整实验（默认参数）
#   make task1-custom     # 使用自定义参数运行 Task 1
#   make task1-data       # 仅生成数据
#   make task1-train      # 仅训练模型
#   make task1-evaluate   # 仅评估（需要先有模型）
#   make test             # 运行所有单元测试
#   make clean            # 清理生成的文件
#   make status           # 显示项目状态
#   make help             # 显示帮助信息

.PHONY: help status test task1 task1-data task1-train task1-evaluate task1-custom clean

# 默认配置（可通过命令行覆盖）
EPOCHS ?= 1000
HIDDEN_DIM ?= 256
N_LAYERS ?= 3
BATCH_SIZE ?= 32
LEARNING_RATE ?= 0.001
NOISE_STD ?= 0.05
ROLLOUT_STEPS ?= 200
SEED ?= 42

# 目录配置
EXPERIMENT_DIR := experiments/task1_mass_spring
OUTPUT_DIR := outputs/task1_mass_spring
PYTHON := python

# ==================== 帮助目标 ====================

help:
	@echo ""
	@echo "🔬 HNN Project - Makefile Help"
	@echo "============================"
	@echo ""
	@echo "可用目标:"
	@echo "  make task1           运行 Task 1 完整实验（默认参数）"
	@echo "  make task1-custom     使用自定义参数运行 Task 1"
	@echo "  make task1-data       仅生成训练数据"
	@echo "  make task1-train      仅训练 HNN 和 BaselineNN"
	@echo "  make task1-evaluate   仅执行 rollout 评估"
	@echo "  make test             运行所有单元测试"
	@echo "  make clean            清理生成的文件"
	@echo "  make status           显示项目状态"
	@echo "  make help             显示本帮助信息"
	@echo ""
	@echo "自定义参数示例:"
	@echo "  make task1-custom EPOCHS=500 ROLLOUT_STEPS=100"
	@echo "  make task1-custom HIDDEN_DIM=128 NOISE_STD=0.1"
	@echo ""
	@echo "当前默认参数:"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  HIDDEN_DIM=$(HIDDEN_DIM)"
	@echo "  N_LAYERS=$(N_LAYERS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  LEARNING_RATE=$(LEARNING_RATE)"
	@echo "  NOISE_STD=$(NOISE_STD)"
	@echo "  ROLLOUT_STEPS=$(ROLLOUT_STEPS)"
	@echo "  SEED=$(SEED)"

status:
	@echo ""
	@echo "📊 项目状态报告"
	@echo "====================="
	@echo ""
	@echo "📁 Git 状态:"
	@git status --short | head -10 || echo "   (不是 Git 仓库)"
	@echo ""
	@echo "📝 最新提交:"
	@git log --oneline -3 || echo "   (无提交历史)"
	@echo ""
	@echo "📦 代码统计:"
	@find src/ tests/ experiments/ -name "*.py" -type f | xargs wc -l | tail -1
	@echo ""
	@echo "🧪 测试状态:"
	@if [ -d ".pytest_cache" ]; then \
		echo "   已有 pytest 缓存 (可运行 'make test')"; \
	else \
		echo "   尚未运行过测试"; \
	fi
	@echo ""
	@echo "📂 实验输出:"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		find $(OUTPUT_DIR) -name "*.png" 2>/dev/null | wc -l | xargs -I {} echo "   图表文件数: {}"; \
		find $(OUTPUT_DIR) -name "*.pth" 2>/dev/null | wc -l | xargs -I {} echo "   模型文件数: {}"; \
	else \
		echo "   (暂无实验输出)"; \
	fi
	@echo ""
	@echo "⚡ 当前状态: ✅ Task 1-2 Completed | 🔄 Task 3-5 Pending"

# ==================== 测试目标 ====================

test:
	@echo ""
	@echo "🧪 运行单元测试..."
	@echo "=================="
	$(PYTHON) tests/test_phase1_models.py
	$(PYTHON) tests/test_phase2_integrator_and_loss.py
	@echo ""
	@echo "✅ 所有测试通过！"

# ==================== Task 1 目标 ====================

task1: task1-data task1-train task1-evaluate
	@echo ""
	@echo "🎉 Task 1 完整实验流程已完成！"
	@echo ""
	@echo "📊 结果位置: $(OUTPUT_DIR)/run_*/"
	@echo "   ├── models/      训练好的模型权重 (.pth)"
	@echo "   ├── results/     数值评估指标 (.json)"
	@echo "   └── figures/     可视化图表 (.png)"
	@echo ""
	@echo "查看结果:"
	@ls -lh $(OUTPUT_DIR)/run_*/figures/ 2>/dev/null || echo "   (请先运行实验)"

task1-custom:
	@echo ""
	@echo "🚀 运行 Task 1 实验（自定义参数）..."
	@echo "==============================="
	@echo "参数配置:"
	@echo "  Epochs:        $(EPOCHS)"
	@echo "  Hidden Dim:    $(HIDDEN_DIM)"
	@echo "  Layers:        $(N_LAYERS)"
	@echo "  Batch Size:    $(BATCH_SIZE)"
	@echo "  Learning Rate: $(LEARNING_RATE)"
	@echo "  Noise Std:     $(NOISE_STD)"
	@echo "  Rollout Steps: $(ROLLOUT_STEPS)"
	@echo "  Seed:          $(SEED)"
	@echo ""
	cd $(EXPERIMENT_DIR) && \
	$(PYTHON) run.py \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--hidden-dim $(HIDDEN_DIM) \
		--n-layers $(N_LAYERS) \
		--lr $(LEARNING_RATE) \
		--noise-std $(NOISE_STD) \
		--rollout-steps $(ROLLOUT_STEPS) \
		--seed $(SEED)
	@echo ""

task1-data:
	@echo ""
	@echo "📥 生成 Task 1 训练数据..."
	@echo "==========================="
	@echo "参数: noise_std=$(NOISE_STD), seed=$(SEED)"
	@echo ""
	@mkdir -p $(OUTPUT_DIR)
	cd $(EXPERIMENT_DIR) && \
	$(PYTHON) -c "from src.physics.systems import create_mass_spring_data; import torch; dataset = create_mass_spring_data(n_trajectories=25, n_points=30, noise_std=$(NOISE_STD), seed=$(SEED)); print(f'✅ 数据生成完成!'); print(f'   训练集: {dataset[\"train_coords\"].shape[0]}, 验证集: {dataset[\"val_coords\"].shape[0]}'); torch.save(dataset, '../$(OUTPUT_DIR)/task1_dataset.pt'); print(f'💾 已保存至: $(OUTPUT_DIR)/task1_dataset.pt')"

task1-train:
	@echo ""
	@echo "🎓 训练 Task 1 模型..."
	@echo "======================="
	@echo "参数: epochs=$(EPOCHS), hidden_dim=$(HIDDEN_DIM), layers=$(N_LAYERS)"
	@echo ""
	cd $(EXPERIMENT_DIR) && \
	$(PYTHON) run.py \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--hidden-dim $(HIDDEN_DIM) \
		--n-layers $(N_LAYERS) \
		--lr $(LEARNING_RATE) \
		--noise-std $(NOISE_STD) \
		--rollout-steps 10 \
		--seed $(SEED)

task1-evaluate:
	@echo ""
	@echo "📈 执行 Task 1 Rollout 评估..."
	@echo "========================="
	@echo "参数: rollout_steps=$(ROLLOUT_STEPS)"
	@echo ""
	@if [ ! -f "$(OUTPUT_DIR)/task1_dataset.pt" ]; then \
		echo "❌ 错误: 未找到训练数据，请先运行 'make task1-data'"; \
		exit 1; \
	fi
	cd $(EXPERIMENT_DIR) && \
	$(PYTHON) run.py \
		--epochs 1 \
		--batch-size $(BATCH_SIZE) \
		--hidden-dim $(HIDDEN_DIM) \
		--n-layers $(N_LAYERS) \
		--lr $(LEARNING_RATE) \
		--noise-std $(NOISE_STD) \
		--rollout-steps $(ROLLOUT_STEPS) \
		--seed $(SEED)

# ==================== 清理目标 ====================

clean:
	@echo ""
	@echo "🧹 清理生成文件..."
	@echo "==================="
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		rm -rf $(OUTPUT_DIR); \
		echo "✅ 已删除实验输出目录: $(OUTPUT_DIR)/"; \
	else \
		echo "ℹ️  无需清理（输出目录不存在)"; \
	fi
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 清理完成！"

# ==================== 开发者工具 ====================

# 格式化代码（可选，需要安装 black/isort）
format:
	@echo "🎨 格式化代码..."
	black src/ tests/ experiments/
	isort src/ tests/ experiments/

# 类型检查（需要 mypy）
typecheck:
	@echo "🔍 类型检查..."
	mypy src/ --ignore-missing-imports

# 代码复杂度检查（需要 radon）
complexity:
	@echo "📊 代码复杂度分析..."
	radon cc src/ -a -s

# 安全性检查（需要 bandit）
security:
	@echo "🔒 安全性检查..."
	bandit -r src/

# 完整的 CI 检查
ci: test typecheck format
	@echo ""
	@echo "✅ CI 检查全部通过！"
