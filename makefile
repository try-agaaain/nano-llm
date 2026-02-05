# nano-llm 项目 Makefile
# 环境设置
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "NanoLLM 项目 - 可用命令:"
	@echo ""
	@echo "  环境管理:"
	@echo "  make setup        初始化开发环境"
	@echo "  make install      安装依赖包"
	@echo ""
	@echo "  训练和推理:"
	@echo "  make train        训练模型"
	@echo "  make test         运行单元测试"
	@echo "  make infer        运行推理脚本"
	@echo ""
	@echo "  Kaggle 集成:"
	@echo "  make kinit        初始化 Kaggle Notebook 元数据"
	@echo "  make kpush        推送 Notebook 到 Kaggle"
	@echo "  make kpull        从 Kaggle 拉取 Notebook"
	@echo "  make kstatus      检查 Kaggle Notebook 状态"
	@echo "  make koutput      获取 Kaggle Notebook 输出"
	@echo ""
	@echo "  清理:"
	@echo "  make clean        清理生成文件"
	@echo ""

# 初始化环境
setup:
	@echo "初始化 Python 虚拟环境..."
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -q uv
	@echo "✅ 虚拟环境已创建"

# 安装依赖
install:
	@echo "安装项目依赖..."
	$(PIP) install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -q transformers datasets tokenizers tqdm wandb
	@echo "✅ 依赖安装完成"

# 训练模型
train:
	@echo "启动模型训练..."
	$(PYTHON) train.py

# 运行推理
infer:
	@echo "运行推理脚本..."
	$(PYTHON) inference.py

# 运行测试
test:
	@echo "运行测试..."
	$(PYTHON) -m pytest test_*.py -v

# 清理
clean:
	@echo "清理生成文件..."
	rm -rf __pycache__ .pytest_cache *.pyc
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 清理完成"

# Kaggle 集成命令
kinit:
	@echo "初始化 Kaggle Notebook 元数据..."
	$(PYTHON) -m kaggle kernels init -p kaggle
	@echo "✅ 元数据文件已创建: kaggle/kernel-metadata.json"

kpush:
	@echo "推送 Notebook 到 Kaggle..."
	uv run kaggle kernels push -p kaggle
	@echo "✅ Notebook 已推送到 Kaggle"

kpull:
	@echo "从 Kaggle 拉取 Notebook..."
	uv run kaggle kernels pull team317/nano-llm -p ./kaggle -m
	@echo "✅ Notebook 已拉取"

kstatus:
	@echo "检查 Kaggle Notebook 状态..."
	$(PYTHON) -m kaggle kernels status team317/train-llm

koutput:
	@echo "获取 Kaggle Notebook 输出..."
	mkdir -p kaggle/output
	$(PYTHON) -m kaggle kernels output team317/train-llm -p ./kaggle/output
	@echo "✅ 输出已保存到 kaggle/output"

.PHONY: help setup install train infer test clean kinit kpush kpull kstatus koutput