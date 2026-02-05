# nano-llm Makefile
.DEFAULT_GOAL := help

help:
	@echo "NanoLLM Makefile Commands:"
	@echo ""
	@echo "  Training:"
	@echo "  make train        Run training"
	@echo "  make test         Run tests"
	@echo "  make infer        Run inference"
	@echo ""
	@echo "  Kaggle:"
	@echo "  make kinit        Init Kaggle metadata"
	@echo "  make kdataset     Upload secrets dataset"
	@echo "  make kpush        Push notebook to Kaggle"
	@echo "  make kpull        Pull notebook from Kaggle"
	@echo "  make kstatus      Check notebook status"
	@echo "  make koutput      Get notebook output"
	@echo ""
setup:
	@echo "Setting up uv run virtual environment..."
	uv run venv .venv
	@echo "Virtual environment ready"

install:
	@echo "Installing dependencies..."
	uv run pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	uv run pip install -q transformers datasets tokenizers tqdm wandb
	@echo "Dependencies installed"

train:
	@echo "Starting training..."
	uv run train.py

infer:
	@echo "Running inference..."
	uv run inference.py

test:
	@echo "Running tests..."
	uv run pytest test_*.py -v

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache *.pyc
kinit:
	@echo "Initializing Kaggle metadata..."
	uv run kaggle kernels init -p kaggle
	@echo "Metadata created: kaggle/kernel-metadata.json"

kdataset:
	@echo "Uploading secrets dataset to Kaggle..."
	uv run kaggle datasets create -p my-secrets
	@echo "Secrets dataset uploaded"

kpush:
	@echo "Pushing notebook to Kaggle..."
	uv run kaggle kernels push -p kaggle
	@echo "Notebook pushed to Kaggle"

kpull:
	@echo "Pulling notebook from Kaggle..."
	uv run kaggle kernels pull team317/nano-llm -p ./kaggle -m
	@echo "Notebook pulled"

kstatus:
	@echo "Checking notebook status..."
	uv run -m kaggle kernels status team317/nano-llm

koutput:
	@echo "Getting notebook output..."
	mkdir -p kaggle/output
	pythonlp setup install train infer test clean kinit kdataset kpush kpull kstatus koutput
