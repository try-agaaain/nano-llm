# nano-llm Makefile
.DEFAULT_GOAL := help

ifeq ($(OS),Windows_NT)
	CP = powershell -NoProfile -Command "Copy-Item -Force"
else
	CP = cp
endif

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
	@echo "  make kpush        Upload secrets & push notebook to Kaggle"
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
	uv run src/train.py

infer:
	@echo "Running inference..."
	uv run src/inference.py

test:
	@echo "Running tests..."
	uv run pytest src/test_*.py -v

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ .pytest_cache *.pyc
kinit:
	@echo "Initializing Kaggle metadata..."
	uv run kaggle kernels init -p kaggle/notebook
	@echo "Metadata created: kaggle/notebook/kernel-metadata.json"

kpush:
	@echo "Copying config.yaml to kaggle/secrets..."
	$(CP) config.yaml kaggle/secrets/config.yaml
	@echo "Uploading secrets & pushing notebook to Kaggle..."
	uv run -m src.utils.upload_kaggle

kpull:
	@echo "Pulling notebook from Kaggle..."
	uv run kaggle kernels pull team317/nano-llm -p ./kaggle/notebook -m
	@echo "Notebook pulled"

kstatus:
	@echo "Checking notebook status..."
	uv run -m kaggle kernels status team317/nano-llm

koutput:
	@echo "Getting notebook output..."
	mkdir -p kaggle/output
