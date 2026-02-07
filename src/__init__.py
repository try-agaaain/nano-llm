import os
# 加载环境变量配置（默认使用kaggle模式）
from src.utils.env_loader import load_secrets
load_secrets(mode=os.getenv("CONFIG_MODE", "kaggle"))