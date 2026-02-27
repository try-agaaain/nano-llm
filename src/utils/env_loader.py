"""环境变量加载模块 - 支持Kaggle和本地两种模式"""

import os
import yaml
from pathlib import Path
from typing import Literal


def load_secrets() -> None:
    """
    加载配置secrets并设置为环境变量
    
    Raises:
        ValueError: 如果配置文件不存在或无法加载secrets
    """
    config = {}

    # 获取项目根目录（当前文件所在目录）
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        raise ValueError(
            f"本地配置文件不存在: {config_path}\n"
            "请确保项目根目录下存在config.yaml文件"
        )
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        print(f"[OK] 配置从 {config_path} 加载成功")
    except Exception as e:
        raise ValueError(f"无法从 {config_path} 加载配置: {e}")
    
    # 设置环境变量
    if not config:
        raise ValueError("未能加载任何配置")
    
    for key, value in config['secrets'].items():
        if value:
            os.environ[key] = str(value)
            print(f"[OK] {key} 已设置为环境变量")
    
    print(f"[INFO] 共设置 {len(config['secrets'])} 个环境变量")
    print("=" * 70)

