"""环境变量加载模块 - 支持Kaggle和本地两种模式"""

import os
import yaml
from pathlib import Path
from typing import Literal


def load_secrets(mode: Literal["kaggle", "local"] = "kaggle") -> None:
    """
    加载配置secrets并设置为环境变量
    
    Args:
        mode: 加载模式
            - "kaggle": 从Kaggle secrets或/kaggle/input/secrets/config.yaml加载（默认）
            - "local": 从项目目录下的config.yaml加载
    
    Raises:
        ValueError: 如果配置文件不存在或无法加载secrets
    """
    config = {}
    
    if mode == "kaggle":
        print(os.listdir("/kaggle/input"))
        config_path = Path("/kaggle/input/secrets/config.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                print(f"[OK] 配置从 {config_path} 加载成功")
            except Exception as e:
                raise ValueError(f"无法从 {config_path} 加载配置: {e}")
        else:
            raise ValueError(
                f"Kaggle模式下配置文件不存在: {config_path}\n"
                "请确保已上传secrets数据集到Kaggle"
            )
    
    elif mode == "local":
        # 本地模式：从项目目录的config.yaml加载
        print(f"[INFO] 使用本地模式加载配置...")
        
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
    
    else:
        raise ValueError(f"不支持的模式: {mode}。请使用 'kaggle' 或 'local'")
    
    # 设置环境变量
    if not config:
        raise ValueError("未能加载任何配置")
    
    for key, value in config.items():
        if value:
            os.environ[key] = str(value)
            print(f"[OK] {key} 已设置为环境变量")
    
    print(f"[INFO] 共设置 {len(config)} 个环境变量")
    print("=" * 70)


def get_secret(key: str, default: str = None) -> str:
    """
    从环境变量获取secret
    
    Args:
        key: 配置键名
        default: 默认值（如果环境变量不存在）
    
    Returns:
        配置值
    
    Raises:
        ValueError: 如果配置不存在且没有提供默认值
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(
            f"环境变量 {key} 未设置。请先调用 load_secrets() 加载配置"
        )
    return value


if __name__ == "__main__":
    # 测试代码
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    print(f"测试环境变量加载（模式: {mode}）")
    print("=" * 70)
    
    try:
        load_secrets(mode=mode)
        
        # 测试获取配置
        print("\n测试获取配置:")
        for key in ['HF_TOKEN', 'WANDB_API_KEY', 'KAGGLE_API_TOKEN']:
            try:
                value = get_secret(key)
                # 只显示前10个字符
                masked = value[:10] + "..." if len(value) > 10 else value
                print(f"{key}: {masked}")
            except ValueError as e:
                print(f"{key}: 未设置")
    
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
