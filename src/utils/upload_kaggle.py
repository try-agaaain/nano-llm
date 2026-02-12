#!/usr/bin/env python
"""Upload secrets dataset and notebook to Kaggle"""
import subprocess
import sys
import json
import yaml
from pathlib import Path

# 导入env_loader模块来加载环境变量
from src.utils.env_loader import load_secrets


def _load_config():
    """加载config.yaml配置"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _slug(text: str) -> str:
    """将文本转换为slug格式（小写，空格转连字符）"""
    return text.lower().replace(" ", "-")


def _generate_metadata_file(metadata_dir: Path, metadata: dict, filename: str, username: str = "") -> bool:
    """生成元数据文件"""
    metadata_file = metadata_dir / filename
    try:
        metadata["id"] = f"{username}/{_slug(metadata['title'])}"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"生成{filename}失败: {e}")
        return False


def upload_secrets(path: str="kaggle/secrets"):
    """上传secrets数据集到Kaggle"""
    secrets_dir = Path(path)
    if not secrets_dir.exists():
        print(f"错误: {secrets_dir} 目录不存在")
        return False

    # 生成dataset-metadata.json
    config = _load_config()
    username = config.get("kaggle", {}).get("username", "")
    dataset_meta = config.get("kaggle", {}).get("dataset", {})
    if dataset_meta and not _generate_metadata_file(secrets_dir, dataset_meta, "dataset-metadata.json", username):
        return False

    print(f"正在上传secrets数据集: {secrets_dir}")
    cmd = ['kaggle', 'datasets', 'create', '-p', str(secrets_dir), '-q']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Secrets数据集上传失败:")
        print(result.stderr)
        return False
    else:
        print("Secrets数据集上传成功")
        print(result.stdout)
        return True


def upload_notebook():
    """推送notebook到Kaggle"""
    notebook_dir = Path('kaggle/notebook')
    if not notebook_dir.exists():
        print(f"错误: {notebook_dir} 目录不存在")
        return False

    # 生成kernel-metadata.json
    config = _load_config()
    username = config.get("kaggle", {}).get("username", "")
    notebook_meta = config.get("kaggle", {}).get("notebook", {})
    if notebook_meta and not _generate_metadata_file(notebook_dir, notebook_meta, "kernel-metadata.json", username):
        return False

    accelerator = notebook_meta.get("accelerator", "NvidiaTeslaP100")

    print(f"正在推送notebook到Kaggle...")
    cmd = ['kaggle', 'kernels', 'push', '-p', str(notebook_dir), '--accelerator', accelerator]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Notebook推送失败:")
        print(result.stderr)
        return False
    else:
        print("Notebook推送成功")
        print(result.stdout)
        return True


def main():
    """上传secrets数据集和notebook到Kaggle"""
    # 加载环境变量（本地模式）
    try:
        load_secrets()
    except ValueError as e:
        print(f"加载配置失败: {e}")
        sys.exit(1)
    
    # 上传secrets数据集
    if not upload_secrets():
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")
    
    # 推送notebook
    if not upload_notebook():
        sys.exit(1)
    
    print("\n🎉 所有上传完成！")


if __name__ == "__main__":
    main()
