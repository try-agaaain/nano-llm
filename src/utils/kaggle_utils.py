#!/usr/bin/env python
"""Kaggle utilities for managing notebooks, codespace, and metadata"""
import subprocess
import sys
import json
import yaml
import shutil
from pathlib import Path
from typing import Optional

from src.utils.env_loader import load_secrets
from src.utils.file_collector import FileCollector


class KaggleManager:
    """Kaggle操作管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化KaggleManager
        
        Args:
            config_path: config.yaml路径，默认为项目根目录下的config.yaml
        
        Raises:
            ValueError: 当config.yaml中缺少必要的username或notebook.title配置时
        """
        # 如果未提供config_path，使用默认路径
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        self.config_path = config_path
        self.config = self._load_config()
        self.kaggle_config = self.config.get("kaggle", {})
        self.username = self.kaggle_config.get("username", "")
        self.notebook_title = self.kaggle_config.get("notebook", {}).get("title", "")
        self.kernel_ref = f"{self.username}/{self._slug(self.notebook_title)}"
        
        # 路径配置
        self.codespace_dir = Path("output/kaggle/codespace")
        self.notebook_dir = Path("output/kaggle/notebook")
        self.output_dir = Path("output/kaggle/output")
    
    def _load_config(self) -> dict:
        """加载config.yaml配置"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件{self.config_path}不存在")
    
    @staticmethod
    def _slug(text: str) -> str:
        """将文本转换为slug格式（小写，空格转连字符）"""
        return text.lower().replace(" ", "-")
    
    def _parse_patterns(self, patterns_config, collector: FileCollector) -> list:
        """解析模式配置（支持文件路径字符串或模式列表）"""
        if isinstance(patterns_config, str):
            # 如果是字符串，则当作文件路径处理
            patterns_file = collector.root / patterns_config
            return collector.parse_exclude_patterns(patterns_file)
        elif isinstance(patterns_config, list):
            # 如果已经是列表，直接返回
            return patterns_config
        else:
            # 其他情况返回空列表
            return []
    
    def _generate_metadata_file(self, metadata_dir: Path, metadata: dict, filename: str) -> bool:
        """生成元数据文件"""
        metadata_file = metadata_dir / filename
        try:
            metadata["id"] = f"{self.username}/{self._slug(metadata['title'])}"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"生成{filename}失败: {e}")
            return False
    
    def _run_kaggle_command(self, cmd: list, description: str) -> bool:
        """运行kaggle CLI命令的通用方法"""
        print(f"{description}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"失败:")
            print(result.stderr)
            return False
        else:
            print(f"成功")
            if result.stdout.strip():
                print(result.stdout)
            return True
    
    def _check_dataset_exists(self, dataset_slug: str) -> bool:
        """检查数据集是否存在
        
        Args:
            dataset_slug: 数据集的 slug，格式为 "username/dataset-name"
        
        Returns:
            存在返回 True，不存在返回 False
        """
        cmd = ['kaggle', 'datasets', 'status', dataset_slug]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def init_metadata(self) -> bool:
        """初始化Kaggle元数据（kinit）"""
        self.notebook_dir.mkdir(parents=True, exist_ok=True)
        cmd = ['kaggle', 'kernels', 'init', '-p', str(self.notebook_dir)]
        return self._run_kaggle_command(cmd, "初始化Kaggle元数据")
    
    def upload_codespace(self) -> bool:
        """上传代码包数据集到 Kaggle（kpush的一部分）"""
        # 清理并重建目录
        if self.codespace_dir.exists():
            shutil.rmtree(self.codespace_dir)
        self.codespace_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用FileCollector收集文件
        collector = FileCollector()
        code_config = self.kaggle_config.get("codespace", {})
        
        # 解析排除模式（支持文件路径字符串或模式列表）
        ignore_patterns_config = code_config.get("ignore_patterns", ".gitignore")
        exclude_patterns = self._parse_patterns(ignore_patterns_config, collector)
        
        # 解析包含模式（支持文件路径字符串或模式列表）
        include_patterns_config = code_config.get("include_patterns", [])
        include_patterns = self._parse_patterns(include_patterns_config, collector)
        
        # 收集文件
        all_files = collector.collect_files(exclude_patterns, include_patterns)
        collector.copy_files(all_files, self.codespace_dir)
        
        # 生成dataset-metadata.json
        dataset_meta = code_config.copy()
        dataset_meta.pop("ignore_patterns", None)
        dataset_meta.pop("include_patterns", None)
        
        if dataset_meta and not self._generate_metadata_file(self.codespace_dir, dataset_meta, "dataset-metadata.json"):
            return False
        
        # 检查数据集是否存在
        dataset_slug = f"{self.username}/{self._slug(dataset_meta.get('title', 'codespace'))}"
        dataset_exists = self._check_dataset_exists(dataset_slug)
        
        if dataset_exists:
            # 存在则更新
            cmd = ['kaggle', 'datasets', 'version', '-m', "更新", '-p', str(self.codespace_dir), '-r', 'tar']
            return self._run_kaggle_command(cmd, "正在更新代码包数据集")
        else:
            # 不存在则创建
            cmd = ['kaggle', 'datasets', 'create', '-p', str(self.codespace_dir), '-r', 'tar']
            return self._run_kaggle_command(cmd, "正在创建代码包数据集")
    
    def upload_notebook(self) -> bool:
        """推送notebook到Kaggle（kpush的一部分）"""
        self.notebook_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成kernel-metadata.json
        notebook_meta = self.kaggle_config.get("notebook", {})
        
        # 从config中获取code_file路径（如src/notebook.ipynb）
        code_file_path = notebook_meta.get("code_file", "")
        source_notebook = Path(code_file_path)
        
        if not source_notebook.exists():
            print(f"错误: {source_notebook} 不存在")
            return False
        
        # 复制notebook文件到kaggle/notebook目录
        filename = source_notebook.name
        target_notebook = self.notebook_dir / filename
        shutil.copy2(source_notebook, target_notebook)
        notebook_meta["code_file"] = filename
        
        # 添加codespace
        dataset_sources = notebook_meta.get("dataset_sources", [])
        codespace_source = f"{self.username}/codespace"
        if codespace_source not in dataset_sources:
            dataset_sources.append(codespace_source)
        notebook_meta["dataset_sources"] = dataset_sources
        
        if notebook_meta and not self._generate_metadata_file(self.notebook_dir, notebook_meta, "kernel-metadata.json"):
            return False
        
        accelerator = notebook_meta.get("accelerator", "NvidiaTeslaP100")
        cmd = ['kaggle', 'kernels', 'push', '-p', str(self.notebook_dir), '--accelerator', accelerator]
        return self._run_kaggle_command(cmd, "正在推送notebook到Kaggle")
    
    def pull(self) -> bool:
        """从Kaggle拉取notebook（kpull）"""
        cmd = ['kaggle', 'kernels', 'pull', self.kernel_ref, '-p', str(self.notebook_dir), '-m']
        return self._run_kaggle_command(cmd, "正在从Kaggle拉取notebook")
    
    def status(self) -> bool:
        """检查kernel状态（kstatus）"""
        cmd = ['kaggle', 'kernels', 'status', self.kernel_ref]
        return self._run_kaggle_command(cmd, "正在检查notebook状态")
    
    def output(self) -> bool:
        """获取kernel输出（koutput）"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = ['kaggle', 'kernels', 'output', self.kernel_ref, '-p', str(self.output_dir)]
        return self._run_kaggle_command(cmd, "正在获取notebook输出")
    
    def push(self) -> bool:
        """上传代码包数据集和notebook到Kaggle（完整推送）"""
        try:
            load_secrets()
        except ValueError as e:
            print(f"加载配置失败: {e}")
            return False
        
        # 上传代码包数据集
        if not self.upload_codespace():
            return False
        
        print("\n" + "="*70 + "\n")
        
        # 推送notebook
        if not self.upload_notebook():
            return False
        
        print("\n🎉 所有上传完成！")
        return True


def main():
    """主函数，处理命令行参数"""
    if len(sys.argv) < 2:
        print("使用方法: kaggle_utils.py <command>")
        print("\n可用命令:")
        print("  init      - 初始化Kaggle元数据")
        print("  push      - 上传codespace和notebook（完整推送）")
        print("  pull      - 拉取notebook")
        print("  status    - 检查notebook状态")
        print("  output    - 获取notebook输出")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        manager = KaggleManager()
    except (FileNotFoundError, ValueError) as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    
    if command == "init":
        success = manager.init_metadata()
    elif command == "push":
        success = manager.push()
    elif command == "pull":
        success = manager.pull()
    elif command == "status":
        success = manager.status()
    elif command == "output":
        success = manager.output()
    else:
        print(f"未知命令: {command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
