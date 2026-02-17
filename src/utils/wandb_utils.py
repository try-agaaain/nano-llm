#!/usr/bin/env python
"""W&B utilities for model artifact management"""
import os
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import wandb


class WandbManager:
    """W&B操作管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化WandbManager，自动完成登陆和run初始化
        
        Args:
            config_path: config.yaml路径，默认为项目根目录下的config.yaml
        
        Raises:
            ValueError: 当config或API密钥缺失、登陆失败时
        """
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        self.config_path = config_path
        self.config = self._load_config()
        self.wandb_config = self.config.get("wandb", {})
        self.training_config = self.config.get("training", {})
        
        # 配置项
        self.username = self.wandb_config.get("username", "")
        self.project = self.wandb_config.get("project", "")
        
        # Artifact配置
        self.artifact_path = self.wandb_config.get("artifact_path", "output/best_model.pt")
        self.artifact_name = self.wandb_config.get("artifact_name", "nano-llm-model")
        self.artifact_type = self.wandb_config.get("artifact_type", "model")
        
        # 运行时状态
        self.run = None
        
        api_key = os.getenv("WANDB_API_KEY", "")
        if not api_key:
            raise ValueError("WANDB_API_KEY 环境变量未设置")
        if not self.login(api_key):
            raise ValueError("W&B 登陆失败")
        
        # 自动初始化run
        output_dir = os.getenv("WANDB_OUTPUT_DIR", "output/wandb")
        if not self.init_run(
            run_name=self.project,
            job_type="train",
            config=self.training_config,
            output_dir=output_dir
        ):
            raise ValueError("W&B run 初始化失败")
    
    def _load_config(self) -> dict:
        """加载config.yaml配置"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件{self.config_path}不存在")
    
    def login(self, api_key: Optional[str] = None) -> bool:
        """登录W&B
        
        Args:
            api_key: API密钥，如果未提供则从环境变量WANDB_API_KEY读取
        
        Returns:
            成功返回True，失败返回False
        """
        api_key = api_key or os.getenv("WANDB_API_KEY", "")
        if not api_key:
            raise ValueError("WANDB_API_KEY未提供且环境变量未设置")
        
        try:
            wandb.login(key=api_key)
            print(f"✓ W&B登录成功 | 用户: {self.username}")
            return True
        except Exception as e:
            print(f"登录失败: {e}")
            return False
    
    def init_run(self, run_name: Optional[str] = None, job_type: str = "train", 
                 config: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> bool:
        """初始化W&B运行
        
        Args:
            run_name: 运行名称，默认使用project_name
            job_type: 作业类型（train, eval, inference等）
            config: 训练配置字典
            output_dir: wandb输出目录
        
        Returns:
            成功返回True，失败返回False
        """
        if self.run is not None:
            print("警告: W&B run已存在，将先结束当前run")
            self.finish()
        
        try:
            run_name = run_name or self.project
            kwargs = {
                "project": self.project,
                "entity": self.username,
                "name": run_name,
                "job_type": job_type,
            }
            
            if config:
                kwargs["config"] = config
            
            if output_dir:
                kwargs["dir"] = str(output_dir)
            
            self.run = wandb.init(**kwargs)
            return True
        except Exception as e:
            print(f"初始化运行失败: {e}")
            return False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> bool:
        """记录指标
        
        Args:
            metrics: 指标字典
            step: 步数（可选）
        
        Returns:
            成功返回True，失败返回False
        """
        if self.run is None:
            print("错误: 未初始化W&B run，请先调用init_run()")
            return False
        
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
            return True
        except Exception as e:
            print(f"记录指标失败: {e}")
            return False
    
    def finish(self) -> bool:
        """结束W&B运行
        
        Returns:
            成功返回True，失败返回False
        """
        if self.run is None:
            return True
        
        try:
            wandb.finish()
            self.run = None
            return True
        except Exception as e:
            print(f"结束运行失败: {e}")
            return False
    
    def upload_model(self, model_path: Optional[str] = None, version_notes: str = "训练完成") -> bool:
        """上传模型artifact到W&B
        
        Args:
            model_path: 模型文件路径，默认使用config中的artifact_path
            version_notes: 版本说明
        
        Returns:
            成功返回True，失败返回False
        """
        model_path = model_path or self.artifact_path
        model_file = Path(model_path)
        
        if not model_file.exists():
            print(f"错误: 模型文件 {model_path} 不存在")
            return False
        
        try:
            print(f"正在上传模型到 W&B...")
            print(f"  项目: {self.username}/{self.project}")
            print(f"  名称: {self.artifact_name}")
            print(f"  文件: {model_path}")
            
            # 创建artifact
            artifact = wandb.Artifact(
                name=self.artifact_name,
                type=self.artifact_type,
                description=f"{self.project} 训练模型"
            )
            
            # 添加模型文件
            artifact.add_file(str(model_file))
            
            # 上传
            self.run.log_artifact(artifact)
            
            print(f"✓ 模型上传成功")
            return True
            
        except Exception as e:
            print(f"上传失败: {e}")
            return False
    
    def download_model(self, version: str = "latest", output_dir: Optional[str] = None) -> Optional[str]:
        """从W&B下载模型artifact
        
        Args:
            version: 版本标签，默认为"latest"
            output_dir: 下载目录，默认为"output/wandb_models"
        
        Returns:
            下载成功返回模型文件路径，失败返回None
        """
        output_dir = output_dir or "output/wandb_models"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"正在从 W&B 下载模型...")
            print(f"  项目: {self.username}/{self.project}")
            print(f"  名称: {self.artifact_name}")
            print(f"  版本: {version}")
            
            # 确保已登录
            if not self.login():
                print("错误: W&B 登录失败")
                return None
            
            # 构造artifact引用：entity/project/artifact_name:version
            artifact_ref = f"{self.username}/{self.project}/{self.artifact_name}:{version}"
            artifact = self.run.use_artifact(artifact_ref)
            
            # 下载artifact
            artifact_dir = artifact.download(root=str(output_path))
            
            print(f"✓ artifact已下载至: {artifact_dir}")
            return Path(artifact_dir)
            
        except Exception as e:
            print(f"下载失败: {e}")
            return None

def main():
    """主函数，处理命令行参数"""
    if len(sys.argv) < 2:
        print("使用方法: wandb_utils.py <command> [args]")
        print("\n可用命令:")
        print("  upload [model_path]  - 上传模型到W&B")
        print("  download [version]   - 从W&B下载模型（默认latest）")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        # 创建WandbManager（自动登陆和初始化run）
        manager = WandbManager()
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    
    try:
        if command == "upload":
            model_path = sys.argv[2] if len(sys.argv) > 2 else None
            success = manager.upload_model(model_path)
        elif command == "download":
            version = sys.argv[2] if len(sys.argv) > 2 else "latest"
            model_path = manager.download_model(version)
            success = model_path is not None
        else:
            print(f"未知命令: {command}")
            success = False
    finally:
        manager.finish()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
