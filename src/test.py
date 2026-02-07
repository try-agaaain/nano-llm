"""交互式模型测试脚本 - 从wandb拉取最新模型并进行文本生成"""

import os
import torch

# 加载环境变量配置（默认使用kaggle模式）
from src.utils.env_loader import load_secrets
load_secrets(mode=os.getenv("CONFIG_MODE", "kaggle"))
import wandb
from pathlib import Path
from typing import Optional
from model import NanoLLM
from tokenizer import load_or_train_tokenizer


class InteractiveModelTester:
    """交互式模型测试器"""
    
    def __init__(
        self,
        project_name: str = "nano-llm",
        model_name: str = "TinyStories-training-step-based",
        download_from_wandb: bool = True,
        local_model_path: Optional[str] = None,
    ):
        """
        初始化测试器
        
        Args:
            project_name: wandb项目名称
            model_name: wandb运行名称（用于过滤特定运行）
            download_from_wandb: 是否从wandb下载最新模型
            local_model_path: 本地模型路径（如果不从wandb下载）
        """
        self.project_name = project_name
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        print("=" * 70)
        print("NanoLLM 交互式测试工具")
        print("=" * 70)
        
        # 初始化wandb
        wandb_api_key = os.getenv("WANDB_API_KEY", "")
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY未设置。请在环境变量中设置API密钥")
        wandb.login(key=wandb_api_key)
        
        # 加载模型和tokenizer
        if download_from_wandb:
            self._download_model_from_wandb()
        elif local_model_path:
            self._load_local_model(local_model_path)
        else:
            raise ValueError("必须指定从wandb下载或本地模型路径")
        
        # 加载tokenizer
        self._load_tokenizer()
        
        print(f"\n设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Tokenizer词汇表大小: {self.tokenizer.vocab_size}")
        print("\n" + "=" * 70 + "\n")
    
    def _download_model_from_wandb(self):
        """从wandb下载最新的最佳模型"""
        print("正在连接wandb并获取最新运行信息...")
        
        try:
            # 获取项目的所有运行
            api = wandb.Api()
            runs = api.runs(f"{api.default_entity}/{self.project_name}")
            
            # 查找匹配的运行
            target_run = None
            for run in runs:
                if self.model_name in run.name:
                    target_run = run
                    break
            
            if not target_run:
                # 如果找不到具体的运行名称，使用最新的运行
                print(f"未找到名称为'{self.model_name}'的运行，使用最新的运行")
                if runs:
                    target_run = runs[0]
                else:
                    raise ValueError("wandb中没有任何运行")
            
            print(f"找到运行: {target_run.name} (ID: {target_run.id})")
            print(f"状态: {target_run.state}")
            print(f"创建时间: {target_run.created_at}")
            
            # 下载best_model.pt
            print("\n正在下载最佳模型...")
            model_artifact = target_run.file("best_model.pt")
            model_path = "best_model_from_wandb.pt"
            model_artifact.download(root=".")
            
            # wandb下载的文件可能会在子目录中
            if os.path.exists(f"best_model.pt"):
                model_path = "best_model.pt"
            elif os.path.exists(f"./best_model.pt"):
                model_path = "./best_model.pt"
            
            print(f"✓ 模型已下载到: {model_path}")
            
            # 加载模型配置
            print("\n正在读取模型配置...")
            self.model_config = target_run.config
            print(f"模型配置:")
            for key, value in self.model_config.items():
                print(f"  {key}: {value}")
            
            # 创建和加载模型
            self._create_and_load_model(model_path)
            
        except Exception as e:
            print(f"从wandb下载失败: {e}")
            raise
    
    def _load_local_model(self, model_path: str):
        """从本地路径加载模型"""
        print(f"正在从本地加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 尝试从checkpoint文件读取配置
        if model_path.endswith(".pt"):
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # 这是一个checkpoint文件
                print("检测到checkpoint文件格式")
                state_dict = checkpoint["model_state_dict"]
            else:
                # 这是一个直接的模型文件
                state_dict = checkpoint
            
            self._create_and_load_model(model_path, state_dict)
    
    def _create_and_load_model(self, model_path: str, state_dict=None):
        """创建模型并加载状态"""
        # 使用配置的参数或默认参数
        if self.model_config:
            d_model = self.model_config.get("d_model", 384)
            num_heads = self.model_config.get("num_heads", 8)
            num_layers = self.model_config.get("num_layers", 8)
            max_length = self.model_config.get("max_length", 1024)
        else:
            # 默认参数
            d_model = 384
            num_heads = 8
            num_layers = 8
            max_length = 1024
        
        # 先加载tokenizer以获取vocab_size
        self._load_tokenizer()
        vocab_size = self.tokenizer.vocab_size
        
        # 创建模型
        self.model = NanoLLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_model * 4,
            max_seq_len=max_length,
        ).to(self.device)
        
        # 加载模型权重
        if state_dict is None:
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ 模型已加载")
    
    def _load_tokenizer(self):
        """加载tokenizer"""
        if self.tokenizer is None:
            print("正在加载tokenizer...")
            self.tokenizer = load_or_train_tokenizer(
                tokenizer_path="./tokenizer",
                vocab_size=8192,
                num_samples=50000,
                force_retrain=False,
                dataset_dir="./dataset",
                csv_path=None,
                text_column="text"
            )
            print(f"✓ Tokenizer已加载，词汇表大小: {self.tokenizer.vocab_size}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            max_length: 生成的最大长度
            temperature: 采样温度
            top_k: top-k采样参数
        
        Returns:
            生成的文本
        """
        with torch.no_grad():
            # 编码提示词
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            generated_ids = self.model.generate(
                prompt=prompt_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
            )
            
            # 解码生成的token（跳过提示词部分）
            generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
            generated_text = self.tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            
            return generated_text.strip()
    
    def interactive_session(self):
        """启动交互式对话会话"""
        print("\n" + "=" * 70)
        print("交互式文本生成会话")
        print("=" * 70)
        print("\n提示:")
        print("  - 输入任何提示词来生成文本")
        print("  - 输入 'help' 查看命令列表")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'config' 查看生成参数")
        print("\n" + "=" * 70 + "\n")
        
        # 生成参数
        max_length = 100
        temperature = 0.8
        top_k = 50
        
        while True:
            try:
                user_input = input("提示词> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["quit", "exit"]:
                    print("退出程序...")
                    break
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                if user_input.lower() == "config":
                    self._print_config(max_length, temperature, top_k)
                    continue
                
                if user_input.lower().startswith("maxlen "):
                    try:
                        max_length = int(user_input.split()[1])
                        print(f"✓ 最大长度已设置为: {max_length}")
                    except (ValueError, IndexError):
                        print("❌ 无效的长度值。用法: maxlen <数字>")
                    continue
                
                if user_input.lower().startswith("temp "):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"✓ 温度已设置为: {temperature}")
                    except (ValueError, IndexError):
                        print("❌ 无效的温度值。用法: temp <0.0-2.0>")
                    continue
                
                if user_input.lower().startswith("topk "):
                    try:
                        top_k = int(user_input.split()[1])
                        print(f"✓ top-k已设置为: {top_k}")
                    except (ValueError, IndexError):
                        print("❌ 无效的top-k值。用法: topk <数字>")
                    continue
                
                # 生成文本
                print("\n生成中...\n")
                generated_text = self.generate(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                )
                
                print(f"提示词: {user_input}")
                print(f"生成文本: {generated_text}\n")
                
            except KeyboardInterrupt:
                print("\n\n退出程序...")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                print("请重试...\n")
    
    def _print_help(self):
        """打印帮助信息"""
        print(
            """
命令列表:
  <任何文本>        生成文本
  help              显示此帮助信息
  config            显示当前生成参数
  maxlen <数字>     设置最大生成长度（默认100）
  temp <数字>       设置采样温度（默认0.8，推荐0.5-1.5）
  topk <数字>       设置top-k采样值（默认50）
  quit/exit         退出程序
"""
        )
    
    def _print_config(self, max_length, temperature, top_k):
        """打印当前配置"""
        print(f"""
当前生成参数:
  最大长度 (maxlen): {max_length}
  采样温度 (temp):   {temperature}
  top-k采样 (topk):  {top_k}
""")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="交互式模型测试工具")
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="本地模型路径（如果指定，将跳过从wandb下载）"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="nano-llm",
        help="wandb项目名称"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="TinyStories-training-step-based",
        help="wandb运行名称（用于过滤特定运行）"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="不从wandb下载，需要指定--local-model"
    )
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = InteractiveModelTester(
        project_name=args.project,
        model_name=args.run_name,
        download_from_wandb=not args.no_wandb,
        local_model_path=args.local_model,
    )
    
    # 启动交互式会话
    tester.interactive_session()


if __name__ == "__main__":
    main()
