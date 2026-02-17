"""模型推理脚本"""

import torch
import yaml
from pathlib import Path

from src.dataset.tinystories import TinyStoriesDataset
from src.model import NanoLLM
from src.tokenizer import load_or_train_tokenizer
from src.utils.wandb_utils import WandbManager


TEST_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "In the forest",
    "The brave knight",
    "A mysterious door",
    "The wizard cast",
    "Under the moonlight",
    "The treasure was",
    "A strange creature",
    "The adventure began"
]


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        # 默认查找项目根目录的config.yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_model(model_path="output/best_model.pt", from_wandb=True, wandb_version="latest", device=None, config_path=None):
    """加载模型
    
    Args:
        model_path: 本地模型路径
        from_wandb: 是否从W&B下载模型
        wandb_version: W&B模型版本
        device: 设备
        config_path: 配置文件路径
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    config = load_config(config_path)
    training_config = config.get("training", {})
    inference_config = config.get("inference", {})
    
    from_wandb = inference_config.get("from_wandb", False)
    
    # 从配置中提取参数
    d_model = training_config.get("d_model", 384)
    num_heads = training_config.get("num_heads", 8)
    num_layers = training_config.get("num_layers", 8)
    d_ff_multiplier = training_config.get("d_ff_multiplier", 4)
    max_length = training_config.get("max_length", 1024)
    vocab_size = training_config.get("vocab_size", 8192)
    
    # 如果指定从W&B下载
    if from_wandb:
        print(f"正在从 W&B 下载模型 (版本: {wandb_version})...")
        try:
            manager = WandbManager()
            artifact_dir = manager.download_model(version=wandb_version)
            manager.finish()
            if artifact_dir:
                model_path = artifact_dir / "best_model.pt"
            else:
                print("W&B下载失败，尝试使用本地模型...")
        except Exception as e:
            print(f"W&B下载出错: {e}，尝试使用本地模型...")
    
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    model = NanoLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=d_model * d_ff_multiplier,
        max_seq_len=max_length,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, device


def generate(model, tokenizer, device, prompt, max_length=100, temperature=0.1, top_k=1):
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt=prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )
    
    # 只解码生成的部分
    generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
    return tokenizer.decode(generated_only_ids, skip_special_tokens=True).strip()


def test_mode(model, tokenizer, device, temperature=0.1, top_k=1):
    """预设测试模式"""
    print("\n" + "="*70)
    print("Test Mode - Running 10 test cases")
    print("="*70 + "\n")
    
    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{idx}/10] {prompt}")
        output = generate(model, tokenizer, device, prompt, temperature=temperature, top_k=top_k)
        print(f"  → {output}\n")


def interactive_mode(model, tokenizer, device, temperature=0.1, top_k=1):
    """交互模式"""
    print("\n" + "="*70)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt or prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            output = generate(model, tokenizer, device, prompt, temperature=temperature, top_k=top_k)
            print(f"Model: {output}\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Goodbye!")


def main(mode="test", from_wandb=True, wandb_version="latest", config_path=None):
    """主函数
    
    Args:
        mode: 运行模式 ("test" 或 "interactive")
        from_wandb: 是否从W&B下载模型
        wandb_version: W&B模型版本
        config_path: 配置文件路径
    """
    workspace_dir = Path(__file__).parent.parent
    
    if config_path is None:
        config_path = workspace_dir / "config.yaml"
    
    # 加载配置
    config = load_config(config_path)
    inference_config = config.get("inference", {})
    
    # 从配置中读取推理参数
    temperature = inference_config.get("temperature", 0.1)
    top_k = inference_config.get("top_k", 2)
    
    dataset_dir = workspace_dir / "dataset" / "tinystories-narrative-classification"
    train_dataset_raw, val_dataset_raw = TinyStoriesDataset.load_datasets(dataset_dir)
    tokenizer = load_or_train_tokenizer(tokenizer_path="./output/tokenizer", dataset=train_dataset_raw, force_retrain=False)
    model, device = load_model(from_wandb=from_wandb, wandb_version=wandb_version, config_path=str(config_path))
    
    if mode == "test":
        test_mode(model, tokenizer, device, temperature=temperature, top_k=top_k)
    else:
        interactive_mode(model, tokenizer, device, temperature=temperature, top_k=top_k)


if __name__ == "__main__":
    # 使用示例：
    # main(mode="test")                              # 使用本地模型测试(根据config.yaml配置)
    # main(mode="test", from_wandb=True)             # 从W&B下载最新模型测试
    # main(mode="interactive", from_wandb=True)      # 从W&B下载模型进入交互
    # main(mode="interactive")                       # 使用本地模型进入交互模式
    main(mode="test")  # 改为 "interactive" 进入交互模式
