"""模型推理脚本"""

import torch
from pathlib import Path

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


def load_model(model_path="output/best_model.pt", from_wandb=True, wandb_version="latest", device=None):
    """加载模型
    
    Args:
        model_path: 本地模型路径
        from_wandb: 是否从W&B下载模型
        wandb_version: W&B模型版本
        device: 设备
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果指定从W&B下载
    if from_wandb:
        print(f"正在从 W&B 下载模型 (版本: {wandb_version})...")
        try:
            manager = WandbManager()
            downloaded_path = manager.download_model(version=wandb_version)
            manager.finish()
            if downloaded_path:
                model_path = downloaded_path
            else:
                print("W&B下载失败，尝试使用本地模型...")
        except Exception as e:
            print(f"W&B下载出错: {e}，尝试使用本地模型...")
    
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # 从state_dict推断模型配置
    d_model = state_dict['embedding.weight'].shape[1]
    vocab_size = state_dict['embedding.weight'].shape[0]
    max_seq_len = state_dict['pos_encoding.pe'].shape[1]
    
    model = NanoLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=8,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, device


def generate(model, tokenizer, device, prompt, max_length=100, temperature=0.8, top_k=50):
    """生成文本"""
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


def test_mode(model, tokenizer, device):
    """预设测试模式"""
    print("\n" + "="*70)
    print("Test Mode - Running 10 test cases")
    print("="*70 + "\n")
    
    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"[{idx}/10] {prompt}")
        output = generate(model, tokenizer, device, prompt)
        print(f"  → {output}\n")


def interactive_mode(model, tokenizer, device):
    """交互模式"""
    print("\n" + "="*70)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt or prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            output = generate(model, tokenizer, device, prompt)
            print(f"Model: {output}\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("Goodbye!")


def main(mode="test", from_wandb=True, wandb_version="latest"):
    """主函数
    
    Args:
        mode: 运行模式 ("test" 或 "interactive")
        from_wandb: 是否从W&B下载模型
        wandb_version: W&B模型版本
    """
    tokenizer = load_or_train_tokenizer(tokenizer_path="./tokenizer", force_retrain=False)
    model, device = load_model(from_wandb=from_wandb, wandb_version=wandb_version)
    
    if mode == "test":
        test_mode(model, tokenizer, device)
    else:
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    # 使用示例：
    # main(mode="test")                              # 使用本地模型测试
    # main(mode="test", from_wandb=True)             # 从W&B下载最新模型测试
    # main(mode="interactive", from_wandb=True)      # 从W&B下载模型进入交互
    main(mode="test")  # 改为 "interactive" 进入交互模式
