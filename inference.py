import torch
import sys
from pathlib import Path
from model import NanoLLM
from tokenizer import load_or_train_tokenizer


# 预设的10条测试用例
TEST_CASES = [
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


def generate_text(model, tokenizer, device, prompt_text, max_length=100, temperature=0.8, top_k=50):
    """生成文本的辅助函数"""
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt=prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )
    
    generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
    generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
    return generated_text.strip()


def run_test_mode(model, tokenizer, device):
    """预设测试模式 - 运行10条测试用例"""
    print("\n" + "=" * 70)
    print("预设测试模式 - 运行10条测试用例")
    print("=" * 70 + "\n")
    
    model.eval()
    
    for idx, prompt_text in enumerate(TEST_CASES, 1):
        print(f"[测试用例 {idx}/10]")
        print(f"  提示词: {prompt_text}")
        
        generated_text = generate_text(model, tokenizer, device, prompt_text)
        print(f"  生成文本: {generated_text}")
        print()
    
    print("=" * 70)
    print("测试完成")
    print("=" * 70)


def run_interactive_mode(model, tokenizer, device):
    """交互模式 - 用户输入提示词"""
    print("\n" + "=" * 70)
    print("交互模式 - 输入提示词进行生成")
    print("=" * 70)
    print("输入 'quit'、'exit' 或 'q' 退出\n")
    
    model.eval()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            generated_text = generate_text(model, tokenizer, device, user_input)
            print(f"Model: {generated_text}\n")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading tokenizer...")
    tokenizer = load_or_train_tokenizer(tokenizer_path="./tokenizer", force_retrain=False)
    
    print("Loading model...")
    model_state = torch.load("best_model.pt", map_location=device)
    
    # Extract model dimensions from state dict
    d_model = model_state['embedding.weight'].shape[1]
    vocab_size = model_state['embedding.weight'].shape[0]
    
    # Get max_seq_len from pos_encoding shape
    max_seq_len = model_state['pos_encoding.pe'].shape[1]
    
    model = NanoLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=4,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"Model loaded. Device: {device}\n")
    
    # 确定运行模式
    mode = "test"  # 默认模式：测试
    
    # 执行相应模式
    if mode == "test":
        run_test_mode(model, tokenizer, device)
    else:
        run_interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
