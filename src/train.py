"""模型训练脚本 - 使用TinyStories数据集"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import wandb

from src.model import NanoLLM
from src.tokenizer import load_or_train_tokenizer
from src.dataset import TinyStoriesDataset


def create_data_loaders(
    tokenizer, 
    batch_size=32, 
    max_length=512,
    train_csv_path=None,
    val_csv_path=None,
    text_column="text",
):
    """创建训练和验证数据加载器"""
    
    train_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        csv_path=train_csv_path,
        max_length=max_length,
        text_column=text_column,
    )
    
    val_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        csv_path=val_csv_path,
        max_length=max_length,
        text_column=text_column,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_step(model, batch, optimizer, criterion, device):
    """执行单个训练step"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # 前向传播
    optimizer.zero_grad()
    logits = model(input_ids)
    
    # 计算损失
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # 反向传播
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def evaluate(model, val_loader, criterion, device, num_steps=100, tokenizer=None, step_num=None):
    """验证模型 - 执行固定数量的step，返回损失和困惑度，并生成测试文本"""
    model.eval()
    total_loss = 0.0
    
    # 定义10条测试用例
    test_prompts = [
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
    
    with torch.no_grad():
        for step in range(num_steps):
            try:
                batch = next(iter(val_loader))
            except StopIteration:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    # 困惑度 = exp(平均损失)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # 生成测试文本
    if tokenizer is not None:
        print(f"\n  测试文本生成 (Step {step_num}):")
        print("  " + "-" * 60)
        
        for idx, prompt_text in enumerate(test_prompts, 1):
            try:
                prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        prompt=prompt_ids,
                        max_length=50,
                        temperature=0.8,
                        top_k=50,
                    )
                
                generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
                generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
                
                print(f"  [{idx:2d}] 提示词: {prompt_text}")
                print(f"      生成: {generated_text.strip()}\n")
            except Exception as e:
                print(f"  [{idx:2d}] 提示词: {prompt_text}")
                print(f"      生成失败: {str(e)}\n")
        
        print("  " + "-" * 60)
    
    return avg_loss, perplexity


def main():
    """主训练函数"""
    
    # 初始化wandb
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY未设置。请在环境变量中设置API密钥")
    wandb.login(key=wandb_api_key)

    # 超参数
    d_model = 384
    num_heads = 8
    num_layers = 8
    learning_rate = 0.0001
    batch_size = 32
    max_length = 1024
    max_steps = 10000  # 最大训练步数
    validation_interval = 1000
    
    wandb.init(
        project="nano-llm",
        name="TinyStories-training-step-based",
        config={
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "validation_interval": validation_interval,
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("NanoLLM 训练 - TinyStories数据集 (基于Step)")
    print("=" * 70)
    
    # 加载分词器
    print("正在加载TinyStories分词器...")
    
    # 检查 Kaggle CSV 路径
    train_csv_path = "/kaggle/input/tinystories-narrative-classification/train.csv"
    csv_path_for_tokenizer = train_csv_path if os.path.exists(train_csv_path) else None
    
    tokenizer = load_or_train_tokenizer(
        tokenizer_path="./tokenizer",
        vocab_size=8192,
        num_samples=50000,
        force_retrain=False,  # 改为False，只在需要时重新训练
        dataset_dir="./dataset",
        csv_path=csv_path_for_tokenizer,
        text_column="text"
    )
    vocab_size = tokenizer.vocab_size
    print(f"分词器词汇表大小: {vocab_size}")
    
    print(f"\n配置:")
    print(f"  设备: {device}")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")
    print(f"  层数: {num_layers}")
    print(f"  批次大小: {batch_size}")
    print(f"  最大序列长度: {max_length}")
    print(f"  学习率: {learning_rate}")
    print(f"  最大训练步数: {max_steps:,}")
    print(f"  验证间隔: 每 {validation_interval:,} 步\n")
    
    # 初始化模型
    print("\n初始化模型...")
    model = NanoLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 4,
        max_seq_len=max_length,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    
    # CSV 文件路径
    train_csv_path = "/kaggle/input/tinystories-narrative-classification/train.csv"
    val_csv_path = "/kaggle/input/tinystories-narrative-classification/validation.csv"
    
    if not os.path.exists(train_csv_path):
        raise ValueError(f"训练数据CSV文件不存在: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise ValueError(f"验证数据CSV文件不存在: {val_csv_path}")
    
    train_loader, val_loader = create_data_loaders(
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        text_column="text"
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 创建迭代器
    train_iter = iter(train_loader)
    
    # 训练循环 - 基于step而非epoch
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70 + "\n")
    
    best_val_loss = float("inf")
    best_val_perplexity = float("inf")
    model.train()
    start_time = time.time()
    
    pbar = tqdm(total=max_steps, desc="训练进度")
    
    for step in range(max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            # 数据集迭代完成，重新创建迭代器（无限随机采样）
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # 执行训练step
        loss = train_step(model, batch, optimizer, criterion, device)
        # 计算训练困惑度
        train_perplexity = torch.exp(torch.tensor(loss)).item()
        
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss:.4f}", "ppl": f"{train_perplexity:.2f}"})
        
        # 记录到wandb
        wandb.log({
            "train_loss": loss,
            "train_perplexity": train_perplexity,
            "step": step + 1
        })
        
        # 验证和保存模型
        if (step + 1) % validation_interval == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed_time
            
            print(f"\n✓ Step {step + 1:,}/{max_steps:,} | 耗时: {elapsed_time:.1f}s | 速度: {steps_per_sec:.2f} steps/s")
            
            # 验证
            model.eval()
            val_loss, val_perplexity = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                num_steps=100,
                tokenizer=tokenizer,
                step_num=step + 1
            )
            model.train()
            
            print(f"  训练损失: {loss:.4f} | 困惑度: {train_perplexity:.2f}")
            print(f"  验证损失: {val_loss:.4f} | 困惑度: {val_perplexity:.2f}")
            
            # 记录验证指标
            wandb.log({
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "step": step + 1,
                "elapsed_time": elapsed_time,
            })
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_perplexity = val_perplexity
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f} | val_ppl: {val_perplexity:.2f})")
                wandb.save("best_model.pt")
            
            # 定期保存检查点
            checkpoint_path = f"checkpoint_step_{step + 1}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"  ✓ 保存检查点: {checkpoint_path}")
    
    pbar.close()
    
    total_elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"总耗时: {total_elapsed_time:.1f}s")
    print(f"平均速度: {max_steps / total_elapsed_time:.2f} steps/s")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳验证困惑度: {best_val_perplexity:.2f}")
    print("=" * 70)
    
    # 结束wandb记录
    wandb.finish()
    
    # 文本生成演示
    print("\n文本生成演示:")
    print("-" * 70)
    
    model.eval()
    
    # 测试提示词
    prompts = [
        "Once upon a time",
        "The little girl",
        "In the forest"
    ]
    
    for prompt_text in prompts:
        # 编码提示词
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        
        print(f"\n提示词: {prompt_text}")
        
        with torch.no_grad():
            generated_ids = model.generate(
                prompt=prompt_ids,
                max_length=50,
                temperature=0.8,
                top_k=50,
            )
        
        # 解码生成的文本 - 只使用生成部分的token IDs（跳过提示词部分）
        generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
        generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

