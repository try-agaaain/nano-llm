"""模型训练脚本 - 使用TinyStories数据集和HuggingFace分词器"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import time
from tqdm import tqdm
import wandb
from model import NanoLLM
from tokenizer import load_or_train_tokenizer


class TinyStoriesDataset(IterableDataset):
    """TinyStories数据集包装器"""
    
    def __init__(self, tokenizer, max_length=512, split="train", num_samples=None):
        """
        Args:
            tokenizer: HuggingFace分词器
            max_length: 最大序列长度
            split: 数据集分割（train/validation）
            num_samples: 限制样本数量（调试用）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        
        # 加载TinyStories数据集
        print(f"正在加载TinyStories数据集 ({split} 分割)...")
        # self.dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        self.dataset = load_dataset("./dataset", split=split, streaming=False)
        
        if num_samples:
            self.dataset = self.dataset.take(num_samples)
    
    def __iter__(self):
        """迭代数据集"""
        for idx, example in enumerate(self.dataset):
            text = example["text"]
            
            # 分词并对齐到固定长度，方便 DataLoader 批量堆叠
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            # tokens["input_ids"] shape: (1, max_length)
            input_ids = tokens["input_ids"][0]
            
            # 需要至少2个token
            if len(input_ids) < 2:
                continue
            
            # 创建输入和目标
            # 输入：除了最后一个token
            # 目标：除了第一个token
            yield {
                "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
                "labels": torch.tensor(input_ids[1:], dtype=torch.long),
            }


def create_data_loaders(tokenizer, batch_size=32, max_length=512, num_samples_train=None, num_samples_val=None):
    """创建训练和验证数据加载器"""
    
    train_dataset = TinyStoriesDataset(
        tokenizer,
        max_length=max_length,
        split="train",
        num_samples=num_samples_train
    )
    
    val_dataset = TinyStoriesDataset(
        tokenizer,
        max_length=max_length,
        split="validation",
        num_samples=num_samples_val
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, max_steps=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        if max_steps and batch_idx >= max_steps:
            break
        
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
        
        total_loss += loss.item()
        num_batches += 1
        
        # 显示当前batch的loss
        pbar.set_postfix(loss=loss.item())
        
        # 记录到wandb
        wandb.log({
            "train_loss": loss.item(),
            "batch": batch_idx
        })
    
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, elapsed_time


def evaluate(model, val_loader, criterion, device, max_steps=None):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="验证中")
        for batch_idx, batch in enumerate(pbar):
            if max_steps and batch_idx >= max_steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=loss.item())
            
            # 记录到wandb
            wandb.log({
                "val_loss": loss.item()
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main():
    """主训练函数"""
    
    # 初始化wandb
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY未设置。请在环境变量中设置API密钥")
    wandb.login(key=wandb_api_key)
    wandb.init(
        project="nano-llm",
        name="TinyStories-training",
        config={
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4,
            "batch_size": 32,
            "max_length": 256,
            "learning_rate": 0.001,
            "num_epochs": 3,
        }
    )
    
    # 超参数
    d_model = 256
    num_heads = 8
    num_layers = 4
    num_epochs = 3
    learning_rate = 0.001
    batch_size = 32
    max_length = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 对于调试，可以限制样本数量
    num_samples_train = None  # 使用完整数据集
    num_samples_val = None
    max_steps_per_epoch = None  # 使用所有步骤
    
    print("=" * 70)
    print("NanoLLM 训练 - TinyStories数据集")
    print("=" * 70)
    
    # 加载分词器
    print("正在加载TinyStories分词器...")
    tokenizer = load_or_train_tokenizer(
        tokenizer_path="./tokenizer",
        vocab_size=8192,
        num_samples=50000,
        force_retrain=False,  # 改为False，只在需要时重新训练
        dataset_dir="./dataset"
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
    print(f"  Epochs: {num_epochs}\n")
    
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
    train_loader, val_loader = create_data_loaders(
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_samples_train=num_samples_train,
        num_samples_val=num_samples_val,
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # 训练
        train_loss, train_time = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_steps=max_steps_per_epoch,
        )
        
        # 验证
        val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            max_steps=10,
        )
        
        print(f"\nEpoch {epoch+1} 完成:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  耗时: {train_time:.2f}秒")
        
        # 记录epoch级别的指标到wandb
        wandb.log({
            "epoch": epoch + 1,
            "epoch_train_loss": train_loss,
            "epoch_val_loss": val_loss,
            "epoch_time": train_time
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            
            # 上传最佳模型到wandb
            wandb.save("best_model.pt")
    
    print("\n" + "=" * 70)
    print("训练完成!")
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
        # generated_ids 包含了原始提示词 + 新生成的部分
        # 我们只解码新生成的部分，从原始长度开始
        generated_only_ids = generated_ids[0, prompt_ids.size(1):].tolist()
        generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

