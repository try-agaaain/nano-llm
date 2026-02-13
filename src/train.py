"""模型训练脚本"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
import wandb

from src.model import NanoLLM
from src.tokenizer import load_or_train_tokenizer
from src.dataset import TinyStoriesDataset, TokenizedDataset


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        # 默认查找项目根目录的config.yaml
        config_path = Path(__file__).parent.parent.parent / "nano-llm" / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def train_step(model, batch, optimizer, criterion, device):
    """执行单个训练step"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def evaluate(model, val_loader, criterion, device, num_steps=100):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_steps):
            try:
                batch = next(iter(val_loader))
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
            except StopIteration:
                break
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def train(dataset_dir: str, output_dir: str = "./output", config_path: str = None):
    """主训练函数"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_config(config_path)
    training_config = config.get("training", {})
    
    # 从配置中提取参数
    d_model = training_config.get("d_model", 384)
    num_heads = training_config.get("num_heads", 8)
    num_layers = training_config.get("num_layers", 8)
    d_ff_multiplier = training_config.get("d_ff_multiplier", 4)
    max_length = training_config.get("max_length", 1024)
    learning_rate = training_config.get("learning_rate", 0.0001)
    batch_size = training_config.get("batch_size", 16)
    max_steps = training_config.get("max_steps", 20000)
    validation_interval = training_config.get("validation_interval", 1000)
    vocab_size = training_config.get("vocab_size", 8192)
    num_samples = training_config.get("num_samples", 50000)
    
    print(f"配置已加载 | d_model={d_model} | num_heads={num_heads} | num_layers={num_layers}")
    
    # 初始化wandb
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="nano-llm",
            name="TinyStories-training",
            dir=str(output_path / "wandb"),
            config={
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "batch_size": batch_size,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
            }
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据和分词器
    print(f"Loading dataset from {dataset_dir}")
    train_dataset_raw, val_dataset_raw = TinyStoriesDataset.load_datasets(dataset_dir)
    
    print("Loading tokenizer...")
    tokenizer_path = output_path / "tokenizer"
    tokenizer = load_or_train_tokenizer(
        tokenizer_path=str(tokenizer_path),
        dataset=train_dataset_raw,  # 使用训练集训练分词器
        vocab_size=vocab_size,
        num_samples=num_samples,
        force_retrain=False,
    )
    
    print(f"Vocab size: {tokenizer.vocab_size} | Device: {device}")
    
    # 创建数据加载器
    train_dataset = TokenizedDataset(train_dataset_raw, tokenizer, max_length)
    val_dataset = TokenizedDataset(val_dataset_raw, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = NanoLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=d_model * d_ff_multiplier,
        max_seq_len=max_length,
    ).to(device)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print(f"\nTraining for {max_steps:,} steps...")
    best_val_loss = float("inf")
    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    
    pbar = tqdm(total=max_steps, desc="Training")
    step_start_time = time.time()
    for step in range(max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = train_step(model, batch, optimizer, criterion, device)
        step_time = time.time() - step_start_time
        step_start_time = time.time()
        
        train_ppl = torch.exp(torch.tensor(loss)).item()
        
        time_per_sample_ms = (step_time / batch_size) * 1000
        
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss:.4f}", "ppl": f"{train_ppl:.2f}", "time/sample (ms)": f"{time_per_sample_ms:.2f}"})
        
        if wandb_api_key:
            wandb.log({
                "train_loss": loss, 
                "train_perplexity": train_ppl, 
                "time_per_sample_ms": time_per_sample_ms,
                "step": step + 1
            })
        
        # 验证和保存
        if (step + 1) % validation_interval == 0:
            elapsed = time.time() - start_time
            val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
            model.train()
            
            print(f"\nStep {step+1}/{max_steps} | {elapsed:.1f}s | Train: {loss:.4f}/{train_ppl:.2f} | Val: {val_loss:.4f}/{val_ppl:.2f}")
            
            if wandb_api_key:
                wandb.log({"val_loss": val_loss, "val_perplexity": val_ppl, "step": step + 1})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = output_path / "best_model.pt"
                torch.save(model.state_dict(), str(model_save_path))
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # 保存检查点
            checkpoint_path = output_path / f"checkpoint_step_{step + 1}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, str(checkpoint_path))
    
    pbar.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s | Best val_loss: {best_val_loss:.4f}")
    
    if wandb_api_key:
        wandb.finish()


if __name__ == "__main__":
    workspace_dir = Path(__file__).parent.parent  # nano-llm目录
    dataset_dir = workspace_dir / "dataset" / "tinystories-narrative-classification"
    config_path = workspace_dir / "config.yaml"
    train(dataset_dir, config_path=str(config_path))

