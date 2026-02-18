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

from src.model import NanoLLM
from src.tokenizer import load_or_train_tokenizer
from src.dataset import create_dataset, TokenizedDataset
from src.utils.wandb_utils import WandbManager


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        # 默认查找项目根目录的config.yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def train_step(model, batch, optimizer, criterion, device):
    """执行单个训练step"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    optimizer.zero_grad(set_to_none=True)  # 使用 set_to_none=True 更高效
    logits = model(input_ids)
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # 立即获取标量值并释放计算图
    loss_val = loss.item()
    del logits, loss  # 显式删除以帮助内存回收
    
    return loss_val


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


def train(output_dir: str = "./output", config_path: str = None, profile_interval: int = 500):
    """主训练函数
    
    Args:
        output_dir: 输出目录
        config_path: 配置文件路径
        profile_interval: 时间分析间隔（步数），0表示不启用，建议500
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = load_config(config_path)
    training_config = config.get("training", {})
    wandb_config = config.get("wandb", {})
    dataset_config = config.get("dataset", {})
    
    # 从配置中提取参数
    d_model = training_config.get("d_model")
    num_heads = training_config.get("num_heads")
    num_layers = training_config.get("num_layers")
    d_ff_multiplier = training_config.get("d_ff_multiplier")
    max_length = training_config.get("max_length")
    learning_rate = training_config.get("learning_rate")
    batch_size = training_config.get("batch_size")
    max_steps = training_config.get("max_steps")
    validation_interval = training_config.get("validation_interval")
    vocab_size = training_config.get("vocab_size")
    num_samples = training_config.get("num_samples")
    
    # 数据集配置
    dataset_select = dataset_config.get("select", "tinystories")
    dataset_configs = dataset_config.get("configs", {})
    current_dataset_config = dataset_configs.get(dataset_select)

    print(f"配置已加载 | dataset={dataset_select} | d_model={d_model} | num_heads={num_heads} | num_layers={num_layers}")

    # 初始化WandbManager（自动登陆和初始化run）
    try:
        wandb_manager = WandbManager(config_path=str(config_path))
    except Exception as e:
        raise ValueError(f"初始化W&B失败: {e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据和分词器
    print(f"Loading dataset '{dataset_select}'...")
    train_dataset_raw, val_dataset_raw = create_dataset(current_dataset_config)
    
    print("Loading tokenizer...")
    tokenizer_path = output_path / "tokenizer"
    tokenizer = load_or_train_tokenizer(
        tokenizer_path=str(tokenizer_path),
        dataset=train_dataset_raw,  # 使用训练集训练分词器
        vocab_size=vocab_size,
        num_samples=num_samples,
        force_retrain=False
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
    
    # 用于累积指标以减少 wandb 日志频率
    log_interval = 10  # 每10步记录一次
    accumulated_metrics = {"train_loss": [], "train_perplexity": [], "time_per_sample_ms": []}
    
    for step in range(max_steps):
        if device.type == "cuda" and (step + 1) % 100 == 0:
            torch.cuda.empty_cache()
        
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
        
        # ========== 累积指标 ==========
        accumulated_metrics["train_loss"].append(loss)
        accumulated_metrics["train_perplexity"].append(train_ppl)
        accumulated_metrics["time_per_sample_ms"].append(time_per_sample_ms)
        
        # ========== W&B日志（批量记录以减少同步开销）==========
        if (step + 1) % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss:.4f}", "ppl": f"{train_ppl:.2f}", "time/sample (ms)": f"{time_per_sample_ms:.2f}"})
            # 记录平均值
            avg_metrics = {
                "train_loss": sum(accumulated_metrics["train_loss"]) / len(accumulated_metrics["train_loss"]),
                "train_perplexity": sum(accumulated_metrics["train_perplexity"]) / len(accumulated_metrics["train_perplexity"]),
                "time_per_sample_ms": sum(accumulated_metrics["time_per_sample_ms"]) / len(accumulated_metrics["time_per_sample_ms"]),
            }
            wandb_manager.log(avg_metrics, step=step + 1)
            # 清空累积
            accumulated_metrics = {"train_loss": [], "train_perplexity": [], "time_per_sample_ms": []}
        
        # ========== 验证和保存 ==========
        if (step + 1) % validation_interval == 0:
            elapsed = time.time() - start_time
            val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
            model.train()
            
            print(f"\nStep {step+1}/{max_steps} | {elapsed:.1f}s | Train: {loss:.4f}/{train_ppl:.2f} | Val: {val_loss:.4f}/{val_ppl:.2f}")
            wandb_manager.log({"val_loss": val_loss, "val_perplexity": val_ppl}, step=step + 1)
            
            # 保存检查点
            checkpoint_path = output_path / f"lastest_model.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, str(checkpoint_path))
    
    pbar.close()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s | Best val_loss: {best_val_loss:.4f}")
    
    # 上传最佳模型到 W&B
    try:
        model_save_path = output_path / "lastest_model.pt"
        wandb_manager.upload_model(
            model_path=str(model_save_path),
            version_notes=f"训练完成 | val_loss={best_val_loss:.4f} | steps={max_steps}"
        )
    except Exception as e:
        print(f"上传模型到 W&B 失败: {e}")
    
    # 结束 W&B 运行
    wandb_manager.finish()


if __name__ == "__main__":
    import sys
    import argparse
    
    workspace_dir = Path(__file__).parent.parent  # nano-llm目录
    config_path = workspace_dir / "config.yaml"
    
    # 支持命令行参数
    parser = argparse.ArgumentParser(description="NanoLLM 训练脚本")
    parser.add_argument("--profile", type=int, default=0, 
                        help="时间分析间隔（步数），0表示不启用，建议500")
    parser.add_argument("--config", type=str, default=str(config_path),
                        help="配置文件路径")
    parser.add_argument("--output", type=str, default="./output",
                        help="输出目录")
    
    args = parser.parse_args()
    
    train(output_dir=args.output, config_path=args.config, profile_interval=args.profile)

