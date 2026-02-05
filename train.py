"""模型训练脚本 - 使用TinyStories数据集或 Kaggle CSV 数据和HuggingFace分词器"""

import os
import csv
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
    """TinyStories数据集或CSV数据包装器"""
    
    def __init__(self, tokenizer, max_length=512, split="train", num_samples=None, csv_path=None, text_column="text"):
        """
        Args:
            tokenizer: HuggingFace分词器
            max_length: 最大序列长度
            split: 数据集分割（train/validation）- 仅对数据集使用
            num_samples: 限制样本数量（调试用）
            csv_path: CSV 文件路径（如果提供，优先于数据集）
            text_column: CSV 中文本列的名称
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.csv_path = csv_path
        self.text_column = text_column
        self.split = split
        self.data = []
        
        # 加载数据
        if csv_path and os.path.exists(csv_path):
            print(f"正在加载 CSV 数据集: {csv_path}")
            self._load_csv_data(csv_path)
        else:
            print(f"正在加载TinyStories数据集 ({split} 分割)...")
            self._load_hf_dataset(split)
    
    def _load_csv_data(self, csv_path):
        """从 CSV 文件加载数据"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for idx, row in enumerate(csv_reader):
                    if self.num_samples and idx >= self.num_samples:
                        break
                    
                    # 获取文本字段
                    if self.text_column not in row:
                        # 尝试寻找其他可能的列名
                        text = None
                        for col in ["text", "story", "content", "narrative"]:
                            if col in row:
                                text = row[col]
                                break
                        if not text:
                            text = next((v for v in row.values() if v), None)
                    else:
                        text = row[self.text_column]
                    
                    if text and isinstance(text, str) and text.strip():
                        self.data.append(text.strip())
            
            print(f"已加载 {len(self.data)} 个 CSV 样本")
        except Exception as e:
            print(f"加载 CSV 失败: {e}")
            raise
    
    def _load_hf_dataset(self, split):
        """从 HuggingFace 数据集加载数据"""
        try:
            dataset = load_dataset("./dataset", split=split, streaming=False)
            
            for idx, example in enumerate(dataset):
                if self.num_samples and idx >= self.num_samples:
                    break
                
                # 尝试多种可能的字段名
                story = None
                for field_name in ["text", "story", "content", "narrative", "sentence"]:
                    if field_name in example:
                        story = example[field_name]
                        break
                
                if story is None:
                    for key, value in example.items():
                        if isinstance(value, str):
                            story = value
                            break
                
                if story and isinstance(story, str) and story.strip():
                    self.data.append(story.strip())
            
            print(f"已加载 {len(self.data)} 个数据集样本")
        except Exception as e:
            print(f"加载 HuggingFace 数据集失败: {e}")
            # 如果加载失败，尝试从本地 CSV
            if not self.csv_path:
                raise
    
    def __iter__(self):
        """迭代数据集"""
        for text in self.data:
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


def create_data_loaders(
    tokenizer, 
    batch_size=32, 
    max_length=512, 
    num_samples_train=None, 
    num_samples_val=None,
    train_csv_path=None,
    val_csv_path=None,
    text_column="text"
):
    """创建训练和验证数据加载器"""
    
    train_dataset = TinyStoriesDataset(
        tokenizer,
        max_length=max_length,
        split="train",
        num_samples=num_samples_train,
        csv_path=train_csv_path,
        text_column=text_column
    )
    
    val_dataset = TinyStoriesDataset(
        tokenizer,
        max_length=max_length,
        split="validation",
        num_samples=num_samples_val,
        csv_path=val_csv_path,
        text_column=text_column
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
            "d_model": 384,
            "num_heads": 8,
            "num_layers": 6,
            "batch_size": 64,
            "max_length": 512,
            "learning_rate": 0.0008,
            "num_epochs": 4,
        }
    )
    
    # 超参数
    d_model = 384
    num_heads = 8
    num_layers = 6
    num_epochs = 4
    learning_rate = 0.0008
    batch_size = 64
    max_length = 512
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
    
    # 检查 Kaggle CSV 路径（Kaggle 上的路径）
    train_csv_path = "/kaggle/input/tinystories-narrative-classification/train.csv"
    val_csv_path = "/kaggle/input/tinystories-narrative-classification/validation.csv"
    
    # 如果本地不存在 CSV，尝试使用默认数据集
    if not os.path.exists(train_csv_path):
        print(f"   ℹ️  CSV 路径不存在，将使用 HuggingFace 数据集")
        train_csv_path = None
        val_csv_path = None
    
    train_loader, val_loader = create_data_loaders(
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_samples_train=num_samples_train,
        num_samples_val=num_samples_val,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        text_column="text"
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

