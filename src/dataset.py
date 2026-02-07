"""统一的数据集加载模块 - 支持 TinyStories CSV 数据"""

import os
import csv
import random
import tempfile
from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm


class TinyStoriesDataset(IterableDataset):
    """
    TinyStories 数据集 - IterableDataset 实现
    支持从 CSV 文件加载，无限迭代和随机采样
    """
    
    def __init__(
        self, 
        tokenizer, 
        csv_path: str,
        max_length: int = 512,
        text_column: str = "text",
    ):
        """
        Args:
            tokenizer: HuggingFace 分词器
            csv_path: CSV 文件路径
            max_length: 最大序列长度
            text_column: CSV 中文本列的名称（默认 "text"）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.csv_path = csv_path
        self.text_column = text_column
        self.data = []
        
        if not csv_path or not os.path.exists(csv_path):
            raise ValueError(f"必须指定有效的CSV文件路径: {csv_path}")
        
        self._load_csv_data(csv_path)
    
    def _load_csv_data(self, csv_path: str):
        """从 CSV 文件加载文本数据"""
        try:
            print(f"正在加载 CSV 数据集: {csv_path}")
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    # 尝试获取指定的文本列
                    if self.text_column in row:
                        text = row[self.text_column]
                    else:
                        # 如果指定列不存在，尝试其他常见列名
                        text = None
                        for col in ["text", "story", "content", "narrative"]:
                            if col in row:
                                text = row[col]
                                break
                        if not text:
                            text = next((v for v in row.values() if v), None)
                    
                    if text and isinstance(text, str) and text.strip():
                        self.data.append(text.strip())
            
            print(f"✅ 已加载 {len(self.data)} 个样本")
        except Exception as e:
            print(f"❌ 加载 CSV 失败: {e}")
            raise
    
    def __iter__(self):
        """无限迭代数据集，每次随机采样"""
        while True:
            # 随机选择一个样本（可能重复）
            text = random.choice(self.data)
            
            # 分词并对齐到固定长度
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


def load_texts_from_csv(
    csv_path: str,
    text_column: str = "text",
    num_samples: Optional[int] = None,
    clean_text: bool = True,
) -> List[str]:
    """
    从 CSV 文件加载文本数据（用于分词器训练）
    
    Args:
        csv_path: CSV 文件路径
        text_column: CSV 中文本列的名称
        num_samples: 要加载的样本数量（None 表示加载全部）
        clean_text: 是否清理文本中的换行符
    
    Returns:
        List[str]: 文本列表
    """
    if not csv_path or not os.path.exists(csv_path):
        raise ValueError(f"必须指定有效的CSV文件路径: {csv_path}")
    
    texts = []
    
    try:
        print(f"正在从 CSV 加载文本数据: {csv_path}")
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in tqdm(csv_reader, desc="加载文本"):
                if num_samples and len(texts) >= num_samples:
                    break
                
                # 尝试获取指定的文本列
                if text_column in row:
                    text = row[text_column]
                else:
                    # 如果指定列不存在，尝试其他常见列名
                    text = None
                    for col in ["text", "story", "content", "narrative"]:
                        if col in row:
                            text = row[col]
                            break
                    if not text:
                        text = next((v for v in row.values() if v), None)
                
                if text and isinstance(text, str) and text.strip():
                    if clean_text:
                        # 清理文本中的换行符
                        cleaned_text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
                    else:
                        cleaned_text = text.strip()
                    texts.append(cleaned_text)
        
        print(f"✅ 已加载 {len(texts)} 个文本样本")
        return texts
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise


def create_training_file_from_csv(
    csv_path: str,
    output_path: str,
    text_column: str = "text",
    num_samples: Optional[int] = None,
) -> str:
    """
    从 CSV 创建训练用的临时文本文件（用于 tokenizer.train()）
    
    Args:
        csv_path: CSV 文件路径
        output_path: 输出文件路径
        text_column: CSV 中文本列的名称
        num_samples: 要处理的样本数量
    
    Returns:
        str: 临时文件路径
    """
    texts = load_texts_from_csv(
        csv_path=csv_path,
        text_column=text_column,
        num_samples=num_samples,
        clean_text=True,
    )
    
    # 写入临时文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n\n")
    
    print(f"✅ 已创建训练文件: {output_path}")
    return output_path
