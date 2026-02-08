"""用于训练的数据集包装器 - 将迭代数据集适配到 DataLoader"""

import random
import torch
from torch.utils.data import IterableDataset


class TokenizedDataset(IterableDataset):
    """包装数据集迭代器，提供 tokenized 输出"""
    
    def __init__(self, dataset, tokenizer, max_length: int = 512):
        """
        Args:
            dataset: BaseDataset 实例
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __iter__(self):
        """无限迭代，返回 tokenized 数据"""
        for text in self.dataset:
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids = tokens["input_ids"][0]
            
            if len(input_ids) < 2:
                continue
            
            yield {
                "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
                "labels": torch.tensor(input_ids[1:], dtype=torch.long),
            }
