"""用于训练的数据集包装器 - 将迭代数据集适配到 DataLoader"""

import torch
from torch.utils.data import IterableDataset


class TokenizedDataset(IterableDataset):
    """包装数据集迭代器，提供 tokenized 输出
    
    优化说明：
    - 避免预加载整个数据集到内存，直接流式处理
    - 无限循环迭代数据集，避免内存溢出
    - 高效的 tensor 复制和分离
    """
    
    def __init__(self, dataset, tokenizer, max_length: int = 512):
        """
        Args:
            dataset: HuggingFace Dataset 或其他支持迭代的数据集
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __iter__(self):
        """无限迭代，返回 tokenized 数据"""
        while True:
            for item in self.dataset:
                # 从字典中提取文本（如果是字典）
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = item
                
                # 跳过无效文本
                if not text or not isinstance(text, str) or not text.strip():
                    continue
                
                tokens = self.tokenizer(
                    text.strip(),
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                input_ids = tokens["input_ids"][0]
                
                if len(input_ids) < 2:
                    continue
                
                yield {
                    "input_ids": input_ids[:-1].clone(),
                    "labels": input_ids[1:].clone(),
                }
