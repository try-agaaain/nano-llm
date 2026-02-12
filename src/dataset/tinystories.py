"""TinyStories 数据集实现"""

import csv
import random
from typing import Optional, List, Tuple
from tqdm import tqdm

from .base import BaseDataset


class TinyStoriesDataset(BaseDataset):
    """TinyStories 数据集 - 从 CSV 文件加载"""
    
    def __init__(self, data_dir: str, split: str = "train", text_column: str = "text"):
        """
        Args:
            data_dir: 数据集所在目录（应包含 train.csv 和 validation.csv）
            split: 数据集分割类型，"train" 或 "validation"
            text_column: CSV 中文本列的名称
        """
        super().__init__(data_dir, split)
        self.text_column = text_column
        self.texts = []
        self._load_texts()
    
    @classmethod
    def load_datasets(cls, data_dir: str, text_column: str = "text") -> Tuple["TinyStoriesDataset", "TinyStoriesDataset"]:
        """
        加载训练集和验证集
        
        Args:
            data_dir: 数据集所在目录
            text_column: CSV 中文本列的名称
            
        Returns:
            (train_dataset, val_dataset) 元组
        """
        train_dataset = cls(data_dir, split="train", text_column=text_column)
        val_dataset = cls(data_dir, split="validation", text_column=text_column)
        return train_dataset, val_dataset
    
    def _load_texts(self):
        """从 CSV 文件加载所有文本"""
        csv_filename = "train.csv" if self.split == "train" else "validation.csv"
        csv_file = self.data_dir / csv_filename
        
        if not csv_file.exists():
            raise ValueError(f"找不到 {csv_filename}: {csv_file}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in tqdm(csv_reader, desc=f"加载TinyStories-{self.split}"):
                    text = row.get(self.text_column, "").strip()
                    if text:
                        self.texts.append(text)
        except Exception as e:
            raise ValueError(f"加载 CSV 失败: {e}")
        
        if not self.texts:
            raise ValueError("未加载到任何文本数据")
    
    def get_texts(self, num_samples: Optional[int] = None) -> List[str]:
        """获取文本数据用于分词器训练"""
        if num_samples is None or num_samples >= len(self.texts):
            return self.texts
        return random.sample(self.texts, num_samples)
    
    def __iter__(self):
        """无限迭代，每次随机返回一个样本"""
        while True:
            yield random.choice(self.texts)
