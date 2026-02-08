"""TinyStories 数据集实现"""

import csv
import random
from typing import Optional, List
from tqdm import tqdm

from .base import BaseDataset


class TinyStoriesDataset(BaseDataset):
    """TinyStories 数据集 - 从 CSV 文件加载"""
    
    def __init__(self, data_dir: str, text_column: str = "text"):
        """
        Args:
            data_dir: 数据集所在目录（应包含 train.csv 和 validation.csv）
            text_column: CSV 中文本列的名称
        """
        super().__init__(data_dir)
        self.text_column = text_column
        self.texts = []
        self._load_texts()
    
    def _load_texts(self):
        """从 CSV 文件加载所有文本"""
        csv_file = self.data_dir / "train.csv"
        
        if not csv_file.exists():
            raise ValueError(f"找不到 train.csv: {csv_file}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in tqdm(csv_reader, desc="加载TinyStories"):
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
