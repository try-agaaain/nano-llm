"""TinyStories 数据集实现"""

import random
from typing import Optional, List, Tuple

from .base import BaseDataset


class TinyStoriesDataset(BaseDataset):
    """TinyStories 数据集 - 支持多种数据源加载"""
    
    def __init__(self, dataset_id: str, source: str = "local", path: str = None, split: str = "train", text_column: str = "text", **kwargs):
        """
        Args:
            dataset_id: 数据集ID
            source: 数据源类型 ("local", "modelscope", "huggingface")
            path: 本地路径（source为local时）或缓存路径
            split: 数据集分割类型，"train" 或 "validation"
            text_column: CSV 中文本列的名称（仅local源）
            **kwargs: 其他参数
        """
        super().__init__(path, split)

        self.dataset_id = dataset_id
        self.source = source
        self.text_column = text_column
        
        # 根据 source 选择加载方式
        if source == "local":
            self._load_from_local()
        elif source == "huggingface":
            self._load_from_huggingface(
                dataset_id=dataset_id,
                split=split,
                cache_dir=str(self.data_dir),
                text_column=text_column,
                **kwargs
            )
        elif source == "modelscope":
            self._load_from_modelscope(
                dataset_id=dataset_id,
                split=split,
                cache_dir=str(self.data_dir),
                text_column=text_column,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的数据源: {source}")
    
    @classmethod
    def load_datasets(cls, dataset_id: str, source: str = "local", path: str = None, text_column: str = "text", **kwargs) -> Tuple["TinyStoriesDataset", "TinyStoriesDataset"]:
        """
        加载训练集和验证集
        
        Args:
            dataset_id: 数据集ID
            source: 数据源类型
            path: 本地路径或缓存路径
            text_column: CSV 中文本列的名称
            **kwargs: 其他参数
            
        Returns:
            (train_dataset, val_dataset) 元组
        """
        train_dataset = cls(dataset_id, source=source, path=path, split="train", text_column=text_column, **kwargs)
        val_dataset = cls(dataset_id, source=source, path=path, split="validation", text_column=text_column, **kwargs)
        return train_dataset, val_dataset
    
    def _load_from_local(self):
        """从本地 CSV 文件加载"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")
        
        csv_filename = "train.csv" if self.split == "train" else "validation.csv"
        csv_file = self.data_dir / csv_filename
        
        if not csv_file.exists():
            raise ValueError(f"找不到 {csv_filename}: {csv_file}")
        
        try:
            # 使用 load_dataset 加载 CSV 文件
            self.dataset = load_dataset('csv', data_files=str(csv_file))
            # 'csv' 数据集返回 'train' split，需要获取第一个 split
            self.dataset = self.dataset['train']
        except Exception as e:
            raise ValueError(f"加载本地CSV失败: {e}")
        
        if len(self.dataset) == 0:
            raise ValueError("未加载到任何文本数据")
        
        # 如果指定的 text_column 不同，进行映射
        if self.text_column != "text" and self.text_column in self.dataset.column_names:
            self.dataset = self.dataset.map(
                lambda x: {"text": x[self.text_column]},
                remove_columns=[col for col in self.dataset.column_names if col != self.text_column]
            )
        
        print(f"成功加载 {self.split} 集（本地），规模: {len(self.dataset)}")
    
    def __len__(self):
        return len(self.dataset)
