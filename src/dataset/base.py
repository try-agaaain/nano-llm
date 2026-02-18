"""数据集基础接口"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from pathlib import Path


class BaseDataset(ABC):
    """所有数据集的基类"""
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: 数据集所在目录
            split: 数据集分割类型，"train" 或 "validation"
        """
        self.data_dir = Path(data_dir)
        self.split = split
        if not self.data_dir.exists():
            raise ValueError(f"数据集目录不存在: {data_dir}")
    
    @classmethod
    @abstractmethod
    def load_datasets(cls, dataset_id: str, source: str = "local", path: str = None, **kwargs) -> Tuple["BaseDataset", "BaseDataset"]:
        """
        加载训练集和验证集
        
        Args:
            dataset_id: 数据集ID（远程平台ID或本地标识）
            source: 数据源类型 ("local", "modelscope", "huggingface")
            path: 本地文件系统路径
            **kwargs: 其他参数
            
        Returns:
            (train_dataset, val_dataset) 元组
        """
        pass
    
    def get_texts(self, num_samples: Optional[int] = None) -> List[str]:
        """
        获取文本数据用于分词器训练（统一接口）
        
        Args:
            num_samples: 样本数量，None表示全部
            
        Returns:
            文本列表
        """
        if num_samples is None:
            return [item['text'] if isinstance(item, dict) else item for item in self.dataset]
        
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        return [
            (self.dataset[i]['text'] if isinstance(self.dataset[i], dict) else self.dataset[i])
            for i in indices
        ]
    
    def __iter__(self):
        """无限迭代数据集"""
        while True:
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            yield item['text'] if isinstance(item, dict) else item
    
    def _load_from_huggingface(self, dataset_id: str, split: str = "train", cache_dir: str = None, train_ratio: float = None, text_column: str = None, **kwargs):
        """从 HuggingFace 加载数据集（通用方法）"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")

        split_name = split if split in ["train", "validation", "test"] else "train"
        full_ds = load_dataset(cache_dir)
        
        # 如果指定了 text_column，则只保留该列的数据
        if text_column:
            full_ds = full_ds.map(
                lambda x: {"text": x[text_column]},
                remove_columns=[col for col in full_ds.column_names if col != text_column]
            )
        
        if train_ratio is not None:
            # 需要进行 train/val 分割
            total_len = len(full_ds)
            split_idx = int(total_len * train_ratio)
            
            if split == "train":
                self.dataset = full_ds.select(range(0, split_idx))
            else:
                self.dataset = full_ds.select(range(split_idx, total_len))
        else:
            # 无分割，直接使用
            self.dataset = full_ds
        
        print(f"成功加载 {split} 集（HuggingFace），规模: {len(self.dataset)}")
    
    def _load_from_modelscope(self, dataset_id: str, split: str = "train", cache_dir: str = None, train_ratio: float = None, text_column: str = None, **kwargs):
        """从 ModelScope 加载数据集（通用方法）"""
        try:
            from modelscope.msdatasets import MsDataset
        except ImportError:
            raise ImportError("请安装 modelscope: pip install modelscope")
        
        split_name = split if split in ["train", "validation", "test"] else "train"
        full_ds = MsDataset.load(
            dataset_id,
            subset_name=kwargs.get('subset_name', 'default'),
            split='train',
            cache_dir=cache_dir
        )
        
        # 如果指定了 text_column，则只保留该列的数据
        if text_column:
            full_ds = full_ds.map(
                lambda x: {"text": x[text_column]},
                remove_columns=[col for col in full_ds.column_names if col != text_column]
            )
        
        if train_ratio is not None:
            # 需要进行 train/val 分割
            total_len = len(full_ds)
            split_idx = int(total_len * train_ratio)
            
            if split == "train":
                self.dataset = full_ds.select(range(0, split_idx))
            else:
                self.dataset = full_ds.select(range(split_idx, total_len))
        else:
            # 无分割，直接使用
            self.dataset = full_ds
        
        print(f"成功加载 {split} 集（ModelScope），规模: {len(self.dataset)}")
    
    @abstractmethod
    def __len__(self):
        """返回数据集大小"""
        pass
