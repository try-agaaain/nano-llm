"""数据集基础接口"""

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
    def load_datasets(cls, data_dir: str, **kwargs) -> Tuple["BaseDataset", "BaseDataset"]:
        """
        加载训练集和验证集
        
        Args:
            data_dir: 数据集所在目录
            **kwargs: 其他参数
            
        Returns:
            (train_dataset, val_dataset) 元组
        """
        pass
    
    @abstractmethod
    def get_texts(self, num_samples: Optional[int] = None) -> List[str]:
        """
        获取文本数据用于分词器训练
        
        Args:
            num_samples: 样本数量，None表示全部
            
        Returns:
            文本列表
        """
        pass
    
    @abstractmethod
    def __iter__(self):
        """无限迭代数据集"""
        pass
