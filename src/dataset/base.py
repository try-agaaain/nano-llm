"""数据集基础接口"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


class BaseDataset(ABC):
    """所有数据集的基类"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 数据集所在目录
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"数据集目录不存在: {data_dir}")
    
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
