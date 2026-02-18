"""数据集工厂"""

from typing import Tuple
from .base import BaseDataset
from .tinystories import TinyStoriesDataset
from .cluecorpus import CLUECorpusDataset


DATASET_REGISTRY = {
    "tinystories": TinyStoriesDataset,
    "cluecorpus": CLUECorpusDataset,
}


def create_dataset(dataset_name: str, data_dir: str, **kwargs) -> Tuple[BaseDataset, BaseDataset]:
    """
    数据集工厂函数
    
    Args:
        dataset_name: 数据集名称 ("tinystories" 或 "cluecorpus")
        data_dir: 数据集所在目录
        **kwargs: 传递给数据集 load_datasets 的额外参数
        
    Returns:
        (train_dataset, val_dataset) 元组
        
    Raises:
        ValueError: 不支持的数据集名称
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"不支持的数据集: {dataset_name}。可用数据集: {available}")
    
    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class.load_datasets(data_dir, **kwargs)
