"""数据集工厂"""

from typing import Tuple, Dict, Any, Optional
from .base import BaseDataset
from .tinystories import TinyStoriesDataset
from .cluecorpus import CLUECorpusDataset


DATASET_REGISTRY = {
    "tinystories": TinyStoriesDataset,
    "cluecorpus": CLUECorpusDataset,
}


def create_dataset(dataset_name: str, dataset_config: Dict[str, Any], **kwargs) -> Tuple[BaseDataset, BaseDataset]:
    """
    数据集工厂函数
    
    Args:
        dataset_name: 数据集名称 ("tinystories" 或 "cluecorpus")
        dataset_config: 数据集配置字典，包含 source、dataset_id、path 等字段
        **kwargs: 传递给 load_datasets 的额外参数
        
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
    
    # 从配置中提取参数
    dataset_id = dataset_config.get("dataset_id", dataset_name)
    source = dataset_config.get("source", "local")
    path = dataset_config.get("path", None)
    
    # 合并额外参数
    all_kwargs = {**dataset_config, **kwargs}
    all_kwargs.pop("dataset_id", None)
    all_kwargs.pop("source", None)
    all_kwargs.pop("path", None)
    
    return dataset_class.load_datasets(
        dataset_id=dataset_id,
        source=source,
        path=path,
        **all_kwargs
    )
