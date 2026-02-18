import random
from typing import Optional, List, Tuple
from pathlib import Path
from .base import BaseDataset

class CLUECorpusDataset(BaseDataset):
    """CLUE Corpus 数据集"""
    
    def __init__(self, dataset_id: str, source: str = "local", path: str = None, split: str = "train", train_ratio: float = 0.9, **kwargs):
        """
        Args:
            dataset_id: 数据集ID（HF/MS格式 或 本地标识）
            source: 数据源类型 ("local", "modelscope", "huggingface")
            path: 本地路径或缓存路径
            split: 数据集分割类型，"train" 或 "validation"
            train_ratio: 训练集比例
            **kwargs: 其他参数
        """
        super().__init__(path, split)
        
        self.dataset_id = dataset_id
        self.source = source
        self.train_ratio = train_ratio
        
        # 根据 source 选择加载方式
        if source == "modelscope":
            self._load_from_modelscope(
                dataset_id=dataset_id,
                split=split,
                cache_dir=str(self.data_dir),
                train_ratio=train_ratio,
                **kwargs
            )
        elif source == "huggingface":
            self._load_from_huggingface(
                dataset_id=dataset_id,
                split=split,
                cache_dir=str(self.data_dir),
                train_ratio=train_ratio,
                **kwargs
            )
        elif source == "local":
            self._load_from_local(**kwargs)
        else:
            raise ValueError(f"不支持的数据源: {source}")
    
    @classmethod
    def load_datasets(cls, dataset_id: str, source: str = "local", path: str = None, train_ratio: float = 0.9, **kwargs) -> Tuple["CLUECorpusDataset", "CLUECorpusDataset"]:
        """
        加载训练集和验证集
        
        Args:
            dataset_id: 数据集ID
            source: 数据源类型
            path: 本地路径或缓存路径
            train_ratio: 训练集比例
            **kwargs: 其他参数
            
        Returns:
            (train_dataset, val_dataset) 元组
        """
        train_dataset = cls(dataset_id, source=source, path=path, split="train", train_ratio=train_ratio, **kwargs)
        val_dataset = cls(dataset_id, source=source, path=path, split="validation", train_ratio=train_ratio, **kwargs)
        return train_dataset, val_dataset
    
    def _load_from_local(self, **kwargs):
        """从本地文件加载（如 CLUECorpusSmall.txt）"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")
        
        # 优先查找 .txt 文件
        txt_file = self.data_dir / "CLUECorpusSmall.txt"
        if not txt_file.exists():
            txt_file = self.data_dir / f"{self.dataset_id}.txt"
        
        if not txt_file.exists():
            raise ValueError(f"找不到本地数据文件: {txt_file}")
        
        try:
            # 使用 load_dataset 加载文本文件
            full_ds = load_dataset('text', data_files=str(txt_file))
            # 'text' 数据集返回 'train' split，需要获取第一个 split
            full_ds = full_ds['train']
        except Exception as e:
            raise ValueError(f"加载本地文件失败: {e}")
        
        # 逻辑分割
        total_len = len(full_ds)
        split_idx = int(total_len * self.train_ratio)
        
        if self.split == "train":
            self.dataset = full_ds.select(range(0, split_idx))
        else:
            self.dataset = full_ds.select(range(split_idx, total_len))
        
        print(f"成功加载 {self.split} 集（本地），规模: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)