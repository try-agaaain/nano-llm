import random
from typing import Optional, List, Tuple
from modelscope.msdatasets import MsDataset
from .base import BaseDataset

class CLUECorpusDataset(BaseDataset):
    """优化后的 CLUE Corpus Small 数据集"""
    
    def __init__(self, data_dir: str, split: str = "train", train_ratio: float = 0.9):
        # 这里的 data_dir 将作为 modelscope 的缓存路径
        super().__init__(data_dir, split)
        self.train_ratio = train_ratio
        
        # 1. 直接使用 MsDataset 加载
        # cache_dir 会自动处理：存在则加载，不存在则下载
        # split='train' 是因为 CLUECorpusSmall 在 ModelScope 上通常只有 train 一个分片
        full_ds = MsDataset.load(
            'austenjs/ClueCorpusSmallDataset',
            subset_name='default', 
            split='train', 
            cache_dir=str(self.data_dir)
        )
        
        # 2. 逻辑分割（利用数据集的长度进行索引分割，而不是加载到 list）
        total_len = len(full_ds)
        split_idx = int(total_len * self.train_ratio)
        
        if self.split == "train":
            self.dataset = full_ds.select(range(0, split_idx))
        else:
            self.dataset = full_ds.select(range(split_idx, total_len))
            
        print(f"成功加载 {self.split} 集，规模: {len(self.dataset)}")

    @classmethod
    def load_datasets(cls, data_dir: str, train_ratio: float = 0.9) -> Tuple["CLUECorpusDataset", "CLUECorpusDataset"]:
        """
        加载训练集和验证集
        
        Args:
            data_dir: 数据集所在目录
            train_ratio: 训练集比例
            
        Returns:
            (train_dataset, val_dataset) 元组
        """
        train_dataset = cls(data_dir, split="train", train_ratio=train_ratio)
        val_dataset = cls(data_dir, split="validation", train_ratio=train_ratio)
        return train_dataset, val_dataset
    
    def get_texts(self, num_samples: Optional[int] = None) -> List[str]:
        """获取用于分词器训练的文本样本"""
        if num_samples is None:
            return [item['text'] for item in self.dataset]
        
        # 随机采样，避免全量转换成 list
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        return [self.dataset[i]['text'] for i in indices]

    def __iter__(self):
        """流式迭代，避免内存堆积"""
        while True:
            # 随机打乱索引进行迭代
            idx = random.randint(0, len(self.dataset) - 1)
            yield self.dataset[idx]['text']

    def __len__(self):
        return len(self.dataset)