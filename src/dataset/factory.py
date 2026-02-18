"""数据集工厂 - 统一加载接口"""

import random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

# 硬编码随机种子用于可复现性
RANDOM_SEED = 42


def load_from_local(
    data_dir: str,
    filetype: str,
    train_ratio: float = 0.9,
) -> Tuple[Any, Any]:
    """
    从本地文件加载数据集
    
    Args:
        data_dir: 数据集目录路径
        filetype: 文件类型 ("csv", "txt", "json" 等)
        train_ratio: 训练集比例（0-1）
        
    Returns:
        (train_dataset, val_dataset) 元组
        
    Raises:
        ImportError: 缺少 datasets 包
        ValueError: 有关文件或参数的错误
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"数据集目录不存在: {data_dir}")
    
    # 根据 filetype 过滤文件
    data_files = sorted([str(f) for f in data_dir.glob(f"*.{filetype}")])
    
    if not data_files:
        raise ValueError(f"在 {data_dir} 中找不到 *.{filetype} 文件")
    
    print(f"发现 {len(data_files)} 个文件: {data_files[:3]}{'...' if len(data_files) > 3 else ''}")
    
    # 使用 load_dataset 加载
    try:
        dataset = load_dataset(filetype, data_files=data_files)
        # 取第一个 split（通常返回 {'train': ...}）
        full_dataset = dataset[list(dataset.keys())[0]]
    except Exception as e:
        raise ValueError(f"加载本地文件失败: {e}")
    
    # 使用固定种子进行确定性随机分割
    total_len = len(full_dataset)
    train_size = int(total_len * train_ratio)
    
    random.seed(RANDOM_SEED)
    train_indices = sorted(random.sample(range(total_len), train_size))
    val_indices = sorted(list(set(range(total_len)) - set(train_indices)))
    
    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)
    
    print(f"成功加载本地数据集，训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def load_from_huggingface(
    dataset_id: str,
    data_dir: str,
    train_ratio: float = 0.9,
) -> Tuple[Any, Any]:
    """
    从 HuggingFace 加载数据集
    
    Args:
        dataset_id: 数据集 ID (如 "wikitext")
        data_dir: 缓存目录路径
        train_ratio: 训练集比例（0-1）
        
    Returns:
        (train_dataset, val_dataset) 元组
        
    Raises:
        ImportError: 缺少 datasets 包
        ValueError: 加载失败
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")
    
    try:
        dataset = load_dataset(dataset_id, cache_dir=data_dir)
        # 取第一个 split
        full_dataset = dataset[list(dataset.keys())[0]]
    except Exception as e:
        raise ValueError(f"从 HuggingFace 加载数据集失败: {e}")
    
    # 使用固定种子进行确定性随机分割
    total_len = len(full_dataset)
    train_size = int(total_len * train_ratio)
    
    random.seed(RANDOM_SEED)
    train_indices = sorted(random.sample(range(total_len), train_size))
    val_indices = sorted(list(set(range(total_len)) - set(train_indices)))
    
    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)
    
    print(f"成功加载 HuggingFace 数据集 '{dataset_id}'，训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def load_from_modelscope(
    dataset_id: str,
    data_dir: str,
    train_ratio: float = 0.9,
) -> Tuple[Any, Any]:
    """
    从 ModelScope 加载数据集
    
    Args:
        dataset_id: 数据集 ID (如 "damo/zh-en-machine-translation")
        data_dir: 缓存目录路径
        train_ratio: 训练集比例（0-1）
        
    Returns:
        (train_dataset, val_dataset) 元组
        
    Raises:
        ImportError: 缺少 modelscope 包
        ValueError: 加载失败
    """
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise ImportError("请安装 modelscope: pip install modelscope")
    
    try:
        full_dataset = MsDataset.load(
            dataset_id,
            cache_dir=data_dir,
            split='train'
        )
    except Exception as e:
        raise ValueError(f"从 ModelScope 加载数据集失败: {e}")
    
    # 使用固定种子进行确定性随机分割
    total_len = len(full_dataset)
    train_size = int(total_len * train_ratio)
    
    random.seed(RANDOM_SEED)
    train_indices = sorted(random.sample(range(total_len), train_size))
    val_indices = sorted(list(set(range(total_len)) - set(train_indices)))
    
    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)
    
    print(f"成功加载 ModelScope 数据集 '{dataset_id}'，训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_dataset(
    dataset_config: Dict[str, Any],
    split: Optional[str] = None
) -> Union[Tuple[Any, Any], Any]:
    """
    根据单个数据集配置创建数据集的工厂函数
    
    Args:
        dataset_config: 单个数据集的配置字典，应包含以下字段：
            - source: "local", "huggingface", 或 "modelscope"
            - data_dir: 数据目录或缓存目录
            - train_ratio: 训练集比例（默认 0.9）
            对于 local 源额外需要：
            - type: 文件类型（csv, txt, json 等）
            对于 huggingface 或 modelscope 源额外需要：
            - dataset_id: 数据集 ID
        split: 若指定 "train" 或 "validation"，则只返回对应的数据集；否则返回 (train, val) 元组
        
    Returns:
        如果 split 为 None，返回 (train_dataset, val_dataset) 元组
        否则返回指定 split 的数据集
        
    Raises:
        ValueError: 配置格式错误或参数不完整
    """
    source = dataset_config.get("source", "local").lower()
    train_ratio = dataset_config.get("train_ratio", 0.9)
    
    # 根据 source 选择加载函数
    if source == "local":
        data_dir = dataset_config.get("data_dir")
        filetype = dataset_config.get("type")
        
        if not data_dir or not filetype:
            raise ValueError(f"本地数据集配置缺少必要字段 (data_dir, type)")
        
        train_ds, val_ds = load_from_local(data_dir, filetype, train_ratio)
    
    elif source == "huggingface":
        dataset_id = dataset_config.get("dataset_id")
        data_dir = dataset_config.get("data_dir")
        
        if not dataset_id or not data_dir:
            raise ValueError(f"HuggingFace 数据集配置缺少必要字段 (dataset_id, data_dir)")
        
        train_ds, val_ds = load_from_huggingface(dataset_id, data_dir, train_ratio)
    
    elif source == "modelscope":
        dataset_id = dataset_config.get("dataset_id")
        data_dir = dataset_config.get("data_dir")
        
        if not dataset_id or not data_dir:
            raise ValueError(f"ModelScope 数据集配置缺少必要字段 (dataset_id, data_dir)")
        
        train_ds, val_ds = load_from_modelscope(dataset_id, data_dir, train_ratio)
    
    else:
        raise ValueError(f"不支持的数据源: {source}。支持: local, huggingface, modelscope")
    
    return train_ds, val_ds
