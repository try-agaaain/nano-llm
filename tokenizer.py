"""
TinyStories 自定义分词器 - 基于 BPE 算法，针对 TinyStories 数据集优化
支持从 CSV 和 HuggingFace 数据集加载
"""

import os
import tempfile
import csv
from pathlib import Path
from typing import List, Optional, Iterator, Union

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset
from tqdm import tqdm


class TinyStoriesTokenizerFast(PreTrainedTokenizerFast):
    """
    TinyStories 分词器 - 继承 PreTrainedTokenizerFast
    针对英文童话故事文本优化的 BPE 分词器
    """
    
    # 告诉父类，底层 tokenizers 库的文件叫什么名字
    tokenizer_file = "tokenizer.json"
    
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        **kwargs
    ):
        """
        初始化方法配置特殊标记并调用父类的 __init__。
        
        Args:
            tokenizer_object: 底层的 tokenizers.Tokenizer 对象
            unk_token: 未知标记，默认 "<unk>"
            pad_token: 填充标记，默认 "<pad>"
            bos_token: 开始标记，默认 "<bos>"
            eos_token: 结束标记，默认 "<eos>"
            **kwargs: 其他参数传递给父类
        """
        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
    
    @classmethod
    def from_pretrained(cls, model_id_or_path: str, **kwargs) -> "TinyStoriesTokenizerFast":
        """
        从预训练模型加载 tokenizer
        
        Args:
            model_id_or_path: 模型 ID 或本地路径
            **kwargs: 其他参数传递给父类
        
        Returns:
            TinyStoriesTokenizerFast: 加载的 tokenizer 实例
        """
        tokenizer = super().from_pretrained(model_id_or_path, **kwargs)
        print(f"✅ 已加载 TinyStories 分词器 (词表大小: {tokenizer.vocab_size})")
        return tokenizer


def train_tokenizer_from_tinystories(
    save_path: str,
    vocab_size: int = 8192,
    num_samples: int = 50000,
    split: str = "train",
    dataset_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    text_column: str = "text",
) -> TinyStoriesTokenizerFast:
    """
    从 TinyStories 数据集或 CSV 文件训练 BPE 分词器
    
    Args:
        save_path: 保存路径
        vocab_size: 词表大小（默认 8192，比 GPT-2 的 50k 小很多）
        num_samples: 用于训练分词器的样本数量（默认 50000）
        split: 数据集分割（train/validation）
        dataset_dir: 数据集缓存目录（如果为 None，使用默认的 dataset 目录）
        csv_path: CSV 文件路径（如果提供，优先于数据集）
        text_column: CSV 中文本列的名称（默认 "text"）
    
    Returns:
        TinyStoriesTokenizerFast: 训练后的 tokenizer
    """
    print(f"📚 从 TinyStories 数据集训练分词器...")
    print(f"   词表大小: {vocab_size}")
    print(f"   训练样本数: {num_samples}")
    
    # 1. 初始化 BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # 使用 Whitespace pre-tokenizer 替代 ByteLevel，避免字符级分割
    # 这样可以让 BPE 学习到真正的子词单元
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # 特殊标记（针对语言模型）
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    
    # 2. 配置 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,  # 最小出现频率
    )
    
    # 3. 收集训练文本
    print("   正在收集训练文本...")
    
    # 创建临时文件来存储训练文本
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    temp_file_path = temp_file.name
    
    try:
        collected_samples = 0
        
        # 从 CSV 或数据集加载数据
        if csv_path and os.path.exists(csv_path):
            print(f"   从 CSV 文件加载: {csv_path}")
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in tqdm(csv_reader, desc="收集文本", total=num_samples):
                    if collected_samples >= num_samples:
                        break
                    
                    # 获取文本字段
                    if text_column not in row:
                        # 尝试寻找其他可能的列名
                        for col in ["text", "story", "content", "narrative"]:
                            if col in row:
                                text = row[col]
                                break
                        else:
                            # 使用第一个非空字段
                            text = next((v for v in row.values() if v), None)
                    else:
                        text = row[text_column]
                    
                    if not text or not isinstance(text, str) or not text.strip():
                        continue
                    
                    # 清理文本
                    cleaned_text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
                    temp_file.write(cleaned_text + "\n\n")
                    collected_samples += 1
        else:
            # 从 HuggingFace 数据集加载
            print(f"   从 HuggingFace 数据集加载 ({split} 分割)...")
            
            # 设置数据集缓存目录
            if dataset_dir is None:
                # 默认使用当前目录下的 dataset 文件夹
                dataset_dir = str(Path.cwd() / "dataset")
            
            # 确保目录存在
            Path(dataset_dir).mkdir(parents=True, exist_ok=True)
            
            # 使用非流式加载，确保数据真实下载到 cache_dir
            try:
                dataset = load_dataset(
                    "./dataset",
                    split=split,
                    streaming=False,
                    cache_dir=dataset_dir,
                    download_mode="force_redownload",
                )
                
                for example in tqdm(dataset, desc="收集文本", total=num_samples):
                    if collected_samples >= num_samples:
                        break
                    
                    # 尝试多种可能的字段名
                    story = None
                    for field_name in ["text", "story", "content", "narrative", "sentence"]:
                        if field_name in example:
                            story = example[field_name]
                            break
                    
                    if story is None:
                        # 如果没有找到，使用第一个字符串字段
                        for key, value in example.items():
                            if isinstance(value, str):
                                story = value
                                break
                    
                    if story is None or not isinstance(story, str) or not story.strip():
                        continue
                    
                    # 清理文本
                    cleaned_story = "\n".join(line.strip() for line in story.split("\n") if line.strip())
                    temp_file.write(cleaned_story + "\n\n")
                    collected_samples += 1
            except Exception as e:
                print(f"   ⚠️  从 HuggingFace 加载失败: {e}")
                print(f"   请确保提供了 CSV 文件路径或数据集目录")
                raise
        
        temp_file.close()
        print(f"   ✅ 已收集 {collected_samples} 个样本")
        
        # 训练分词器
        print("   正在训练 BPE 分词器...")
        tokenizer.train(
            files=[temp_file_path],
            trainer=trainer
        )
        
        print(f"   ✅ 训练完成 (词表大小: {tokenizer.get_vocab_size()})")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    # 5. 设置解码器 - 使用 BPEDecoder
    # 由于使用 Whitespace pre-tokenizer，BPEDecoder 就足够了
    tokenizer.decoder = decoders.BPEDecoder()
    
    # 6. 保存底层文件 (tokenizer.json)
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(Path(save_path) / "tokenizer.json"), pretty=True)
        print(f"   💾 已保存到: {save_path}/tokenizer.json")
    
    # 7. 创建 TinyStoriesTokenizerFast 实例
    fast_tokenizer = TinyStoriesTokenizerFast(tokenizer_object=tokenizer)
    
    # 8. 保存配置文件（生成 tokenizer_config.json）
    fast_tokenizer.save_pretrained(save_path)
    print(f"   ✅ 分词器已保存到: {save_path}")
    
    return fast_tokenizer


def load_or_train_tokenizer(
    tokenizer_path: Optional[str] = None,
    vocab_size: int = 8192,
    num_samples: int = 50000,
    force_retrain: bool = False,
    dataset_dir: Optional[str] = None,
    csv_path: Optional[str] = None,
    text_column: str = "text",
) -> TinyStoriesTokenizerFast:
    """
    加载已存在的分词器，如果不存在则训练新的
    
    Args:
        tokenizer_path: 分词器保存路径（如果为 None，使用默认路径）
        vocab_size: 词表大小（仅在训练时使用）
        num_samples: 训练样本数（仅在训练时使用）
        force_retrain: 是否强制重新训练
        dataset_dir: 数据集缓存目录（如果为 None，使用默认的 dataset 目录）
        csv_path: CSV 文件路径（如果提供，优先于数据集）
        text_column: CSV 中文本列的名称（默认 "text"）
    
    Returns:
        TinyStoriesTokenizerFast: 分词器实例
    """
    if tokenizer_path is None:
        tokenizer_path = "./tokenizer"
    
    tokenizer_path = Path(tokenizer_path)
    tokenizer_json = tokenizer_path / "tokenizer.json"
    
    # 检查是否存在已训练的分词器
    if not force_retrain and tokenizer_json.exists():
        print(f"📖 加载已存在的分词器: {tokenizer_path}")
        return TinyStoriesTokenizerFast.from_pretrained(str(tokenizer_path))
    else:
        print(f"🔨 训练新的分词器...")
        return train_tokenizer_from_tinystories(
            save_path=str(tokenizer_path),
            vocab_size=vocab_size,
            num_samples=num_samples,
            dataset_dir=dataset_dir,
            csv_path=csv_path,
            text_column=text_column,
        )

if __name__ == "__main__":
    # 设置数据集目录（默认使用当前目录下的 dataset 文件夹）
    dataset_dir = str(Path.cwd() / "dataset")
    
    tokenizer = load_or_train_tokenizer(
        tokenizer_path="./tokenizer",
        vocab_size=8192,
        num_samples=10000,  # 先用较小样本演示修复效果
        force_retrain=True,  # 强制重新训练以应用修复
        dataset_dir=dataset_dir,
    )
    print(f"✅ 分词器词表大小: {tokenizer.vocab_size}")
    
    # 测试编解码是否正常
    print("\n" + "="*70)
    print("编解码测试")
    print("="*70)
    
    test_texts = [
        "Hello world",
        "The little girl",
        "In the forest",
        "Once upon a time there was a beautiful day",
    ]
    
    for text in test_texts:
        print(f"\n原始文本: {text}")
        
        # 编码
        encoded = tokenizer.encode(text, return_tensors="pt")
        token_ids = encoded[0].tolist()
        print(f"Token IDs: {token_ids}")
        
        # 解码
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"解码文本: {decoded}")
        
        # 检查是否有Ġ符号
        has_symbols = "Ġ" in decoded or "Ċ" in decoded
        status = "❌ 有乱码符号" if has_symbols else "✅ 正常"
        print(f"状态: {status}")
    
    print("\n" + "="*70)