"""
TinyStories 自定义分词器 - 基于 BPE 算法，针对 TinyStories 数据集优化
"""

import os
from pathlib import Path
from typing import Optional
import tempfile

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from src.dataset import TinyStoriesDataset


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


def train_tokenizer_from_dataset(
    save_path: str,
    dataset,
    vocab_size: int = 8192,
    num_samples: int = 50000,
) -> TinyStoriesTokenizerFast:
    """
    从数据集训练 BPE 分词器
    
    Args:
        save_path: 保存路径
        dataset: BaseDataset 实例
        vocab_size: 词表大小（默认 8192）
        num_samples: 用于训练分词器的样本数量（默认 50000）
    
    Returns:
        TinyStoriesTokenizerFast: 训练后的 tokenizer
    """
    print(f"📚 从数据集训练分词器...")
    print(f"   词表大小: {vocab_size}")
    print(f"   训练样本数: {num_samples}")
    
    # 1. 初始化 BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # 特殊标记（针对语言模型）
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    
    # 2. 配置 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )
    
    # 3. 从数据集获取文本并创建临时训练文件
    print("   正在准备训练数据...")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
        train_file_path = f.name
        texts = dataset.get_texts(num_samples=num_samples)
        for text in texts:
            f.write(text + "\n\n")
    
    try:
        # 4. 训练分词器
        print("   正在训练 BPE 分词器...")
        tokenizer.train(
            files=[train_file_path],
            trainer=trainer
        )
        print(f"   ✅ 训练完成 (词表大小: {tokenizer.get_vocab_size()})")
    finally:
        # 清理临时文件
        if os.path.exists(train_file_path):
            os.unlink(train_file_path)
    
    # 5. 设置解码器
    tokenizer.decoder = decoders.BPEDecoder()
    
    # 6. 保存底层文件
    tokenizer.save(str(Path(save_path) / "tokenizer.json"), pretty=True)
    print(f"   💾 已保存到: {save_path}/tokenizer.json")
    
    # 7. 创建 TinyStoriesTokenizerFast 实例并保存
    fast_tokenizer = TinyStoriesTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.save_pretrained(save_path)
    print(f"   ✅ 分词器已保存到: {save_path}")
    
    return fast_tokenizer


def load_or_train_tokenizer(
    tokenizer_path: Optional[str] = None,
    dataset=None,
    vocab_size: int = 8192,
    num_samples: int = 50000,
    force_retrain: bool = False,
) -> TinyStoriesTokenizerFast:
    """
    加载已存在的分词器，如果不存在则训练新的
    
    Args:
        tokenizer_path: 分词器保存路径（如果为 None，使用默认路径）
        dataset: BaseDataset 实例（训练时必须提供）
        vocab_size: 词表大小（仅在训练时使用）
        num_samples: 训练样本数（仅在训练时使用）
        force_retrain: 是否强制重新训练
    
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
        if dataset is None:
            raise ValueError("训练分词器时必须提供 dataset 参数")
        print(f"🔨 训练新的分词器...")
        return train_tokenizer_from_dataset(
            save_path=str(tokenizer_path),
            dataset=dataset,
            vocab_size=vocab_size,
            num_samples=num_samples,
        )

if __name__ == "__main__":
    # 示例：需要提供数据集目录
    from src.dataset import TinyStoriesDataset
    
    data_dir = "path/to/tinystories/dataset"  # 请替换为实际路径
    
    try:
        dataset = TinyStoriesDataset(data_dir)
        
        tokenizer = load_or_train_tokenizer(
            tokenizer_path="./tokenizer",
            dataset=dataset,
            vocab_size=8192,
            num_samples=10000,
            force_retrain=True,
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
    except ValueError as e:
        print(f"❌ 错误: {e}")
        print("   请提供有效的数据集目录路径")