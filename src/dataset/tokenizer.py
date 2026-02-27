import os
from pathlib import Path
from typing import Optional, Dict, List
import tempfile

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from src.dataset import create_dataset


class TokenizerFast(PreTrainedTokenizerFast):
    """快速分词器，统一管理BPE模型训练和文本编码"""
    
    tokenizer_file = "tokenizer.json"
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
        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
    
    @classmethod
    def from_pretrained(cls, model_id_or_path: str, **kwargs) -> "TokenizerFast":
        tokenizer = super().from_pretrained(model_id_or_path, **kwargs)
        print(f"[OK] Loaded tokenizer (vocab: {tokenizer.vocab_size})")
        return tokenizer

    
    @staticmethod
    def train_from_dataset(
        save_path: str,
        dataset,
        vocab_size: int = 8192,
        num_samples: int = 50000,
    ) -> "TokenizerFast":
        """从数据集训练BPE分词器"""
        print(f"[INFO] Training tokenizer (vocab: {vocab_size}, samples: {num_samples})")
        
        # 初始化BPE Tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            min_frequency=2,
        )
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            train_file_path = f.name
            
            # Extract texts from HuggingFace dataset
            count = 0
            for item in dataset:
                # Handle both dict items (from CSV/HF datasets) and string items
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = item
                
                # Skip empty or invalid text entries
                if text and isinstance(text, str):
                    f.write(text + "\n\n")
                    count += 1
                    if count >= num_samples:
                        break
        
        try:
            tokenizer.train(files=[train_file_path], trainer=trainer)
            print(f"[OK] Training completed (vocab: {tokenizer.get_vocab_size()})")
        finally:
            if os.path.exists(train_file_path):
                os.unlink(train_file_path)
        
        tokenizer.decoder = decoders.BPEDecoder()
        tokenizer.save(str(Path(save_path) / "tokenizer.json"), pretty=True)
        
        fast_tokenizer = TokenizerFast(tokenizer_object=tokenizer)
        fast_tokenizer.save_pretrained(save_path)
        print(f"[OK] Saved to: {save_path}")
        
        return fast_tokenizer


    @staticmethod
    def load_or_train(
        tokenizer_path: Optional[str] = None,
        dataset=None,
        vocab_size: int = 8192,
        num_samples: int = 50000,
        force_retrain: bool = False,
    ) -> "TokenizerFast":
        """加载或训练分词器（需传入已加载的dataset）"""
        if tokenizer_path is None:
            tokenizer_path = Path(__file__).parent.parent / "output" / "tokenizer"
        else:
            tokenizer_path = Path(tokenizer_path)
        tokenizer_json = tokenizer_path / "tokenizer.json"
        
        if not force_retrain and tokenizer_json.exists():
            print(f"[INFO] Loading tokenizer from {tokenizer_path}")
            return TokenizerFast.from_pretrained(str(tokenizer_path))
        
        if dataset is None:
            raise ValueError("Dataset required for training tokenizer")
        
        print(f"[INFO] Training new tokenizer...")
        return TokenizerFast.train_from_dataset(
            save_path=str(tokenizer_path),
            dataset=dataset,
            vocab_size=vocab_size,
            num_samples=num_samples,
        )


    
    @staticmethod
    def load_or_train_from_dir(
        tokenizer_path: Optional[str] = None,
        dataset_config: dict = None,
        vocab_size: int = 8192,
        num_samples: int = 50000,
        force_retrain: bool = False,
    ) -> "TokenizerFast":
        """加载或训练分词器
        
        Args:
            tokenizer_path: 分词器保存路径
            dataset_config: 单个数据集的配置字典
            vocab_size: 词汇表大小
            num_samples: 训练样本数
            force_retrain: 是否强制重新训练
            
        Returns:
            TokenizerFast 分词器
        """
        if tokenizer_path is None:
            tokenizer_path = Path(__file__).parent.parent / "output" / "tokenizer"
        else:
            tokenizer_path = Path(tokenizer_path)
        tokenizer_json = tokenizer_path / "tokenizer.json"
        
        if not force_retrain and tokenizer_json.exists():
            print(f"📖 Loading tokenizer from {tokenizer_path}")
            return TokenizerFast.from_pretrained(str(tokenizer_path))

        if dataset_config is None:
            raise ValueError("Dataset config required for training tokenizer")
        
        print(f"📚 Loading dataset from config...")
        dataset, _ = create_dataset(dataset_config, split="train")
        
        return TokenizerFast.load_or_train(
            tokenizer_path=tokenizer_path,
            dataset=dataset,
            vocab_size=vocab_size,
            num_samples=num_samples,
            force_retrain=force_retrain,
        )
    
    def tokenize_batch(self, examples: Dict[str, List], max_length: int = 512) -> Dict:
        """批量tokenize文本样本"""
        texts = examples["text"]  
        if hasattr(texts, 'to_pylist'):  
            texts = texts.to_pylist()  
        else:  
            texts = list(texts)  
        
        processed_texts = []
        for text in texts:  
            if text is None:  
                text = ""
            elif not isinstance(text, str):  
                text = str(text)
            processed_texts.append(text)
        
        encodings = self(  
            processed_texts,  
            max_length=max_length,  
            truncation=True,  
            padding="max_length",
            return_tensors=None  
        ) 
        
        # For causal language modeling: shift labels by 1 position
        input_ids_list = encodings["input_ids"]
        labels_list = []
        
        for input_ids in input_ids_list:
            labels = input_ids[1:] + [self.pad_token_id]
            labels_list.append(labels)
        
        return {  
            "input_ids": input_ids_list,
            "labels": labels_list,
        }
