"""重新训练分词器脚本"""
from pathlib import Path
import shutil
from tokenizer import load_or_train_tokenizer
import os
os.environ["HF_TOKEN"] = "HF_TOKEN_REMOVED"
# 删除旧的tokenizer
tokenizer_path = Path("./tokenizer")
if tokenizer_path.exists():
    print(f"删除旧的tokenizer: {tokenizer_path}")
    shutil.rmtree(tokenizer_path)

# 重新训练
print("\n开始重新训练分词器...")
tokenizer = load_or_train_tokenizer(
    tokenizer_path="./tokenizer",
    vocab_size=8192,
    num_samples=50000,
    force_retrain=True,
    dataset_dir="./dataset"
)

print(f"\n✅ 分词器训练完成，词表大小: {tokenizer.vocab_size}")
