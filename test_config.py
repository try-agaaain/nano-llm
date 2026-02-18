#!/usr/bin/env python
"""测试新的配置格式和数据集加载"""

import yaml
from pathlib import Path
from src.dataset import create_dataset

# 加载配置
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

dataset_config = config.get("dataset", {})
print("=" * 70)
print("配置文件加载成功")
print("=" * 70)
print(f"当前选择的数据集: {dataset_config.get('select')}")
print(f"可用的数据集: {list(dataset_config.get('configs', {}).keys())}")

# 获取选中的数据集配置
dataset_select = dataset_config.get("select", "tinystories")
selected_config = dataset_config.get("configs", {}).get(dataset_select, {})

print("\n" + "=" * 70)
print(f"数据集 '{dataset_select}' 的配置:")
print("=" * 70)
for key, value in selected_config.items():
    print(f"  {key}: {value}")

# 尝试创建数据集（仅本地源）
if selected_config.get("source") == "local":
    print("\n" + "=" * 70)
    print(f"尝试加载本地数据集 '{dataset_select}'...")
    print("=" * 70)
    try:
        train_dataset, val_dataset = create_dataset(dataset_select, selected_config)
        print(f"✓ 成功加载数据集")
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  验证集大小: {len(val_dataset)}")
        
        # 获取样本
        print(f"\n✓ 获取样本文本:")
        texts = train_dataset.get_texts(num_samples=2)
        for i, text in enumerate(texts, 1):
            print(f"  [样本 {i}] {text[:100]}...")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n⚠️  数据源为 '{selected_config.get('source')}'，需要配置相应的 API token")
    print("  本测试脚本仅支持本地数据源的验证")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
