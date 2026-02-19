"""Tokenization tool for datasets"""

from typing import Dict, List


def tokenize_function(examples: Dict[str, List], tokenizer, max_length: int = 512):  
    """Tokenize text samples"""
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
    
    encodings = tokenizer(  
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
        labels = input_ids[1:] + [tokenizer.pad_token_id]
        labels_list.append(labels)
    
    return {  
        "input_ids": input_ids_list,
        "labels": labels_list,
    }
