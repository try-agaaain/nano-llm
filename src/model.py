"""Transformer模型实现"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x + pe)


class TransformerBlock(nn.Module):
    """Transformer编码块"""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_mask = ~mask[0, 0] if mask is not None else None
        
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_output))
        
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class NanoLLM(nn.Module):
    """轻量级Transformer模型"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(input_ids) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        
        if mask is None:
            mask = self._create_causal_mask(input_ids.size(1), input_ids.device)
        
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        return self.lm_head(self.norm(x))
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        generated = prompt.clone().to(device)
        max_ctx = self.pos_encoding.pe.size(1)
        
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) > max_ctx:
                    generated = generated[:, -max_ctx:]
                
                logits = self.forward(generated)
                next_logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                next_probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(next_probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
