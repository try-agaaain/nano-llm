"""LLM模型的实现"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """位置编码，为序列中的每个位置添加位置信息"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，shape (batch_size, seq_len, d_model)
        
        Returns:
            加入位置编码后的张量
        """
        # 根据当前序列长度裁剪位置编码，并确保与输入在同一设备
        pe = self.pe[:, :x.size(1), :].to(x.device)
        x = x + pe
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    单个Transformer编码块：
    - 多头自注意力
    - 前向全连接网络
    - 残差连接 + LayerNorm
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # 使用 (batch, seq, dim) 形式
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 前向全连接网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量，shape (batch_size, seq_len, d_model)
            mask: 因果掩码，shape (1, 1, seq_len, seq_len) 或 None
        
        Returns:
            经过Transformer编码块后的张量
        """
        attn_mask = None
        if mask is not None:
            # 将 (1, 1, seq_len, seq_len) -> (seq_len, seq_len)
            # 这里 mask 为下三角可见区域，需取反得到需要被 mask 的上三角区域
            attn_mask = ~mask[0, 0]
        
        # 自注意力 + 残差 + LayerNorm
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前向网络 + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x


class NanoLLM(nn.Module):
    """
    轻量级LLM模型，基于Transformer架构
    
    特点：
    - 使用多头自注意力机制
    - 支持因果掩码（Causal Masking）用于自回归生成
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        初始化模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度（默认256）
            num_heads: 注意力头数（默认8）
            num_layers: Transformer层数（默认4）
            d_ff: 前向网络隐层维度（默认1024）
            max_seq_len: 最大序列长度（默认512）
            dropout: dropout概率（默认0.1）
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """
        创建因果掩码，防止注意力查看未来的位置
        
        Args:
            seq_len: 序列长度
            device: 张量所在设备
        
        Returns:
            因果掩码张量，shape (1, 1, seq_len, seq_len)
        """
        # 使用布尔下三角掩码，后续在 MultiheadAttention 中 True 表示需要被 mask（不可见）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs，shape (batch_size, seq_len)
            mask: 可选的注意力掩码
        
        Returns:
            逻辑值张量，shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 缩放嵌入
        x = x * (self.d_model ** 0.5)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 创建因果掩码（用于自回归生成）
        if mask is None:
            mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # 最终层归一化
        x = self.norm(x)
        
        # 投影到词汇表大小
        logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            prompt: 提示词，shape (batch_size, seq_len)
            max_length: 最大生成长度
            temperature: 温度参数，控制随机性
            top_k: Top-K采样的K值
        
        Returns:
            生成的token序列
        """
        # 确保提示词在模型所在设备上
        device = next(self.parameters()).device
        generated = prompt.clone().to(device)

        # 最大上下文长度由位置编码的长度决定
        max_ctx = self.pos_encoding.pe.size(1)
        
        with torch.no_grad():
            for _ in range(max_length):
                # 如果序列长度超过最大上下文，只保留最近 max_ctx 个 token
                if generated.size(1) > max_ctx:
                    generated = generated[:, -max_ctx:]

                # 前向传播
                logits = self.forward(generated)
                
                # 获取最后一个位置的logits
                next_logits = logits[:, -1, :] / temperature
                
                # Top-K采样
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # 应用softmax
                next_probs = torch.softmax(next_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(next_probs, num_samples=1)
                
                # 添加到序列
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
