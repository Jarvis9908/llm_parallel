"""
Transformer Encoder 实现。

包含 EncoderLayer（单层编码器）和 Encoder（完整编码器）。
使用 Post-Norm 架构：Self-Attention → Residual → LayerNorm → FFN → Residual → LayerNorm。

EncoderLayer 直接处理隐藏状态 (B, S, dim)，而 Encoder 在此基础上
增加了 TokenEmbedding 和正弦位置编码 (Sinusoidal PE)，可以接收
token ids 或已嵌入的隐藏状态。
"""
import torch
import torch.nn as nn

from models.common.attention import MultiHeadAttention
from models.common.feedforward import FFN
from models.common.normalization import LayerNorm
from models.common.embeddings import TokenEmbedding
from models.common.positional_encoding import sinusoidal_pe


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 单层。Post-Norm 架构（与原始论文一致）。

    计算流程:
        1. Self-Attention → Dropout → Residual Add → LayerNorm
        2. FFN → Dropout → Residual Add → LayerNorm

    Post-Norm 将 LayerNorm 放在残差连接之后，是原始 Transformer 论文的设计。
    """

    def __init__(self, config):
        """
        Args:
            config: TransformerConfig 实例，包含 dim, n_heads, ff_hidden_dim, dropout, eps
        """
        super().__init__()
        dim = config.dim
        self.self_attn = MultiHeadAttention(dim, config.n_heads, config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = LayerNorm(dim, config.eps)

        self.ffn = FFN(dim, config.ff_hidden_dim, config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = LayerNorm(dim, config.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入隐藏状态 (B, S, dim)

        Returns:
            输出隐藏状态 (B, S, dim)
        """
        # 1. Self-Attention → Dropout → Residual Add → LayerNorm
        attn_out = self.self_attn(x)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        # 2. FFN → Dropout → Residual Add → LayerNorm
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.norm2(x + ffn_out)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder 完整实现。

    计算流程:
        TokenEmbedding (可跳过) + Sinusoidal PE → [EncoderLayer × n_layers]

    接口兼容两种输入：
    - LongTensor (B, S): token ids，先做 embedding 再处理
    - FloatTensor (B, S, dim): 已嵌入的隐藏状态，直接加 PE 后处理
    """

    def __init__(self, config):
        """
        Args:
            config: TransformerConfig 实例
        """
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: token ids (B, S, dtype=long) 或已嵌入隐藏状态 (B, S, dim, dtype=float)

        Returns:
            输出隐藏状态 (B, S, dim)
        """
        # 如果是整数索引，先做 token embedding
        if x.dtype in (torch.long, torch.int):
            x = self.token_embedding(x)

        # 添加正弦位置编码
        seq_len = x.shape[1]
        pe = sinusoidal_pe(seq_len, self.config.dim).to(x.device)
        x = x + pe
        x = self.dropout(x)

        # 通过所有 Encoder 层
        for layer in self.layers:
            x = layer(x)

        return x
