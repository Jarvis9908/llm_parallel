"""
Transformer Decoder 实现。

包含 DecoderLayer（单层解码器）和 Decoder（完整解码器）。
使用 Post-Norm 架构，DecoderLayer 包含三个子层：
  1. Masked Self-Attention → Dropout → Residual → LayerNorm
  2. Cross-Attention → Dropout → Residual → LayerNorm
  3. FFN → Dropout → Residual → LayerNorm

Decoder 在此基础上增加了 TokenEmbedding 和正弦位置编码 (Sinusoidal PE)，
接收目标 token ids 和 Encoder 输出作为输入。
"""
import torch
import torch.nn as nn

from models.common.attention import MultiHeadAttention
from models.common.feedforward import FFN
from models.common.normalization import LayerNorm
from models.common.embeddings import TokenEmbedding
from models.common.positional_encoding import sinusoidal_pe


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 单层。Post-Norm 架构。

    计算流程:
        1. Masked Self-Attention → Dropout → Residual Add → LayerNorm
        2. Cross-Attention → Dropout → Residual Add → LayerNorm
        3. FFN → Dropout → Residual Add → LayerNorm

    其中 Self-Attention 使用因果掩码（causal mask）确保每个位置
    只能看到当前及之前的位置，保证自回归生成。
    Cross-Attention 的 Q 来自 decoder 隐藏状态，K/V 来自 encoder 输出。
    """

    def __init__(self, config):
        """
        Args:
            config: TransformerConfig 实例，包含 dim, n_heads, ff_hidden_dim, dropout, eps
        """
        super().__init__()
        dim = config.dim

        # 子层 1: Masked Self-Attention
        self.self_attn = MultiHeadAttention(dim, config.n_heads, config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = LayerNorm(dim, config.eps)

        # 子层 2: Cross-Attention（Q 来自 decoder，K/V 来自 encoder）
        self.cross_attn = MultiHeadAttention(dim, config.n_heads, config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm2 = LayerNorm(dim, config.eps)

        # 子层 3: FFN
        self.ffn = FFN(dim, config.ff_hidden_dim, config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm3 = LayerNorm(dim, config.eps)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: decoder 隐藏状态 (B, S_dec, dim)
            encoder_output: encoder 输出 (B, S_enc, dim)

        Returns:
            输出隐藏状态 (B, S_dec, dim)
        """
        # 1. Masked Self-Attention → Dropout → Residual Add → LayerNorm
        attn_out = self.self_attn(x, use_causal_mask=True)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        # 2. Cross-Attention → Dropout → Residual Add → LayerNorm
        # Q 来自 decoder x，K/V 来自 encoder_output
        q = self.cross_attn.w_q(x)
        k = self.cross_attn.w_k(encoder_output)
        v = self.cross_attn.w_v(encoder_output)

        q = self.cross_attn._split_heads(q)
        k = self.cross_attn._split_heads(k)
        v = self.cross_attn._split_heads(v)

        cross_out = self.cross_attn._scaled_dot_product_attention(q, k, v)
        cross_out = self.cross_attn._merge_heads(cross_out)
        cross_out = self.cross_attn.w_o(cross_out)

        cross_out = self.dropout2(cross_out)
        x = self.norm2(x + cross_out)

        # 3. FFN → Dropout → Residual Add → LayerNorm
        ffn_out = self.ffn(x)
        ffn_out = self.dropout3(ffn_out)
        x = self.norm3(x + ffn_out)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder 完整实现。

    计算流程:
        TokenEmbedding + Sinusoidal PE → [DecoderLayer × n_layers]

    输入:
        token_ids: 目标序列的 token ids (B, S_dec, dtype=long)
        encoder_output: Encoder 输出 (B, S_enc, dim)
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
            [DecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            token_ids: 目标序列 token ids (B, S_dec)，dtype 为 long
            encoder_output: Encoder 输出 (B, S_enc, dim)

        Returns:
            输出隐藏状态 (B, S_dec, dim)
        """
        # Token Embedding
        x = self.token_embedding(token_ids)

        # 添加正弦位置编码
        seq_len = x.shape[1]
        pe = sinusoidal_pe(seq_len, self.config.dim).to(x.device)
        x = x + pe
        x = self.dropout(x)

        # 通过所有 Decoder 层
        for layer in self.layers:
            x = layer(x, encoder_output)

        return x
