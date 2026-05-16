"""
LLaMA3 模型实现：基于 Pre-Norm 架构的解码器专用 Transformer。

架构: TokenEmbedding → [TransformerBlock × N] → RMSNorm → LM Head

TransformerBlock 使用 Pre-Norm 残差结构:
    x = x + GQA(RMSNorm(x))   ← 自注意力（含 RoPE、KV Cache）
    x = x + SwiGLU(RMSNorm(x)) ← 前馈网络

特性:
- RoPE 旋转位置编码（在 Q/K 投影后、注意力计算前施加）
- GQA (Grouped Query Attention) 减少 KV 缓存大小
- RMSNorm 预归一化（比 LayerNorm 更快）
- SwiGLU 激活前馈网络
- KV Cache 支持自回归生成
"""
import torch
import torch.nn as nn
from models.llama3.config import LLaMA3Config
from models.common.attention import GroupedQueryAttention
from models.common.normalization import RMSNorm
from models.common.feedforward import SwiGLUFFN
from models.common.positional_encoding import RotaryPositionalEncoding
from models.common.embeddings import TokenEmbedding


class TransformerBlock(nn.Module):
    """LLaMA3 的单个 Transformer Block（Pre-Norm 风格）。

    包含一个 GQA 自注意力子层和一个 SwiGLU FFN 子层，
    每个子层前都有 RMSNorm 预归一化，子层后加残差连接。
    支持 RoPE 位置编码和 KV Cache 增量推理。
    """

    def __init__(self, config: LLaMA3Config):
        """
        Args:
            config: LLaMA3 超参数配置
        """
        super().__init__()
        self.config = config

        # Pre-Norm: 两个 RMSNorm
        self.norm1 = RMSNorm(config.dim, eps=config.eps)  # 注意力前归一化
        self.norm2 = RMSNorm(config.dim, eps=config.eps)  # FFN 前归一化

        # GQA 自注意力
        self.attn = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
        )

        # RoPE 位置编码（在 Q/K 投影后施加）
        self.rope = RotaryPositionalEncoding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # SwiGLU 前馈网络
        self.ffn = SwiGLUFFN(
            dim=config.dim,
            hidden_dim=config.ff_hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
        kv_cache: tuple | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 (B, S, dim)
            use_causal_mask: 是否使用因果掩码（自回归场景）
            kv_cache: 可选的 KV 缓存元组 (k_cache, v_cache)，
                      形状均为 (B, n_kv_heads, max_seq_len, head_dim)。
                      传入后会在 start_pos 处写入新的 K/V 并原地更新。
            start_pos: 当前序列的起始位置（KV Cache 增量解码场景）

        Returns:
            x: 输出张量 (B, S, dim)
        """
        B, S, _ = x.shape

        # --- Self-Attention with RoPE ---
        x_norm = self.norm1(x)

        # Q/K/V 投影
        q = self.attn.w_q(x_norm)   # (B, S, dim)
        k = self.attn.w_k(x_norm)   # (B, S, n_kv_heads * head_dim)
        v = self.attn.w_v(x_norm)   # (B, S, n_kv_heads * head_dim)

        # 拆分为多头
        q = self.attn._split_heads_q(q)     # (B, n_heads, S, head_dim)
        k = self.attn._split_heads_kv(k)    # (B, n_kv_heads, S, head_dim)
        v = self.attn._split_heads_kv(v)    # (B, n_kv_heads, S, head_dim)

        # 施加 RoPE 位置编码
        q, k = self.rope(q, k, start_pos=start_pos)

        # KV Cache 处理：写入新 K/V 并取全部历史
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache[:, :, start_pos:start_pos + S] = k
            v_cache[:, :, start_pos:start_pos + S] = v
            total_len = start_pos + S
            k = k_cache[:, :, :total_len]
            v = v_cache[:, :, :total_len]

        # 将 KV 头复制到与 Q 头数一致
        k = self.attn._repeat_kv(k)   # (B, n_heads, total_len, head_dim)
        v = self.attn._repeat_kv(v)   # (B, n_heads, total_len, head_dim)

        # 因果掩码：仅在 S > 1 时创建（单 token 解码不需要掩码）
        mask = None
        if use_causal_mask and S > 1:
            mask = self.attn._create_causal_mask(S, x.device)

        # 缩放点积注意力
        out = self.attn._scaled_dot_product_attention(q, k, v, mask)
        out = self.attn._merge_heads(out)    # (B, S, dim)
        out = self.attn.w_o(out)

        # 残差连接
        x = x + out

        # --- SwiGLU FFN ---
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class LLaMA3Model(nn.Module):
    """LLaMA3 基础模型（不含 LM Head）。

    将 token id 序列映射为隐状态序列，包含 Token Embedding、
    N 层 TransformerBlock 和最终 RMSNorm。
    """

    def __init__(self, config: LLaMA3Config):
        """
        Args:
            config: LLaMA3 超参数配置
        """
        super().__init__()
        self.config = config

        self.embed = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.eps)

    def create_kv_cache(self, batch_size: int) -> list:
        """创建 KV Cache。

        返回一个列表，每层一个 (k, v) 元组，
        k 和 v 的形状均为 (batch_size, n_kv_heads, max_seq_len, head_dim)，
        初始化为全零。

        Args:
            batch_size: 批次大小

        Returns:
            list of (k_cache, v_cache) 元组，长度等于 n_layers
        """
        device = next(self.parameters()).device
        cache = []
        for _ in range(self.config.n_layers):
            k = torch.zeros(
                batch_size,
                self.config.n_kv_heads,
                self.config.max_seq_len,
                self.config.head_dim,
                device=device,
            )
            v = torch.zeros(
                batch_size,
                self.config.n_kv_heads,
                self.config.max_seq_len,
                self.config.head_dim,
                device=device,
            )
            cache.append((k, v))
        return cache

    def forward(
        self,
        tokens: torch.LongTensor,
        use_causal_mask: bool = True,
        kv_cache: list | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            tokens: token id 序列 (B, S)
            use_causal_mask: 是否使用因果掩码
            kv_cache: 可选的 KV Cache 列表，会在各层中原地更新
            start_pos: 起始位置（KV Cache 增量解码场景）

        Returns:
            out: 隐状态序列 (B, S, dim)
        """
        x = self.embed(tokens)  # (B, S, dim)

        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = layer(
                x,
                use_causal_mask=use_causal_mask,
                kv_cache=layer_cache,
                start_pos=start_pos,
            )

        x = self.norm(x)
        return x


class LLaMA3ForCausalLM(nn.Module):
    """LLaMA3 因果语言模型（含 LM Head）。

    在 LLaMA3Model 之上添加线性投影层（LM Head），
    将隐状态映射到词汇表空间以输出 logits，
    并提供自回归文本生成功能。
    """

    def __init__(self, config: LLaMA3Config):
        """
        Args:
            config: LLaMA3 超参数配置
        """
        super().__init__()
        self.config = config
        self.model = LLaMA3Model(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.LongTensor,
        use_causal_mask: bool = True,
        kv_cache: list | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        前向传播，输出 logits。

        Args:
            tokens: token id 序列 (B, S)
            use_causal_mask: 是否使用因果掩码
            kv_cache: 可选的 KV Cache 列表
            start_pos: 起始位置

        Returns:
            logits: (B, S, vocab_size)
        """
        h = self.model(
            tokens,
            use_causal_mask=use_causal_mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        return self.lm_head(h)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> torch.LongTensor:
        """从 logits 中采样下一个 token。

        Args:
            logits: (B, 1, vocab_size)
            temperature: 温度参数，0 表示贪心解码

        Returns:
            next_token: (B, 1)
        """
        if temperature == 0:
            return logits.argmax(dim=-1)  # (B, 1)

        logits = logits / temperature
        # logits: (B, 1, V) -> (B, V)
        probs = torch.softmax(logits.squeeze(1), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """自回归文本生成（使用 KV Cache）。

        1. Prefill: 一次性处理整个 prompt，初始化 KV Cache
        2. Decode: 逐 token 采样，复用 KV Cache 避免重复计算

        Args:
            prompt: 提示 token id (B, prompt_len)
            max_new_tokens: 最多生成的新 token 数
            temperature: 采样温度，0 表示贪心解码

        Returns:
            generated: (B, prompt_len + max_new_tokens)，包含 prompt + 生成序列
        """
        B, prompt_len = prompt.shape

        # 创建 KV Cache
        kv_cache = self.model.create_kv_cache(B)

        # Prefill: 处理整个 prompt
        h = self.model(prompt, kv_cache=kv_cache, start_pos=0)  # (B, prompt_len, dim)
        logits = self.lm_head(h[:, -1:, :])  # (B, 1, vocab_size)
        next_token = self._sample(logits, temperature)  # (B, 1)

        generated_tokens = [next_token]

        # Decode: 逐 token 生成
        for step in range(1, max_new_tokens):
            cur_pos = prompt_len + step - 1
            h = self.model(
                next_token,
                kv_cache=kv_cache,
                start_pos=cur_pos,
            )
            logits = self.lm_head(h[:, -1:, :])
            next_token = self._sample(logits, temperature)
            generated_tokens.append(next_token)

        return torch.cat([prompt] + generated_tokens, dim=1)  # (B, prompt_len + max_new_tokens)


if __name__ == "__main__":
    # 快速冒烟测试
    config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=64)

    # TransformerBlock
    block = TransformerBlock(config)
    x = torch.randn(2, 16, 128)
    out = block(x)
    print(f"TransformerBlock: {x.shape} -> {out.shape}")

    # LLaMA3Model
    model = LLaMA3Model(config)
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    h = model(tokens)
    print(f"LLaMA3Model: {tokens.shape} -> {h.shape}")

    # KV Cache
    cache = model.create_kv_cache(batch_size=2)
    print(f"KV Cache: {len(cache)} layers, k={cache[0][0].shape}, v={cache[0][1].shape}")

    # LLaMA3ForCausalLM
    lm = LLaMA3ForCausalLM(config)
    logits = lm(tokens)
    print(f"LLaMA3ForCausalLM: {tokens.shape} -> {logits.shape}")

    # Generate
    lm.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    generated = lm.generate(prompt, max_new_tokens=8, temperature=1.0)
    print(f"Generate: prompt {prompt.shape} -> generated {generated.shape}")

    print("All LLaMA3 checks passed.")
    print(f"Total params: {sum(p.numel() for p in lm.parameters()):,}")
