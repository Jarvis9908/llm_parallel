"""
Token Embedding 层手写实现。

本质上就是一个查找表（lookup table）：给定 token id，返回对应的向量。
使用 PyTorch 的 nn.Embedding 作为底层实现。
"""
import torch


class TokenEmbedding(torch.nn.Module):
    """Token Embedding 层。将整数 token id 映射为 dim 维的稠密向量。"""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.dim = dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(token_ids)


if __name__ == "__main__":
    emb = TokenEmbedding(vocab_size=1000, dim=64)
    tokens = torch.randint(0, 1000, (2, 8))
    out = emb(tokens)
    print(f"Token Embedding: {tokens.shape} -> {out.shape}")
