import torch
import sys
sys.path.insert(0, '.')
from models.common.embeddings import TokenEmbedding


class TestTokenEmbedding:
    def test_shape(self):
        emb = TokenEmbedding(vocab_size=1000, dim=64)
        tokens = torch.randint(0, 1000, (2, 16))
        out = emb(tokens)
        assert out.shape == (2, 16, 64)

    def test_same_token_same_embedding(self):
        emb = TokenEmbedding(vocab_size=1000, dim=64)
        tokens = torch.tensor([[5, 5], [3, 3]])
        out = emb(tokens)
        assert torch.equal(out[0, 0], out[0, 1])
        assert torch.equal(out[1, 0], out[1, 1])

    def test_backward(self):
        emb = TokenEmbedding(vocab_size=1000, dim=64)
        tokens = torch.randint(0, 1000, (2, 16))
        out = emb(tokens)
        out.sum().backward()
        assert emb.embedding.weight.grad is not None
