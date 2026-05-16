import torch
import sys
sys.path.insert(0, '.')
from models.common.positional_encoding import sinusoidal_pe, RotaryPositionalEncoding


class TestSinusoidalPE:
    def test_shape(self):
        pe = sinusoidal_pe(seq_len=100, dim=64)
        assert pe.shape == (1, 100, 64)

    def test_unique_positions(self):
        """不同位置的编码应该不同"""
        pe = sinusoidal_pe(seq_len=50, dim=64)
        assert not torch.allclose(pe[0, 0], pe[0, 1])


class TestRoPE:
    def test_shape(self):
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        q = torch.randn(2, 8, 16, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(2, 8, 16, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_modifies_qk(self):
        """RoPE 确实修改了 Q/K 向量"""
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        q = torch.ones(1, 1, 4, 64) * 0.5
        k = torch.ones(1, 1, 4, 64) * 0.5
        q_rot, k_rot = rope(q, k)
        assert not torch.allclose(q_rot, q)

    def test_backward(self):
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        q = torch.randn(2, 8, 4, 64, requires_grad=True)
        k = torch.randn(2, 8, 4, 64, requires_grad=True)
        q_rot, k_rot = rope(q, k)
        (q_rot.sum() + k_rot.sum()).backward()
        assert q.grad is not None
        assert k.grad is not None
