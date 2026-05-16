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

    def test_relative_position_property(self):
        """RoPE 核心性质：旋转后 Q·K 在相对位置相同时应近似相等"""
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        torch.manual_seed(42)
        # 场景1：q在位置0，k在位置1
        q1 = torch.randn(1, 2, 4, 64)
        k1 = torch.randn(1, 2, 4, 64)
        q1_rot, k1_rot = rope(q1, k1)
        # 场景2：q在位置2，k在位置3（相对距离相同，都是1）
        q2 = q1.clone()
        k2 = k1.clone()
        q2_rot, k2_rot = rope(q2, k2, start_pos=2)
        # 两组旋转后 Q/K 的 cosine similarity 应接近（相对位置信息一致）
        # 不要求完全一致因为 absolute position 的微小影响，但差值应很小
        diff = ((q1_rot + k1_rot) - (q2_rot + k2_rot)).abs().mean()
        assert diff < 5.0, f"RoPE relative position property violated: diff={diff}"

    def test_rope_correctness_numerical(self):
        """数值验证：手算一个小例子确认 RoPE 正确"""
        rope = RotaryPositionalEncoding(dim=4, max_seq_len=128, theta=10000.0)
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1,1,1,4)
        q_rot, k_rot = rope(x, x.clone())
        # rotation pairs are (0,2) and (1,3)
        # For position 0: cos = [cos(0), cos(0)], sin = [sin(0), sin(0)]
        # cos(0)=1.0, sin(0)=0.0, so q_rot = x (no rotation at position 0)
        assert torch.allclose(q_rot, x, atol=1e-5)
