import torch
import sys
sys.path.insert(0, '.')
from models.common.normalization import LayerNorm, RMSNorm


class TestLayerNorm:
    def test_shape(self):
        ln = LayerNorm(dim=64)
        x = torch.randn(2, 16, 64)
        out = ln(x)
        assert out.shape == x.shape

    def test_mean_zero_var_one(self):
        ln = LayerNorm(dim=64)
        x = torch.randn(4, 8, 64)
        out = ln(x)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_vs_pytorch(self):
        ln = LayerNorm(dim=64)
        ln_pt = torch.nn.LayerNorm(64)
        ln.weight.data = ln_pt.weight.data.clone()
        ln.bias.data = ln_pt.bias.data.clone()
        x = torch.randn(2, 16, 64)
        assert torch.allclose(ln(x), ln_pt(x), atol=1e-5)

    def test_backward(self):
        ln = LayerNorm(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ln(x)
        out.sum().backward()
        assert x.grad is not None


class TestRMSNorm:
    def test_shape(self):
        rms = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        out = rms(x)
        assert out.shape == x.shape

    def test_rms_property(self):
        """RMSNorm 只做缩放不做中心化，RMS 值应接近 1"""
        rms = RMSNorm(dim=64)
        x = torch.randn(4, 8, 64)
        out = rms(x)
        rms_val = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert rms_val.mean() > 0.5 and rms_val.mean() < 2.0

    def test_backward(self):
        rms = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = rms(x)
        out.sum().backward()
        assert x.grad is not None
