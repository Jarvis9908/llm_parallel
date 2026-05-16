import torch
import sys
sys.path.insert(0, '.')
from models.llama3.config import LLaMA3Config
from models.llama3.model import TransformerBlock, LLaMA3Model, LLaMA3ForCausalLM


class TestTransformerBlock:
    def test_shape(self):
        config = LLaMA3Config(dim=64, n_heads=8, n_kv_heads=2)
        block = TransformerBlock(config)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_backward(self):
        config = LLaMA3Config(dim=64, n_heads=8, n_kv_heads=2)
        block = TransformerBlock(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestLLaMA3Model:
    def test_shape(self):
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=4)
        model = LLaMA3Model(config)
        tokens = torch.randint(0, config.vocab_size, (2, 32))
        out = model(tokens)
        assert out.shape == (2, 32, config.dim)

    def test_backward(self):
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=2)
        model = LLaMA3Model(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        out = model(tokens)
        out.sum().backward()

    def test_kv_cache_shape(self):
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=64)
        model = LLaMA3Model(config)
        cache = model.create_kv_cache(batch_size=2)
        assert len(cache) == config.n_layers
        for k, v in cache:
            assert k.shape == (2, config.n_kv_heads, 64, config.head_dim)
            assert v.shape == (2, config.n_kv_heads, 64, config.head_dim)


class TestLLaMA3ForCausalLM:
    def test_shape(self):
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=4)
        model = LLaMA3ForCausalLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(tokens)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_generate(self):
        config = LLaMA3Config(
            dim=128, n_heads=4, n_kv_heads=2, n_layers=2,
            vocab_size=100, max_seq_len=32
        )
        model = LLaMA3ForCausalLM(config)
        model.eval()
        prompt = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=8, temperature=1.0)
        assert generated.shape[1] == 4 + 8

    def test_backward(self):
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=2)
        model = LLaMA3ForCausalLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(tokens)
        logits.sum().backward()
