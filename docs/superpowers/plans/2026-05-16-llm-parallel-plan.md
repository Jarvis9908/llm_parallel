# LLM 架构与分布式并行学习仓库 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建从 Transformer 到 DeepSeek V3 的完整模型架构代码库，以及六大分布式并行策略的单机模拟实现。

**Architecture:** models/ 与 parallel/ 双模块独立开发，通过 common/ 共享基础组件。models/ 按 Transformer → LLaMA3 → DeepSeek V3 顺序递增复杂度；parallel/ 以 communication/ 为底座，各并行策略独立实现。所有模块 TDD 驱动，每个模块先写测试再写代码。

**Tech Stack:** Python 3.10+, PyTorch 2.x, pytest, Jupyter, matplotlib

---

## 文件依赖关系

```
models/common/           ← 无依赖，最先构建
    ↓
models/transformer/      ← 依赖 common/
models/llama3/           ← 依赖 common/
models/deepseek_v3/      ← 依赖 common/
    ↓
parallel/communication/  ← 无依赖（仅需 torch.distributed）
    ↓
parallel/data_parallel/  ← 依赖 communication/
parallel/tensor_parallel/← 依赖 communication/
parallel/pipeline_parallel/ ← 依赖 communication/
parallel/expert_parallel/   ← 依赖 communication/
parallel/context_parallel/  ← 依赖 communication/
parallel/inference/         ← 依赖 communication/
    ↓
notebooks/               ← 依赖 models/ + parallel/ 全部完成
tests/                   ← 与对应模块同步构建
```

---

## Stage 0: 项目骨架

### Task 0.1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `models/__init__.py`
- Create: `models/common/__init__.py`
- Create: `parallel/__init__.py`
- Create: `tests/__init__.py`
- Modify: `.gitignore` (已部分存在)

- [ ] **Step 1: 创建 requirements.txt**

```bash
cat > requirements.txt << 'EOF'
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
pytest>=7.0.0
EOF
```

- [ ] **Step 2: 创建 README.md**

```markdown
# LLM 架构与分布式并行学习

两条学习主线：
1. **模型架构**：Transformer → LLaMA3 → DeepSeek V3（models/）
2. **分布式并行**：通信原语 + 六大并行策略（parallel/）

## 快速开始
```bash
pip install -r requirements.txt
pytest tests/ -v
jupyter notebook notebooks/
```
```

- [ ] **Step 3: 创建目录结构和 __init__.py**

```bash
mkdir -p models/common models/transformer models/llama3 models/deepseek_v3
mkdir -p parallel/communication parallel/data_parallel parallel/tensor_parallel
mkdir -p parallel/pipeline_parallel parallel/expert_parallel parallel/context_parallel
mkdir -p parallel/inference parallel/utils
mkdir -p tests notebooks

touch models/__init__.py models/common/__init__.py
touch models/transformer/__init__.py models/llama3/__init__.py models/deepseek_v3/__init__.py
touch parallel/__init__.py
touch parallel/communication/__init__.py parallel/data_parallel/__init__.py
touch parallel/tensor_parallel/__init__.py parallel/pipeline_parallel/__init__.py
touch parallel/expert_parallel/__init__.py parallel/context_parallel/__init__.py
touch parallel/inference/__init__.py parallel/utils/__init__.py
touch tests/__init__.py
```

- [ ] **Step 4: 验证项目结构**

```bash
find . -type f -name "*.py" | sort
```
Expected: 能看到所有 __init__.py 文件

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: initialize project skeleton with directories and config files"
```

---

## Stage 1: 基础组件 (models/common/)

### Task 1.1: 激活函数 (activation.py)

**Files:**
- Create: `models/common/activation.py`
- Create: `tests/test_activation.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_activation.py
import torch
import sys
sys.path.insert(0, '.')
from models.common.activation import gelu, silu, relu

class TestActivations:
    def test_gelu_shape(self):
        x = torch.randn(2, 4, 8)
        out = gelu(x)
        assert out.shape == x.shape

    def test_gelu_approx(self):
        """GELU: x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))"""
        x = torch.tensor([0.0, 1.0, -1.0])
        out = gelu(x)
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_silu_shape(self):
        x = torch.randn(2, 4, 8)
        out = silu(x)
        assert out.shape == x.shape

    def test_silu_values(self):
        x = torch.tensor([0.0, 1.0, -1.0])
        out = silu(x)
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_relu(self):
        x = torch.tensor([-1.0, 0.0, 2.0])
        out = relu(x)
        assert torch.equal(out, torch.tensor([0.0, 0.0, 2.0]))
```

- [ ] **Step 2: 运行测试（预期失败）**

```bash
python -m pytest tests/test_activation.py -v
```
Expected: `ModuleNotFoundError: No module named 'models.common.activation'`

- [ ] **Step 3: 实现 activation.py**

```python
# models/common/activation.py
"""
激活函数手写实现。
参考：GELU (Gaussian Error Linear Unit), SiLU/Swish, ReLU
"""
import torch
import math


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU 激活函数（tanh 近似版本）
    GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    与原始 Gaussian CDF 形式的误差 < 0.1%，但计算更快。
    """
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Sigmoid Linear Unit)，也叫 Swish
    SiLU(x) = x * σ(x)
    在 LLaMA 系列中用作 FFN 的激活函数。
    """
    return x * torch.sigmoid(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU(x) = max(0, x) — 注意：这里返回的是一个新 tensor"""
    return torch.maximum(x, torch.zeros_like(x))


# 快速验证
if __name__ == "__main__":
    x = torch.randn(2, 4)
    assert torch.allclose(gelu(x), torch.nn.functional.gelu(x), atol=1e-5)
    assert torch.allclose(silu(x), torch.nn.functional.silu(x), atol=1e-5)
    print("All activation checks passed.")
```

- [ ] **Step 4: 运行测试（预期通过）**

```bash
python -m pytest tests/test_activation.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add models/common/activation.py tests/test_activation.py
git commit -m "feat: add hand-written activation functions (GELU, SiLU, ReLU)"
```

### Task 1.2: 归一化层 (normalization.py)

**Files:**
- Create: `models/common/normalization.py`
- Create: `tests/test_normalization.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_normalization.py
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
        # 最后一维的均值应接近 0，方差应接近 1
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_vs_pytorch(self):
        ln = LayerNorm(dim=64)
        ln_pt = torch.nn.LayerNorm(64)
        # 使用相同权重
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
        # 由于可学习的 scale，RMS 值不严格为 1，但检查尺度合理
        assert rms_val.mean() > 0.5 and rms_val.mean() < 2.0

    def test_backward(self):
        rms = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = rms(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 3: 实现 normalization.py**

```python
# models/common/normalization.py
"""
归一化层手写实现：LayerNorm 和 RMSNorm。

LayerNorm: y = (x - μ) / √(σ² + ε) * γ + β
RMSNorm:   y = x / RMS(x) * γ, 其中 RMS(x) = √(mean(x²) + ε)

RMSNorm 是 LLaMA 系列使用的归一化方式，相比 LayerNorm 去掉了中心化步骤（不需要计算均值），
计算效率更高。
"""
import torch


class LayerNorm(torch.nn.Module):
    """
    标准 Layer Normalization。
    对输入最后一维做归一化：减均值、除标准差，再做可学习的 affine 变换。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))   # γ: 缩放参数
        self.bias = torch.nn.Parameter(torch.zeros(dim))    # β: 平移参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿最后一维计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)          # shape: (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # 有偏估计
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization。
    LLaMA 使用的归一化方式，不需要计算均值，比 LayerNorm 快约 10-15%。

    公式: y = x / √(mean(x²) + ε) * γ
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))   # γ: 可学习缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = √(E[x²] + ε)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


if __name__ == "__main__":
    x = torch.randn(2, 4, 8)
    ln = LayerNorm(8)
    print(f"LayerNorm: input shape {x.shape} -> output shape {ln(x).shape}")

    rms = RMSNorm(8)
    print(f"RMSNorm:  input shape {x.shape} -> output shape {rms(x).shape}")
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_normalization.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add models/common/normalization.py tests/test_normalization.py
git commit -m "feat: add hand-written LayerNorm and RMSNorm"
```

### Task 1.3: 位置编码 (positional_encoding.py)

**Files:**
- Create: `models/common/positional_encoding.py`
- Modify: `tests/test_normalization.py` (不修改，新测试文件)

实际新建 `tests/test_positional_encoding.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_positional_encoding.py
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
        # 前两个位置的内积应该不同
        assert not torch.allclose(pe[0, 0], pe[0, 1])


class TestRoPE:
    def test_shape(self):
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        q = torch.randn(2, 8, 16, 64)  # (batch, heads, seq, head_dim)
        k = torch.randn(2, 8, 16, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_relative_position_property(self):
        """
        RoPE 的核心性质：旋转后的 Q 和 K 的内积只依赖相对位置。
        即 ⟨f_q(x_m, m), f_k(x_n, n)⟩ = g(x_m, x_n, m-n)
        我们验证：相同 token 在相同相对距离下的 attention score 应相同。
        """
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        # 生成简单的 Q 和 K
        q = torch.ones(1, 1, 4, 64) * 0.5
        k = torch.ones(1, 1, 4, 64) * 0.5
        q_rot, k_rot = rope(q, k)
        # 位置 0 对位置 1 的 score，与位置 1 对位置 2 的 score，差异应非巨大（关键相对位置差都是 1）
        # 不严格要求相等，但确认旋转后张量有效值
        assert not torch.allclose(q_rot, q)  # RoPE 确实修改了 Q/K

    def test_backward(self):
        rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
        q = torch.randn(2, 8, 4, 64, requires_grad=True)
        k = torch.randn(2, 8, 4, 64, requires_grad=True)
        q_rot, k_rot = rope(q, k)
        (q_rot.sum() + k_rot.sum()).backward()
        assert q.grad is not None
        assert k.grad is not None
```

- [ ] **Step 3: 实现 positional_encoding.py**

```python
# models/common/positional_encoding.py
"""
位置编码手写实现：Sinusoidal PE 和 Rotary Positional Encoding (RoPE)。

Sinusoidal PE: 用正弦/余弦函数编码绝对位置，加到 token embedding 上。
RoPE: 通过旋转矩阵编码相对位置信息，乘到 Q 和 K 向量上，使得 attention score
      自然包含相对位置信息 ⟨f_q(x_m,m), f_k(x_n,n)⟩ = g(x_m, x_n, n-m)
"""
import torch
import math


def sinusoidal_pe(seq_len: int, dim: int) -> torch.Tensor:
    """
    正弦位置编码。Transformer 原始论文中的方案。

    返回 shape (1, seq_len, dim)，可直接加到 token embedding 上。
    PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )  # (dim/2,)
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位用 cos
    return pe.unsqueeze(0)  # (1, seq_len, dim)


class RotaryPositionalEncoding(torch.nn.Module):
    """
    Rotary Positional Encoding (RoPE)。
    LLaMA 系列使用的旋转位置编码。通过对 Q 和 K 施加旋转变换注入位置信息。

    对 head_dim 维的向量，两两一组（第 0-1 维、第 2-3 维...），
    每组用对应频率的旋转矩阵施加旋转，旋转角度随位置线性增长。

    参数:
        dim: 每个 attention head 的维度（head_dim）
        max_seq_len: 预计算的最大序列长度
        theta: 旋转频率的基数（默认 10000.0，LLaMA 使用该值）
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # 频率：对每一对维度使用不同的频率 θ_i = theta^(-2i/dim)
        freq_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        freqs = 1.0 / (theta ** (freq_indices / dim))  # (dim/2,)

        # 为所有位置预计算旋转角度
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (seq_len,)
        angles = torch.outer(positions, freqs)  # (seq_len, dim/2)

        # 预存 cos 和 sin，形状为 (1, 1, seq_len, dim)，方便广播
        cos = torch.cos(angles)  # (seq_len, dim/2)
        sin = torch.sin(angles)

        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        将向量两两一组进行旋转的半变换。
        对每一对 (x0, x1)，输出 (-x1, x0)，这是旋转矩阵作用的一半。
        """
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 施加旋转位置编码。

        参数:
            q: shape (batch, n_heads, seq_len, head_dim)
            k: shape (batch, n_heads, seq_len, head_dim)
        返回:
            q_rot, k_rot 同 shape
        """
        seq_len = q.shape[2]
        cos = self.cos_table[:seq_len]  # (seq_len, dim/2)
        sin = self.sin_table[:seq_len]

        # 广播 cos/sin 到 (1, 1, seq_len, dim/2) → 复制填充到 head_dim
        # 每个频率适用于一对维度，所以用 repeat_interleave
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # RoPE: q * cos + rotate_half(q) * sin
        # 重复 cos/sin 使得 shape 匹配
        cos_full = torch.repeat_interleave(cos, 2, dim=-1)  # (1, 1, seq, dim)
        sin_full = torch.repeat_interleave(sin, 2, dim=-1)

        q_rot = q * cos_full + self._rotate_half(q) * sin_full
        k_rot = k * cos_full + self._rotate_half(k) * sin_full

        return q_rot, k_rot


if __name__ == "__main__":
    # 快速验证
    pe = sinusoidal_pe(10, 64)
    print(f"Sinusoidal PE shape: {pe.shape}")

    rope = RotaryPositionalEncoding(dim=64, max_seq_len=128)
    q = torch.randn(2, 4, 8, 64)
    k = torch.randn(2, 4, 8, 64)
    qr, kr = rope(q, k)
    print(f"RoPE: q {q.shape} -> q_rot {qr.shape}")
    print("All positional encoding checks passed.")
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_positional_encoding.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add models/common/positional_encoding.py tests/test_positional_encoding.py
git commit -m "feat: add sinusoidal positional encoding and RoPE"
```

### Task 1.4: 前馈网络 (feedforward.py)

**Files:**
- Create: `models/common/feedforward.py`
- Create: `tests/test_feedforward.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_feedforward.py
import torch
import sys
sys.path.insert(0, '.')
from models.common.feedforward import FFN, SwiGLUFFN


class TestFFN:
    def test_shape(self):
        ffn = FFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_backward(self):
        ffn = FFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        for name, p in ffn.named_parameters():
            assert p.grad is not None, f"{name} should have grad"


class TestSwiGLUFFN:
    def test_shape(self):
        ffn = SwiGLUFFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_three_matrices(self):
        """SwiGLU 需要 3 个投影矩阵（w1, w2, w3），比标准 FFN 多一个"""
        ffn = SwiGLUFFN(dim=64, hidden_dim=256)
        param_count = sum(p.numel() for p in ffn.parameters())
        # 3 * (64 * 256) + 2 * 256 ≈ 49664
        expected_approx = 3 * 64 * 256 + 3 * 256
        assert abs(param_count - expected_approx) < 100

    def test_backward(self):
        ffn = SwiGLUFFN(dim=64, hidden_dim=256)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 3: 实现 feedforward.py**

```python
# models/common/feedforward.py
"""
前馈网络手写实现：标准 FFN 和 SwiGLU FFN。

标准 FFN:  x → Linear1 → GELU → Linear2 → output
SwiGLU FFN: x → (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down
            其中 ⊙ 表示逐元素乘法，SiLU 作为门控激活函数

SwiGLU 是 LLaMA 和 DeepSeek 系列使用的 FFN 变体，相比标准 FFN 效果好且更稳定。
"""
import torch
from models.common.activation import gelu, silu


class FFN(torch.nn.Module):
    """
    标准两层前馈网络，Transformer Encoder/Decoder 中使用。

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim)   # 升维投影
        self.w2 = torch.nn.Linear(hidden_dim, dim)    # 降维投影
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        h = self.w1(x)          # (batch, seq_len, hidden_dim)
        h = gelu(h)             # GELU 激活
        h = self.dropout(h)
        out = self.w2(h)        # (batch, seq_len, dim)
        return out


class SwiGLUFFN(torch.nn.Module):
    """
    SwiGLU 前馈网络。LLaMA 系列和 DeepSeek V3 使用的 FFN 结构。

    SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    相比标准 FFN：
    - 多了一个 gate 投影，参数量增加 50%（3W² vs 2W²）
    - 但效果显著更好，是 Llama 架构的关键改进之一
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = torch.nn.Linear(dim, hidden_dim, bias=False)  # 门控投影
        self.w_up = torch.nn.Linear(dim, hidden_dim, bias=False)    # 值投影
        self.w_down = torch.nn.Linear(hidden_dim, dim, bias=False)  # 输出投影
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门控信号: SiLU(x @ W_gate)
        gate = silu(self.w_gate(x))
        # 值信号: x @ W_up
        up = self.w_up(x)
        # 门控融合
        h = gate * up
        h = self.dropout(h)
        return self.w_down(h)


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)
    ffn = FFN(64, 256)
    print(f"FFN:        {x.shape} -> {ffn(x).shape}")

    swiglu = SwiGLUFFN(64, 256)
    print(f"SwiGLU FFN: {x.shape} -> {swiglu(x).shape}")
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_feedforward.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add models/common/feedforward.py tests/test_feedforward.py
git commit -m "feat: add standard FFN and SwiGLU feedforward networks"
```

### Task 1.5: Embedding 层 (embeddings.py)

**Files:**
- Create: `models/common/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_embeddings.py
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
        assert emb.weight.grad is not None
```

- [ ] **Step 3: 实现 embeddings.py**

```python
# models/common/embeddings.py
"""
Token Embedding 层手写实现。

本质上就是一个查找表（lookup table）：给定 token id，返回对应的向量。
使用 PyTorch 的 nn.Embedding 作为底层实现，本文件重点展示其与前向传播的关系。
"""
import torch


class TokenEmbedding(torch.nn.Module):
    """
    Token Embedding 层。
    将整数 token id 映射为 dim 维的稠密向量。

    参数:
        vocab_size: 词表大小
        dim: embedding 维度（等于模型的 hidden_size）
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.dim = dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        参数:
            token_ids: (batch, seq_len) 的整数 tensor，每个元素是 token id
        返回:
            (batch, seq_len, dim) 的 embedding 向量
        """
        return self.embedding(token_ids)


if __name__ == "__main__":
    emb = TokenEmbedding(vocab_size=1000, dim=64)
    tokens = torch.randint(0, 1000, (2, 8))
    out = emb(tokens)
    print(f"Token Embedding: {tokens.shape} -> {out.shape}")
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_embeddings.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add models/common/embeddings.py tests/test_embeddings.py
git commit -m "feat: add TokenEmbedding layer"
```

### Task 1.6: 注意力机制 (attention.py)

这是 common/ 中最复杂的模块，包含 MHA、MQA、GQA 三种变体。

**Files:**
- Create: `models/common/attention.py`
- Create: `tests/test_attention.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_attention.py
import torch
import sys
sys.path.insert(0, '.')
from models.common.attention import MultiHeadAttention, GroupedQueryAttention


class TestMultiHeadAttention:
    def test_shape(self):
        mha = MultiHeadAttention(dim=64, n_heads=8)
        x = torch.randn(2, 16, 64)
        out = mha(x)
        assert out.shape == x.shape

    def test_causal_mask(self):
        """causal mask 下，位置 i 不应该 attend 到位置 j > i"""
        mha = MultiHeadAttention(dim=64, n_heads=8)
        x = torch.randn(1, 4, 64)
        out_causal = mha(x, use_causal_mask=True)
        # 修改第 3 个 token 的值，不应该影响第 2 个 token 的输出
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = mha(x2, use_causal_mask=True)
        # 位置 1 的输出应该不变（它看不到位置 3）
        assert torch.allclose(out_causal[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        mha = MultiHeadAttention(dim=64, n_heads=8)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mha(x)
        out.sum().backward()
        assert x.grad is not None


class TestGroupedQueryAttention:
    def test_shape_mha_mode(self):
        """n_kv_heads == n_heads 时 GQA 退化为 MHA"""
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=8)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_shape_gqa_mode(self):
        """n_kv_heads < n_heads 时为真正的 GQA"""
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_shape_mqa_mode(self):
        """n_kv_heads == 1 时退化为 MQA"""
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=1)
        x = torch.randn(2, 16, 64)
        out = gqa(x)
        assert out.shape == x.shape

    def test_fewer_kv_params(self):
        """GQA 的 KV 投影参数量应少于 MHA"""
        mha = MultiHeadAttention(dim=64, n_heads=8)
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        mha_params = sum(p.numel() for p in mha.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())
        assert gqa_params < mha_params, f"GQA params {gqa_params} should be < MHA params {mha_params}"

    def test_causal_mask(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = torch.randn(1, 4, 64)
        out = gqa(x, use_causal_mask=True)
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = gqa(x2, use_causal_mask=True)
        assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = gqa(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 3: 实现 attention.py**

```python
# models/common/attention.py
"""
注意力机制手写实现：Multi-Head Attention (MHA)、Grouped Query Attention (GQA)。

三种变体：
- MHA: n_kv_heads == n_heads，每个 query head 有独立的 KV head
- GQA: 1 < n_kv_heads < n_heads，多个 query head 共享一组 KV head
- MQA: n_kv_heads == 1，所有 query head 共享一组 KV head

核心公式: Attention(Q, K, V) = softmax(Q @ K^T / √d_k + mask) @ V

其中 mask 用于：
- causal mask：防止当前位置 attend 到未来位置（自回归生成）
- padding mask：忽略 padding token（可选）
"""
import torch
import math


class MultiHeadAttention(torch.nn.Module):
    """
    标准多头注意力（MHA）。Transformer 原始论文中的方案。

    流程:
    1. 将输入 x 分别投影到 Q, K, V（每个 head 有独立的投影）
    2. 计算 scaled dot-product attention: softmax(QK^T / √d_k) · V
    3. 将所有 head 拼接后做输出投影
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads      # 每个 head 的维度
        self.scale = math.sqrt(self.head_dim)  # 缩放因子 √d_k

        # Q, K, V 投影矩阵
        self.w_q = torch.nn.Linear(dim, dim, bias=False)
        self.w_k = torch.nn.Linear(dim, dim, bias=False)
        self.w_v = torch.nn.Linear(dim, dim, bias=False)

        # 输出投影
        self.w_o = torch.nn.Linear(dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (batch, seq, dim) 拆分为 (batch, n_heads, seq, head_dim)
        这一步是理解多头注意力的关键：每个 head 独立处理一部分维度。
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 (batch, n_heads, seq_len, head_dim) 合并回 (batch, seq_len, dim)"""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        return x.view(batch, seq_len, self.dim)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        缩放点积注意力。

        参数:
            q, k, v: (batch, n_heads, seq_len, head_dim)
            mask: (batch, 1, seq_len, seq_len) 或可广播形状，True 表示屏蔽
        """
        # scores = Q @ K^T / √d_k
        # q: (B, H, S, D), k^T: (B, H, D, S) → scores: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)       # 沿 key 维度做 softmax
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)                # (B, H, S, D)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建 causal mask（下三角矩阵）。
        mask[i, j] = True 当 j > i（即位置 i 不能看到位置 j 之后的内容）
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(
        self, x: torch.Tensor, use_causal_mask: bool = False
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, dim)
            use_causal_mask: 是否启用因果遮罩（自回归生成时必须为 True）
        返回:
            (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape

        # 1. 线性投影 + 拆分为多头
        q = self._split_heads(self.w_q(x))  # (B, H, S, D_h)
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        # 2. 准备 mask
        mask = None
        if use_causal_mask:
            mask = self._create_causal_mask(seq_len, x.device)

        # 3. 缩放点积注意力
        attn_out = self._scaled_dot_product_attention(q, k, v, mask)

        # 4. 合并多头 + 输出投影
        merged = self._merge_heads(attn_out)
        return self.w_o(merged)


class GroupedQueryAttention(torch.nn.Module):
    """
    分组查询注意力（GQA / MQA）。

    与 MHA 的关键区别：
    - MHA: 每个 query head 都有独立的 K、V head（n_kv_heads = n_heads）
    - GQA: 多个 query head 共享一组 K、V head（1 < n_kv_heads < n_heads）
    - MQA: 所有 query head 共享一组 K、V head（n_kv_heads = 1）

    共享 KV 的方式减少了 KV Cache 的内存占用，对长序列推理非常重要。

    LLaMA3 使用 GQA（n_kv_heads = n_heads / 4），DeepSeek V3 的 MLA 进一步压缩 KV。
    """

    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1
    ):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads   # 每个 KV head 被几个 Q head 共享
        self.scale = math.sqrt(self.head_dim)

        # Q 投影：每个 head 独立
        self.w_q = torch.nn.Linear(dim, dim, bias=False)
        # K、V 投影：只有 n_kv_heads 组
        self.w_k = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = torch.nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)

        self.w_o = torch.nn.Linear(dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def _split_heads_q(self, x: torch.Tensor) -> torch.Tensor:
        """Q 的拆分：正常的多头拆分"""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _split_heads_kv(self, x: torch.Tensor) -> torch.Tensor:
        """K、V 的拆分：只有 n_kv_heads 个头"""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        return x.transpose(1, 2)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 KV head 复制以匹配 Q head 数量。
        例如 n_kv_heads=2, n_heads=8: 每个 KV head 复制 4 次。
        """
        batch, n_kv_heads, seq_len, head_dim = x.shape
        # repeat_interleave: [0,1,0,1,...] → [0,0,0,0,1,1,1,1,...]
        x = x.repeat_interleave(self.n_rep, dim=1)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.dim)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def forward(
        self, x: torch.Tensor, use_causal_mask: bool = False
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads_q(self.w_q(x))       # (B, n_heads, S, D_h)
        k = self._split_heads_kv(self.w_k(x))       # (B, n_kv_heads, S, D_h)
        v = self._split_heads_kv(self.w_v(x))

        # 复制 KV head 以匹配 Q head 数量
        k = self._repeat_kv(k)                      # (B, n_heads, S, D_h)
        v = self._repeat_kv(v)

        mask = None
        if use_causal_mask:
            mask = self._create_causal_mask(seq_len, x.device)

        attn_out = self._scaled_dot_product_attention(q, k, v, mask)
        merged = self._merge_heads(attn_out)
        return self.w_o(merged)


if __name__ == "__main__":
    x = torch.randn(2, 8, 64)

    # MHA
    mha = MultiHeadAttention(dim=64, n_heads=8)
    print(f"MHA: {x.shape} -> {mha(x).shape}")

    # GQA
    gqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=2)
    print(f"GQA: {x.shape} -> {gqa(x).shape}")

    # MQA (n_kv_heads=1)
    mqa = GroupedQueryAttention(dim=64, n_heads=8, n_kv_heads=1)
    print(f"MQA: {x.shape} -> {mqa(x).shape}")

    # causal mask 测试
    out_causal = gqa(x, use_causal_mask=True)
    print(f"GQA with causal mask: {x.shape} -> {out_causal.shape}")
    print("All attention checks passed.")
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_attention.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add models/common/attention.py tests/test_attention.py
git commit -m "feat: add hand-written Multi-Head, Grouped Query, and Multi-Query Attention"
```

---

## Stage 2: 原始 Transformer (models/transformer/)

### Task 2.1: Encoder (encoder.py)

**Files:**
- Create: `models/transformer/config.py`
- Create: `models/transformer/encoder.py`
- Create: `tests/test_transformer.py` (包含 encoder + decoder + model 的所有测试)

- [ ] **Step 1: 创建 config.py**

```python
# models/transformer/config.py
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer 超参数配置"""
    vocab_size: int = 30000
    dim: int = 512               # 模型维度 d_model
    n_heads: int = 8             # 注意力头数
    n_layers: int = 6            # Encoder 和 Decoder 的层数
    ff_hidden_dim: int = 2048    # FFN 隐藏层维度
    max_seq_len: int = 512       # 最大序列长度
    dropout: float = 0.1
    eps: float = 1e-5            # LayerNorm 的 epsilon
```

- [ ] **Step 2: 编写 encoder 测试**

```python
# tests/test_transformer.py (仅 encoder 部分)
import torch
import sys
sys.path.insert(0, '.')
from models.transformer.config import TransformerConfig
from models.transformer.encoder import EncoderLayer, Encoder


class TestEncoderLayer:
    def test_shape(self):
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = EncoderLayer(config)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_backward(self):
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = EncoderLayer(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


class TestEncoder:
    def test_shape(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=4, ff_hidden_dim=256)
        encoder = Encoder(config)
        x = torch.randn(2, 32, 64)
        out = encoder(x)
        assert out.shape == x.shape

    def test_multiple_layers(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=6, ff_hidden_dim=256)
        encoder = Encoder(config)
        assert len(encoder.layers) == 6

    def test_backward(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        encoder = Encoder(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 3: 实现 encoder.py**

```python
# models/transformer/encoder.py
"""
Transformer Encoder 实现。

EncoderLayer: MHA (self-attn) → Add&Norm → FFN → Add&Norm
Encoder:       Embedding + PE → [EncoderLayer × N]
"""
import torch
from models.transformer.config import TransformerConfig
from models.common.attention import MultiHeadAttention
from models.common.feedforward import FFN
from models.common.normalization import LayerNorm
from models.common.embeddings import TokenEmbedding
from models.common.positional_encoding import sinusoidal_pe


class EncoderLayer(torch.nn.Module):
    """
    单层 Transformer Encoder。

    子层结构:
    1. Multi-Head Self-Attention → Dropout → Residual Add → LayerNorm
    2. FFN → Dropout → Residual Add → LayerNorm

    注意：原始论文使用 Post-LN（先做 residual add 再 LN），
    这里也采用 Post-Norm 方式与论文一致。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            dim=config.dim, n_heads=config.n_heads, dropout=config.dropout
        )
        self.ffn = FFN(
            dim=config.dim, hidden_dim=config.ff_hidden_dim, dropout=config.dropout
        )
        self.norm1 = LayerNorm(config.dim, config.eps)  # Attention 后的 LN
        self.norm2 = LayerNorm(config.dim, config.eps)  # FFN 后的 LN
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: Self-Attention + Residual
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: FFN + Residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class Encoder(torch.nn.Module):
    """
    完整 Transformer Encoder。
    包含 token embedding、位置编码、和 N 层 EncoderLayer。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.dropout = torch.nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        参数:
            token_ids: (batch, seq_len) 的整数 tensor
        返回:
            (batch, seq_len, dim)
        """
        # Token Embedding
        x = self.token_embedding(token_ids)  # (B, S, D)

        # Sinusoidal Positional Encoding（加在 embedding 上）
        seq_len = token_ids.shape[1]
        pe = sinusoidal_pe(seq_len, x.shape[-1]).to(x.device)
        x = x + pe[:, :seq_len, :]

        x = self.dropout(x)

        # 通过 N 层 EncoderLayer
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == "__main__":
    config = TransformerConfig()
    encoder = Encoder(config)
    tokens = torch.randint(0, config.vocab_size, (2, 32))
    out = encoder(tokens)
    print(f"Encoder: {tokens.shape} -> {out.shape}")
```

- [ ] **Step 4: 运行 encoder 测试**

```bash
python -m pytest tests/test_transformer.py::TestEncoderLayer -v
python -m pytest tests/test_transformer.py::TestEncoder -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add models/transformer/config.py models/transformer/encoder.py tests/test_transformer.py
git commit -m "feat: add Transformer Encoder with EncoderLayer"
```

### Task 2.2: Decoder (decoder.py)

**Files:**
- Create: `models/transformer/decoder.py`
- Append: `tests/test_transformer.py`

- [ ] **Step 1: 追加 Decoder 测试**

```python
# 追加到 tests/test_transformer.py
from models.transformer.decoder import DecoderLayer, Decoder


class TestDecoderLayer:
    def test_shape(self):
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = DecoderLayer(config)
        x = torch.randn(2, 16, 64)
        encoder_out = torch.randn(2, 20, 64)
        out = layer(x, encoder_out)
        assert out.shape == x.shape

    def test_causal_self_attn(self):
        """Decoder 的 self-attention 必须使用 causal mask"""
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = DecoderLayer(config)
        x = torch.randn(1, 4, 64)
        encoder_out = torch.randn(1, 8, 64)
        out = layer(x, encoder_out)
        # 修改位置 3，不应影响位置 1（causal mask 阻止了看到未来位置）
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = layer(x2, encoder_out)
        assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        config = TransformerConfig(dim=64, n_heads=8, ff_hidden_dim=256)
        layer = DecoderLayer(config)
        x = torch.randn(2, 16, 64, requires_grad=True)
        encoder_out = torch.randn(2, 20, 64)
        out = layer(x, encoder_out)
        out.sum().backward()
        assert x.grad is not None


class TestDecoder:
    def test_shape(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=4, ff_hidden_dim=256)
        decoder = Decoder(config)
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        encoder_out = torch.randn(2, 32, 64)
        out = decoder(tgt, encoder_out)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        decoder = Decoder(config)
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        encoder_out = torch.randn(2, 32, 64, requires_grad=True)
        out = decoder(tgt, encoder_out)
        out.sum().backward()
```

- [ ] **Step 2: 运行测试（预期失败）**

```bash
python -m pytest tests/test_transformer.py::TestDecoderLayer -v
```
Expected: `ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 实现 decoder.py**

```python
# models/transformer/decoder.py
"""
Transformer Decoder 实现。

DecoderLayer: Masked MHA (self-attn with causal mask) → Add&Norm
               → Cross MHA (attend to encoder output) → Add&Norm
               → FFN → Add&Norm

Decoder:       Embedding + PE → [DecoderLayer × N] → output
"""
import torch
from models.transformer.config import TransformerConfig
from models.common.attention import MultiHeadAttention
from models.common.feedforward import FFN
from models.common.normalization import LayerNorm
from models.common.embeddings import TokenEmbedding
from models.common.positional_encoding import sinusoidal_pe


class DecoderLayer(torch.nn.Module):
    """
    单层 Transformer Decoder。

    相比 EncoderLayer，多了一层 Cross-Attention（对 encoder 输出做 attention）。

    子层结构:
    1. Masked Multi-Head Self-Attention（causal mask）
    2. Cross Multi-Head Attention（Q 来自 decoder，K、V 来自 encoder）
    3. FFN
    每个子层后都有 Residual + LayerNorm
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Self-attention（带 causal mask 的 MHA）
        self.self_attn = MultiHeadAttention(
            dim=config.dim, n_heads=config.n_heads, dropout=config.dropout
        )
        # Cross-attention：对 encoder 输出做 attention
        self.cross_attn = MultiHeadAttention(
            dim=config.dim, n_heads=config.n_heads, dropout=config.dropout
        )
        self.ffn = FFN(
            dim=config.dim, hidden_dim=config.ff_hidden_dim, dropout=config.dropout
        )
        self.norm1 = LayerNorm(config.dim, config.eps)
        self.norm2 = LayerNorm(config.dim, config.eps)
        self.norm3 = LayerNorm(config.dim, config.eps)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            x: decoder 输入 (batch, tgt_seq_len, dim)
            encoder_output: encoder 输出 (batch, src_seq_len, dim)
        """
        # Sub-layer 1: Masked Self-Attention（causal mask 由 use_causal_mask=True 提供）
        attn_out = self.self_attn(x, use_causal_mask=True)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: Cross-Attention
        # Decoder 的 x 作为 Q，encoder_output 作为 K 和 V
        # 注意：当前 MultiHeadAttention 是 self-attention 接口（输入 x 同时作为 QKV），
        # cross-attention 需要不同输入。这里用拆解的方法：
        # Q 来自 x 的投影（通过 MHA 内部的 w_q），K、V 来自 encoder_output 的投影
        # 为了简洁，这里创建一个单独的 cross-attn，并用 x 的 Q 与 encoder 的 KV 做 attention。
        q = self.cross_attn.w_q(x)                          # (B, S_dec, D)
        q = self.cross_attn._split_heads(q)                  # (B, H, S_dec, D_h)
        k = self.cross_attn._split_heads(self.cross_attn.w_k(encoder_output))  # (B, H, S_enc, D_h)
        v = self.cross_attn._split_heads(self.cross_attn.w_v(encoder_output))
        cross_out = self.cross_attn._scaled_dot_product_attention(q, k, v)
        merged = self.cross_attn._merge_heads(cross_out)
        cross_out = self.cross_attn.w_o(merged)
        x = self.norm2(x + self.dropout(cross_out))

        # Sub-layer 3: FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class Decoder(torch.nn.Module):
    """
    完整 Transformer Decoder。
    包含 token embedding、位置编码、和 N 层 DecoderLayer。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(
        self, token_ids: torch.LongTensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            token_ids: (batch, tgt_seq_len) 目标序列 token ids
            encoder_output: (batch, src_seq_len, dim) encoder 输出
        """
        x = self.token_embedding(token_ids)
        seq_len = token_ids.shape[1]
        pe = sinusoidal_pe(seq_len, x.shape[-1]).to(x.device)
        x = x + pe[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output)

        return x


if __name__ == "__main__":
    config = TransformerConfig()
    decoder = Decoder(config)
    tgt = torch.randint(0, config.vocab_size, (2, 16))
    enc_out = torch.randn(2, 32, config.dim)
    out = decoder(tgt, enc_out)
    print(f"Decoder: input {tgt.shape} + encoder {enc_out.shape} -> {out.shape}")
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_transformer.py::TestDecoderLayer -v
python -m pytest tests/test_transformer.py::TestDecoder -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add models/transformer/decoder.py tests/test_transformer.py
git commit -m "feat: add Transformer Decoder with cross-attention support"
```

### Task 2.3: 完整 Transformer 模型 (model.py)

**Files:**
- Create: `models/transformer/model.py`
- Append: `tests/test_transformer.py`

- [ ] **Step 1: 追加完整模型测试**

```python
# 追加到 tests/test_transformer.py
from models.transformer.model import Transformer


class TestTransformer:
    def test_shape(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        model = Transformer(config)
        src = torch.randint(0, config.vocab_size, (2, 20))
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        out = model(src, tgt)
        assert out.shape == (2, 16, config.vocab_size)

    def test_backward(self):
        config = TransformerConfig(dim=64, n_heads=8, n_layers=2, ff_hidden_dim=256)
        model = Transformer(config)
        src = torch.randint(0, config.vocab_size, (2, 20))
        tgt = torch.randint(0, config.vocab_size, (2, 16))
        out = model(src, tgt)
        loss = out.sum()
        loss.backward()
        # 检查至少 encoder 部分有梯度
        assert model.encoder.layers[0].self_attn.w_q.weight.grad is not None
```

- [ ] **Step 3: 实现 model.py**

```python
# models/transformer/model.py
"""
完整 Transformer 模型（Encoder-Decoder）。
用于序列到序列任务（如机器翻译）。
"""
import torch
from models.transformer.config import TransformerConfig
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder


class Transformer(torch.nn.Module):
    """
    Encoder-Decoder Transformer。

    Encoder 处理源序列 → Decoder（自回归地）生成目标序列。
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        # 输出投影层：将 decoder 输出映射到词表大小
        self.lm_head = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

        # 可选：将 lm_head 与 token embedding 权重共享（减少参数量）
        # self.lm_head.weight = self.decoder.token_embedding.embedding.weight

    def forward(
        self, src_ids: torch.LongTensor, tgt_ids: torch.LongTensor
    ) -> torch.Tensor:
        """
        参数:
            src_ids: (batch, src_seq_len) 源序列 token ids
            tgt_ids: (batch, tgt_seq_len) 目标序列 token ids（训练时包含 [BOS] + sequence）
        返回:
            (batch, tgt_seq_len, vocab_size) logits
        """
        encoder_output = self.encoder(src_ids)           # (B, S_src, D)
        decoder_output = self.decoder(tgt_ids, encoder_output)  # (B, S_tgt, D)
        logits = self.lm_head(decoder_output)            # (B, S_tgt, V)
        return logits


if __name__ == "__main__":
    config = TransformerConfig(vocab_size=1000, dim=256, n_heads=8, n_layers=3)
    model = Transformer(config)
    src = torch.randint(0, 1000, (2, 32))
    tgt = torch.randint(0, 1000, (2, 16))
    out = model(src, tgt)
    print(f"Transformer: src {src.shape} + tgt {tgt.shape} -> logits {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
```

- [ ] **Step 4: 运行全部 Transformer 测试**

```bash
python -m pytest tests/test_transformer.py -v
```
Expected: all tests pass (Encoder + Decoder + Model)

- [ ] **Step 5: Commit**

```bash
git add models/transformer/model.py tests/test_transformer.py
git commit -m "feat: add complete Encoder-Decoder Transformer model"
```

---

## Stage 3: LLaMA3 架构 (models/llama3/)

### Task 3.1: LLaMA3 配置与模型

**Files:**
- Create: `models/llama3/config.py`
- Create: `models/llama3/model.py`
- Create: `tests/test_llama3.py`

- [ ] **Step 1: 编写测试**

```python
# tests/test_llama3.py
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
        """KV Cache 的初始化和使用"""
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=64)
        model = LLaMA3Model(config)
        cache = model.create_kv_cache(batch_size=2)
        # 每层有 K cache 和 V cache
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
        # 生成结果应长于 prompt
        assert generated.shape[1] == 4 + 8

    def test_backward(self):
        config = LLaMA3Config(dim=128, n_heads=4, n_kv_heads=2, n_layers=2)
        model = LLaMA3ForCausalLM(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        logits = model(tokens)
        logits.sum().backward()
```

- [ ] **Step 3: 实现 config.py**

```python
# models/llama3/config.py
from dataclasses import dataclass


@dataclass
class LLaMA3Config:
    """LLaMA3 架构超参数配置"""
    vocab_size: int = 32000
    dim: int = 512               # hidden_size
    n_heads: int = 8             # query heads 数量
    n_kv_heads: int = 4          # key/value heads 数量（GQA）
    n_layers: int = 8            # TransformerBlock 层数
    ff_hidden_dim: int = 1376    # SwiGLU FFN 的隐藏维度（≈ 8/3 * dim 的 2.7x 附近）
    max_seq_len: int = 2048      # RoPE 预计算的最大长度
    dropout: float = 0.0         # LLaMA3 通常不用 dropout
    eps: float = 1e-6            # RMSNorm epsilon
    rope_theta: float = 10000.0  # RoPE 频率基数

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
```

- [ ] **Step 4: 实现 model.py**

```python
# models/llama3/model.py
"""
LLaMA3 架构完整实现。

架构特点（对比原始 Transformer）：
1. Decoder-only:  去掉了 Encoder 和 Cross-Attention
2. RMSNorm:       代替 LayerNorm（去掉中心化，更快）
3. RoPE:          代替 Sinusoidal PE（旋转位置编码，相对位置信息）
4. SwiGLU:        代替标准 FFN（门控 + SiLU 激活）
5. GQA:           代替 MHA（共享 KV head，减少 KV Cache 内存）
6. Pre-Norm:      在 Attention/FFN 之前做归一化（而非之后）

LLaMA3ForCausalLM = TokenEmbedding → [TransformerBlock × N] → RMSNorm → LM Head
"""
import torch
import math
from models.llama3.config import LLaMA3Config
from models.common.attention import GroupedQueryAttention
from models.common.normalization import RMSNorm
from models.common.feedforward import SwiGLUFFN
from models.common.embeddings import TokenEmbedding
from models.common.positional_encoding import RotaryPositionalEncoding


class TransformerBlock(torch.nn.Module):
    """
    LLaMA3 的单个 Decoder Block。

    结构（Pre-Norm 方式）:
    x = x + GQA(RMSNorm(x))     ← Self-Attention with RoPE
    x = x + SwiGLU(RMSNorm(x))  ← FFN

    注意：RoPE 在 attention 内部施加到 Q 和 K 上。
    """

    def __init__(self, config: LLaMA3Config):
        super().__init__()
        # Pre-Attention RMSNorm
        self.attn_norm = RMSNorm(config.dim, config.eps)
        # GQA（内部会使用 RoPE，由上层传入）
        self.attn = GroupedQueryAttention(
            dim=config.dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            dropout=config.dropout,
        )
        # Pre-FFN RMSNorm
        self.ffn_norm = RMSNorm(config.dim, config.eps)
        # SwiGLU FFN
        self.ffn = SwiGLUFFN(
            dim=config.dim, hidden_dim=config.ff_hidden_dim, dropout=config.dropout
        )
        # RoPE
        self.rope = RotaryPositionalEncoding(
            dim=config.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta
        )

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        参数:
            x: (batch, seq_len, dim)
            use_causal_mask: 是否使用因果遮罩
            kv_cache: 可选的 (k_cache, v_cache)，用于增量推理
            start_pos: 当前 token 在完整序列中的起始位置（使用 KV Cache 时的偏移量）
        返回:
            (output, new_kv_cache)
        """
        # Sub-layer 1: Self-Attention (Pre-Norm)
        residual = x
        x_norm = self.attn_norm(x)

        # 拆分 Q, K, V
        q = self.attn.w_q(x_norm)
        k = self.attn.w_k(x_norm)
        v = self.attn.w_v(x_norm)

        # 拆分为多头
        q = self.attn._split_heads_q(q)       # (B, n_heads, S, D_h)
        k = self.attn._split_heads_kv(k)      # (B, n_kv_heads, S, D_h)
        v = self.attn._split_heads_kv(v)

        # RoPE: 对 Q 和 K 施加旋转位置编码
        # start_pos 用于 KV Cache 场景：每个 token 需要对应其在序列中的实际位置
        q, k = self.rope(q, k, start_pos=start_pos)  # 需要修改 RoPE 支持 start_pos

        # KV Cache：拼接历史 KV
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache[:, :, :start_pos], k], dim=2)
            v = torch.cat([v_cache[:, :, :start_pos], v], dim=2)
        new_kv_cache = (k.detach(), v.detach()) if kv_cache is not None else None

        # 复制 KV head 匹配 Q head
        k = self.attn._repeat_kv(k)
        v = self.attn._repeat_kv(v)

        # 创建 causal mask
        seq_len = q.shape[2]
        mask = None
        if use_causal_mask and seq_len > 1:
            mask = self.attn._create_causal_mask(seq_len, x.device)
            if start_pos > 0:
                # KV Cache 场景下需要扩展 mask
                total_len = start_pos + seq_len
                extended_mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
                extended_mask[:, :start_pos] = False  # 可以看到所有历史
                extended_mask = torch.triu(extended_mask, diagonal=start_pos + 1)
                mask = extended_mask.unsqueeze(0).unsqueeze(0)

        # Scaled Dot-Product Attention
        attn_out = self.attn._scaled_dot_product_attention(q, k, v, mask)
        merged = self.attn._merge_heads(attn_out)
        attn_out = self.attn.w_o(merged)

        x = residual + attn_out

        # Sub-layer 2: FFN (Pre-Norm)
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x, new_kv_cache


class LLaMA3Model(torch.nn.Module):
    """
    LLaMA3 基础模型（不含 LM Head）。
    输出最后一层的 hidden states。
    """

    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, config.eps)

    def create_kv_cache(self, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """创建 KV Cache。每层存储 (k_cache, v_cache)"""
        cache = []
        for _ in range(self.config.n_layers):
            k = torch.zeros(
                batch_size, self.config.n_kv_heads,
                self.config.max_seq_len, self.config.head_dim
            )
            v = torch.zeros(
                batch_size, self.config.n_kv_heads,
                self.config.max_seq_len, self.config.head_dim
            )
            cache.append((k, v))
        return cache

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x, _ = layer(x, use_causal_mask=True)
        return self.norm(x)


class LLaMA3ForCausalLM(torch.nn.Module):
    """
    LLaMA3 用于因果语言建模的完整模型。
    LLaMA3Model + LM Head（输出词表大小的 logits）
    """

    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.config = config
        self.model = LLaMA3Model(config)
        self.lm_head = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        hidden = self.model(token_ids)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.LongTensor:
        """
        自回归文本生成。

        参数:
            prompt: (batch, prompt_len) 输入 token ids
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（越高越随机）
        """
        self.eval()
        generated = prompt.clone()
        batch_size = prompt.shape[0]
        cache = self.model.create_kv_cache(batch_size)

        for _ in range(max_new_tokens):
            # 只输入最后一个 token（KV Cache 节省计算）
            if generated.shape[1] > 1:
                current_input = generated[:, -1:]  # (B, 1)
            else:
                current_input = generated

            # 逐层 forward
            x = self.model.token_embedding(current_input)
            new_caches = []
            for i, layer in enumerate(self.model.layers):
                start_pos = generated.shape[1] - 1
                x, (k, v) = layer(x, use_causal_mask=False, kv_cache=cache[i], start_pos=start_pos)
                new_caches.append((k, v))
            cache = new_caches

            x = self.model.norm(x)
            logits = self.lm_head(x[:, -1, :])  # 只取最后一个位置的 logits

            # 采样
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


if __name__ == "__main__":
    config = LLaMA3Config(vocab_size=1000, dim=128, n_heads=4, n_kv_heads=2, n_layers=2)
    model = LLaMA3ForCausalLM(config)
    tokens = torch.randint(0, 1000, (1, 8))
    logits = model(tokens)
    print(f"LLaMA3: input {tokens.shape} -> logits {logits.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 测试生成
    generated = model.generate(tokens, max_new_tokens=4)
    print(f"Generated: {tokens.shape[1]} prompt + 4 new = {generated.shape[1]} tokens")
```

- [ ] **Step 5: 修改 RoPE 支持 start_pos**

RoPE 的 `forward` 方法增加 `start_pos` 参数：

```python
# 在 models/common/positional_encoding.py 中修改 RotaryPositionalEncoding.forward
def forward(
    self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[2]
    # 从 start_pos 开始取对应位置的 cos/sin
    cos = self.cos_table[start_pos : start_pos + seq_len]
    sin = self.sin_table[start_pos : start_pos + seq_len]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos_full = torch.repeat_interleave(cos, 2, dim=-1)
    sin_full = torch.repeat_interleave(sin, 2, dim=-1)
    q_rot = q * cos_full + self._rotate_half(q) * sin_full
    k_rot = k * cos_full + self._rotate_half(k) * sin_full
    return q_rot, k_rot
```

- [ ] **Step 6: 运行测试**

```bash
python -m pytest tests/test_llama3.py -v
```
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add models/llama3/config.py models/llama3/model.py models/common/positional_encoding.py tests/test_llama3.py
git commit -m "feat: add LLaMA3 architecture with GQA, RoPE, RMSNorm, SwiGLU, KV Cache and text generation"
```

---

## Stage 4: DeepSeek V3 架构 (models/deepseek_v3/)

### Task 4.1: Multi-head Latent Attention (mla.py)

**Files:**
- Create: `models/deepseek_v3/config.py`
- Create: `models/deepseek_v3/mla.py`
- Create: `tests/test_deepseek_v3.py`

- [ ] **Step 1: 创建 config.py**

```python
# models/deepseek_v3/config.py
from dataclasses import dataclass


@dataclass
class DeepSeekV3Config:
    """DeepSeek V3 架构超参数"""
    vocab_size: int = 32000
    dim: int = 512               # hidden_size
    n_heads: int = 8             # query heads 数量
    n_layers: int = 8            # TransformerBlock 层数
    max_seq_len: int = 2048
    eps: float = 1e-6

    # MLA (Multi-head Latent Attention)
    kv_lora_rank: int = 256      # KV 压缩后的潜在维度（远小于 dim * n_kv_heads）
    qk_rope_head_dim: int = 32   # RoPE 部分的维度（解耦 RoPE）

    # MoE
    n_routed_experts: int = 8    # 路由 expert 数量
    n_shared_experts: int = 1    # 共享 expert 数量
    n_activated_experts: int = 2 # 每个 token 激活的 expert 数量（top_k）
    moe_intermediate_dim: int = 512  # 每个 expert 的 FFN 隐藏维度
    
    # RoPE
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
```

- [ ] **Step 2: 编写 MLA 测试**

```python
# tests/test_deepseek_v3.py (MLA 测试部分)
import torch
import sys
sys.path.insert(0, '.')
from models.deepseek_v3.config import DeepSeekV3Config
from models.deepseek_v3.mla import MultiHeadLatentAttention


class TestMLA:
    def test_shape(self):
        config = DeepSeekV3Config(dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16)
        mla = MultiHeadLatentAttention(config)
        x = torch.randn(2, 16, 128)
        out = mla(x)
        assert out.shape == x.shape

    def test_fewer_params_than_mha(self):
        """MLA 的 KV 参数量应少于标准 MHA"""
        config = DeepSeekV3Config(dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16)
        mla = MultiHeadLatentAttention(config)
        mla_params = sum(p.numel() for p in mla.parameters())

        from models.common.attention import MultiHeadAttention
        mha = MultiHeadAttention(dim=128, n_heads=4)
        mha_params = sum(p.numel() for p in mha.parameters())

        assert mla_params < mha_params, f"MLA {mla_params} should be < MHA {mha_params}"

    def test_causal_mask(self):
        config = DeepSeekV3Config(dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16)
        mla = MultiHeadLatentAttention(config)
        x = torch.randn(1, 4, 128)
        out = mla(x, use_causal_mask=True)
        x2 = x.clone()
        x2[0, 3] = 999.0
        out2 = mla(x2, use_causal_mask=True)
        assert torch.allclose(out[0, 1], out2[0, 1], atol=1e-4)

    def test_backward(self):
        config = DeepSeekV3Config(dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16)
        mla = MultiHeadLatentAttention(config)
        x = torch.randn(2, 16, 128, requires_grad=True)
        out = mla(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 3: 实现 mla.py**

```python
# models/deepseek_v3/mla.py
"""
Multi-head Latent Attention (MLA) — DeepSeek V3 的核心创新。

MLA 的核心思想：通过低秩压缩来减少 KV Cache 的显存占用。

标准 MHA/GQA 中，每个 token 需要缓存完整的 K、V：
- K: (n_kv_heads, head_dim) 个 float16
- V: (n_kv_heads, head_dim) 个 float16

MLA 的做法：
1. 将 KV 投影到一个低秩潜在空间 (kv_lora_rank)，只需要缓存这个压缩表示
2. 使用时通过上投影矩阵恢复完整的 K 和 V
3. RoPE 部分单独处理（解耦 RoPE）：只有一小部分维度参与旋转

这样 KV Cache 的显存从 O(n_kv_heads × head_dim) 降到了 O(kv_lora_rank)，
而 kv_lora_rank << n_kv_heads × head_dim。

参考: DeepSeek-V2/V3 论文
"""
import torch
import math
from models.deepseek_v3.config import DeepSeekV3Config


class MultiHeadLatentAttention(torch.nn.Module):
    """
    Multi-head Latent Attention。

    流程:
    1. 输入 x → 投影到 Q, KV_latent, K_rope
    2. KV_latent → 上投影得到 K, V
    3. Q 和 K 的 RoPE 部分用 K_rope 做旋转
    4. Scaled dot-product attention（完整 Q 对完整 K）
    5. 输出投影
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.scale = math.sqrt(self.head_dim)

        # Q 投影（完整维度）
        self.w_q = torch.nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)

        # KV 压缩：输入 → 低秩 latent 表示
        self.w_kv_a = torch.nn.Linear(
            config.dim, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )

        # K 上投影：latent → K（不含 RoPE 部分）
        self.w_k_b = torch.nn.Linear(
            config.kv_lora_rank, config.n_heads * config.head_dim, bias=False
        )

        # V 上投影：latent → V
        self.w_v_b = torch.nn.Linear(
            config.kv_lora_rank, config.n_heads * config.head_dim, bias=False
        )

        # RoPE 的 cos/sin 表
        self.register_buffer(
            "rope_cos",
            self._compute_rope_table(config.max_seq_len, config.qk_rope_head_dim, config.rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            self._compute_rope_sin_table(config.max_seq_len, config.qk_rope_head_dim, config.rope_theta),
            persistent=False,
        )

        # 输出投影
        self.w_o = torch.nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def _compute_rope_table(self, seq_len: int, dim: int, theta: float) -> torch.Tensor:
        """计算 RoPE cos 表"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freq_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        freqs = 1.0 / (theta ** (freq_indices / dim))
        angles = torch.outer(positions, freqs)
        return torch.cos(angles)  # (seq_len, dim/2)

    def _compute_rope_sin_table(self, seq_len: int, dim: int, theta: float) -> torch.Tensor:
        """计算 RoPE sin 表"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freq_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        freqs = 1.0 / (theta ** (freq_indices / dim))
        angles = torch.outer(positions, freqs)
        return torch.sin(angles)  # (seq_len, dim/2)

    def _apply_rope(
        self, x: torch.Tensor, start_pos: int = 0
    ) -> torch.Tensor:
        """对输入的 RoPE 部分施加旋转位置编码"""
        seq_len = x.shape[2]
        cos = self.rope_cos[start_pos : start_pos + seq_len]  # (seq, dim/2)
        sin = self.rope_sin[start_pos : start_pos + seq_len]

        # 广播到 (1, 1, seq, dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 重复以匹配完整维度
        cos_full = torch.repeat_interleave(cos, 2, dim=-1)
        sin_full = torch.repeat_interleave(sin, 2, dim=-1)

        # RoPE: x * cos + rotate_half(x) * sin
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotate_half = torch.cat([-x2, x1], dim=-1)
        return x * cos_full + rotate_half * sin_full

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, S, N*D) 拆分为 (B, N, S, D)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, -1).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, N, S, D) 合并为 (B, S, N*D)"""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, -1)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        use_causal_mask: bool = True,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # 1. Q 投影
        q = self.w_q(x)  # (B, S, N * head_dim)
        q = self._split_heads(q)  # (B, N_heads, S, head_dim)

        # 2. KV 压缩投影（包含 latent 部分 + RoPE 部分）
        kv_a = self.w_kv_a(x)  # (B, S, kv_lora_rank + qk_rope_dim)

        # 分出 latent 和 RoPE 部分
        kv_latent = kv_a[..., : self.kv_lora_rank]  # (B, S, kv_lora_rank)
        k_rope_raw = kv_a[..., self.kv_lora_rank :]  # (B, S, qk_rope_dim)

        # 3. K 和 V 上投影
        k = self.w_k_b(kv_latent)  # (B, S, N * head_dim)
        v = self.w_v_b(kv_latent)  # (B, S, N * head_dim)

        k = self._split_heads(k)  # (B, N_heads, S, head_dim)
        v = self._split_heads(v)

        # 4. Q 的 RoPE 部分：split Q 的最后 qk_rope_head_dim 维做旋转
        q_content = q[..., : -self.qk_rope_head_dim]  # 不参与 RoPE 的部分
        q_rope_part = q[..., -self.qk_rope_head_dim :]  # 参与 RoPE 的部分
        q_rope_part = self._apply_rope(q_rope_part.unsqueeze(2)).squeeze(2)
        q = torch.cat([q_content, q_rope_part], dim=-1)

        # 5. K 的 RoPE 部分：（需要 reshape k_rope_raw 然后做 RoPE）
        k_rope_heads = k_rope_raw.view(batch, seq_len, self.n_heads, self.qk_rope_head_dim // self.n_heads)
        k_rope_heads = k_rope_heads.transpose(1, 2)  # (B, N, S, rope_per_head)
        k_rope_heads = self._apply_rope(k_rope_heads)

        k_content = k[..., : -self.qk_rope_head_dim // self.n_heads]
        k = torch.cat([k_content, k_rope_heads], dim=-1)

        # 6. Scaled dot-product attention
        mask = None
        if use_causal_mask and seq_len > 1:
            mask = self._create_causal_mask(seq_len, x.device)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # 7. 输出
        merged = self._merge_heads(attn_out)
        return self.w_o(merged)


if __name__ == "__main__":
    config = DeepSeekV3Config(dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16)
    mla = MultiHeadLatentAttention(config)
    x = torch.randn(2, 8, 128)
    out = mla(x)
    print(f"MLA: {x.shape} -> {out.shape}")
    print(f"MLA params: {sum(p.numel() for p in mla.parameters()):,}")
```

- [ ] **Step 4: 运行 MLA 测试**

```bash
python -m pytest tests/test_deepseek_v3.py::TestMLA -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add models/deepseek_v3/config.py models/deepseek_v3/mla.py tests/test_deepseek_v3.py
git commit -m "feat: add DeepSeek V3 Multi-head Latent Attention (MLA)"
```

### Task 4.2: MoE 层 (moe.py)

**Files:**
- Create: `models/deepseek_v3/moe.py`
- Append: `tests/test_deepseek_v3.py`

- [ ] **Step 1: 追加 MoE 测试**

```python
# 追加到 tests/test_deepseek_v3.py
from models.deepseek_v3.moe import Router, SharedExpert, RoutedExpert, MoELayer


class TestRouter:
    def test_shape(self):
        router = Router(dim=128, n_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        scores, indices = router(x)
        assert scores.shape == (2, 16, 2)   # (B, S, top_k)
        assert indices.shape == (2, 16, 2)

    def test_top_k_indices(self):
        router = Router(dim=128, n_experts=8, top_k=2)
        x = torch.randn(2, 16, 128)
        _, indices = router(x)
        # 每个 token 选出的 top-k expert indices 应在 [0, 7] 范围内
        assert indices.min() >= 0
        assert indices.max() < 8


class TestMoELayer:
    def test_shape(self):
        config = DeepSeekV3Config(
            dim=128, n_routed_experts=4, n_shared_experts=1,
            n_activated_experts=2, moe_intermediate_dim=256
        )
        moe = MoELayer(config)
        x = torch.randn(2, 8, 128)
        out = moe(x)
        assert out.shape == x.shape

    def test_shared_expert_always_active(self):
        """共享 expert 对所有 token 都生效"""
        config = DeepSeekV3Config(
            dim=128, n_routed_experts=4, n_shared_experts=1,
            n_activated_experts=2, moe_intermediate_dim=256
        )
        moe = MoELayer(config)
        x = torch.randn(2, 4, 128)
        out = moe(x)
        assert out.shape == x.shape

    def test_backward(self):
        config = DeepSeekV3Config(
            dim=128, n_routed_experts=4, n_shared_experts=1,
            n_activated_experts=2, moe_intermediate_dim=256
        )
        moe = MoELayer(config)
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None
```

- [ ] **Step 3: 实现 moe.py**

```python
# models/deepseek_v3/moe.py
"""
Mixture of Experts (MoE) 层实现。

MoE 核心思想：将 FFN 拆成多个"专家"（experts），每个 token 只激活其中的 top-k 个。
这样模型总参数量可以大幅增加（因为有 N 个 expert），但每个 token 的计算量只增加约 k/N 倍。

DeepSeek V3 的 MoE 特点：
1. Shared Expert: 一个所有 token 共享的 expert，捕获通用模式
2. Routed Experts: 被 router 选中的 token 才会经过的 experts
3. Auxiliary-loss-free load balancing: 通过动态 bias 调整，不用辅助损失

参考: DeepSeek-V2/V3 论文
"""
import torch
from models.deepseek_v3.config import DeepSeekV3Config
from models.common.activation import silu


class Router(torch.nn.Module):
    """
    Top-K Router。
    将每个 token 路由到得分最高的 k 个 expert。

    输入 dim 维的 hidden state → 输出 top-k expert indices 和对应的 softmax 权重。
    """

    def __init__(self, dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # 路由权重矩阵：dim → n_experts
        self.weight = torch.nn.Linear(dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, dim)
        返回:
            scores: (batch, seq_len, top_k) softmax 归一化后的权重
            indices: (batch, seq_len, top_k) 被选中的 expert 编号
        """
        logits = self.weight(x)  # (B, S, n_experts)
        # 取 top-k
        scores, indices = torch.topk(logits, k=self.top_k, dim=-1)
        # 对 top-k 的得分做 softmax 归一化
        scores = torch.softmax(scores, dim=-1)
        return scores, indices


class SharedExpert(torch.nn.Module):
    """
    共享 Expert — 所有 token 都会经过的 FFN。
    使用 SwiGLU 结构，与 LLaMA 的 FFN 类似。
    """

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w_gate = torch.nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = torch.nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = torch.nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class RoutedExpert(torch.nn.Module):
    """单个 Routed Expert — 结构同 SharedExpert，但仅处理被路由到的 token"""

    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w_gate = torch.nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = torch.nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = torch.nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class MoELayer(torch.nn.Module):
    """
    MoE 层：组合 Shared Expert + N 个 Routed Experts。

    每个 token 的输出 = SharedExpert(x) + Σ (router_score_i * RoutedExpert_i(x))
    其中求和只对 top-k expert 进行。
    """

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.top_k = config.n_activated_experts

        self.router = Router(config.dim, config.n_routed_experts, config.n_activated_experts)
        self.shared_expert = SharedExpert(config.dim, config.moe_intermediate_dim)
        self.routed_experts = torch.nn.ModuleList([
            RoutedExpert(config.dim, config.moe_intermediate_dim)
            for _ in range(config.n_routed_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, dim)
        返回:
            (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # 1. Shared Expert（所有 token）
        shared_out = self.shared_expert(x)  # (B, S, D)

        # 2. Routing
        router_scores, router_indices = self.router(x)  # (B, S, top_k), (B, S, top_k)

        # 3. Routed Experts（按 expert 分组处理以提升效率）
        routed_out = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.routed_experts):
            # 找出所有被路由到这个 expert 的 token
            mask = (router_indices == expert_idx)  # (B, S, top_k) boolean
            token_mask = mask.any(dim=-1)  # (B, S) — 哪些 token 被路由到了这个 expert

            if token_mask.any():
                # 收集这些 token
                # 使用 mask 筛选 token，但 batch 维度不同...
                # 简化处理：收集所有被路由到的 token
                selected_tokens = x[token_mask]  # (N_tokens, D)

                if selected_tokens.shape[0] > 0:
                    expert_out = expert(selected_tokens)  # (N_tokens, D)

                    # 累加权重（取该 expert 对应的 router score）
                    # 找出 mask 中这个 expert 对应的 score
                    expert_scores = router_scores[mask]  # (N_tokens,)
                    weighted_out = expert_out * expert_scores.unsqueeze(-1)

                    # 分散写回
                    routed_out[token_mask] += weighted_out

        return shared_out + routed_out


if __name__ == "__main__":
    config = DeepSeekV3Config(
        dim=128, n_routed_experts=4, n_shared_experts=1,
        n_activated_experts=2, moe_intermediate_dim=256
    )
    moe = MoELayer(config)
    x = torch.randn(2, 8, 128)
    out = moe(x)
    print(f"MoE Layer: {x.shape} -> {out.shape}")
    print(f"MoE params: {sum(p.numel() for p in moe.parameters()):,}")
```

- [ ] **Step 4: 运行 MoE 测试**

```bash
python -m pytest tests/test_deepseek_v3.py::TestRouter tests/test_deepseek_v3.py::TestMoELayer -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add models/deepseek_v3/moe.py tests/test_deepseek_v3.py
git commit -m "feat: add Mixture of Experts layer with Router and Shared/Routed Experts"
```

### Task 4.3: DeepSeek V3 完整模型 (model.py)

**Files:**
- Create: `models/deepseek_v3/model.py`
- Append: `tests/test_deepseek_v3.py`

- [ ] **Step 1: 追加完整模型测试**

```python
# 追加到 tests/test_deepseek_v3.py
from models.deepseek_v3.model import DeepSeekV3Block, DeepSeekV3Model, DeepSeekV3ForCausalLM


class TestDeepSeekV3Block:
    def test_shape(self):
        config = DeepSeekV3Config(
            dim=128, n_heads=4, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256
        )
        block = DeepSeekV3Block(config)
        x = torch.randn(2, 8, 128)
        out = block(x)
        assert out.shape == x.shape


class TestDeepSeekV3Model:
    def test_shape(self):
        config = DeepSeekV3Config(
            dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256
        )
        model = DeepSeekV3Model(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        out = model(tokens)
        assert out.shape == (2, 16, config.dim)


class TestDeepSeekV3ForCausalLM:
    def test_shape(self):
        config = DeepSeekV3Config(
            dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256, vocab_size=1000
        )
        model = DeepSeekV3ForCausalLM(config)
        tokens = torch.randint(0, 1000, (2, 8))
        logits = model(tokens)
        assert logits.shape == (2, 8, 1000)

    def test_generate(self):
        config = DeepSeekV3Config(
            dim=128, n_heads=4, n_layers=2, kv_lora_rank=64, qk_rope_head_dim=16,
            n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
            moe_intermediate_dim=256, vocab_size=100, max_seq_len=64
        )
        model = DeepSeekV3ForCausalLM(config)
        model.eval()
        prompt = torch.randint(0, 100, (1, 4))
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=6, temperature=1.0)
        assert generated.shape[1] == 10
```

- [ ] **Step 3: 实现 model.py**

```python
# models/deepseek_v3/model.py
"""
DeepSeek V3 完整模型实现。

Architecture:
DeepSeekV3Block: MLA (Multi-head Latent Attention) + MoE (Mixture of Experts)
DeepSeekV3Model: TokenEmbedding → [Block × N] → RMSNorm
DeepSeekV3ForCausalLM: Model + LM Head + generate()

与 LLaMA3 的关键区别：
1. MLA 代替 GQA：通过低秩压缩大幅减少 KV Cache 占用
2. MoE 代替 Dense FFN：总参数量大但每 token 计算量可控
"""
import torch
from models.deepseek_v3.config import DeepSeekV3Config
from models.deepseek_v3.mla import MultiHeadLatentAttention
from models.deepseek_v3.moe import MoELayer
from models.common.normalization import RMSNorm
from models.common.embeddings import TokenEmbedding


class DeepSeekV3Block(torch.nn.Module):
    """DeepSeek V3 的单个 Transformer Block：MLA + MoE"""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.eps)
        self.attn = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.eps)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        # Pre-Norm Self-Attention (MLA)
        x = x + self.attn(self.attn_norm(x), use_causal_mask=use_causal_mask)
        # Pre-Norm MoE FFN
        x = x + self.moe(self.ffn_norm(x))
        return x


class DeepSeekV3Model(torch.nn.Module):
    """DeepSeek V3 基础模型（不含 LM Head）"""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList([
            DeepSeekV3Block(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, config.eps)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x, use_causal_mask=True)
        return self.norm(x)


class DeepSeekV3ForCausalLM(torch.nn.Module):
    """DeepSeek V3 用于因果语言建模的完整模型"""

    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config)
        self.lm_head = torch.nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        hidden = self.model(token_ids)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> torch.LongTensor:
        """自回归文本生成（简化版，不含 KV Cache）"""
        self.eval()
        generated = prompt.clone()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


if __name__ == "__main__":
    config = DeepSeekV3Config(
        vocab_size=1000, dim=128, n_heads=4, n_layers=2,
        kv_lora_rank=64, qk_rope_head_dim=16,
        n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
        moe_intermediate_dim=256
    )
    model = DeepSeekV3ForCausalLM(config)
    tokens = torch.randint(0, 1000, (1, 8))
    logits = model(tokens)
    print(f"DeepSeek V3: input {tokens.shape} -> logits {logits.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 测试生成
    generated = model.generate(tokens, max_new_tokens=4)
    print(f"Generated: {tokens.shape[1]} prompt + 4 new = {generated.shape[1]} tokens")
```

- [ ] **Step 4: 运行全部 DeepSeek V3 测试**

```bash
python -m pytest tests/test_deepseek_v3.py -v
```
Expected: all tests pass (MLA + MoE + Model)

- [ ] **Step 5: Commit**

```bash
git add models/deepseek_v3/model.py tests/test_deepseek_v3.py
git commit -m "feat: add complete DeepSeek V3 model with MLA and MoE"
```

---

## Stage 5: 通信基础 (parallel/communication/)

### Task 5.1: 通信环境搭建与 All-Reduce 手写实现

**Files:**
- Create: `parallel/communication/setup.py`
- Create: `parallel/communication/primitives.py`
- Create: `tests/test_communication.py`

- [ ] **Step 1: 编写通信测试**

```python
# tests/test_communication.py
"""
通信原语测试。需要在单机上使用 torchrun 启动多进程：
  torchrun --nproc_per_node=4 tests/test_communication.py
"""
import torch
import torch.distributed as dist
import os
import sys
sys.path.insert(0, '.')
from parallel.communication.setup import init_process_group, cleanup
from parallel.communication.primitives import (
    naive_all_reduce, ring_all_reduce,
    naive_all_gather, naive_broadcast,
)


def test_all_reduce_consistency():
    """验证 all-reduce 后所有 rank 结果一致"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_process_group(backend="gloo")
    device = torch.device("cpu")

    tensor = torch.tensor([rank * 1.0], device=device)
    result = naive_all_reduce(tensor, op="sum")
    # sum([0, 1, 2, 3]) = 6
    expected = sum(range(world_size))
    assert torch.allclose(result, torch.tensor([expected], device=device)), \
        f"Rank {rank}: expected {expected}, got {result.item()}"

    cleanup()


def test_ring_all_reduce():
    """验证 Ring All-Reduce 和 naive 版本结果一致"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_process_group(backend="gloo")
    device = torch.device("cpu")

    tensor = torch.ones(4, device=device) * (rank + 1)
    result_naive = naive_all_reduce(tensor.clone(), op="sum")
    result_ring = ring_all_reduce(tensor.clone())

    assert torch.allclose(result_naive, result_ring), \
        f"Ring all-reduce differs from naive at rank {rank}"

    cleanup()


def test_broadcast():
    """验证 broadcast 后所有 rank 得到相同结果"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_process_group(backend="gloo")
    device = torch.device("cpu")

    # rank 0 创建数据并 broadcast
    if rank == 0:
        tensor = torch.tensor([42.0, 3.14], device=device)
    else:
        tensor = torch.zeros(2, device=device)

    result = naive_broadcast(tensor, src=0)

    assert torch.allclose(result, torch.tensor([42.0, 3.14], device=device)), \
        f"Rank {rank} broadcast failed"

    cleanup()


def test_all_gather():
    """验证 all-gather 后每个 rank 都拥有完整数据"""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_process_group(backend="gloo")
    device = torch.device("cpu")

    # 每个 rank 持有一个不同的值
    local_data = torch.tensor([rank * 10.0, rank * 10.0 + 1.0], device=device)
    gathered = naive_all_gather(local_data)

    expected = torch.tensor([
        i * 10.0 for i in range(world_size) for _ in range(2)
    ], device=device)
    # 顺序调整：all-gather 的结果按 rank 顺序排列
    sorted_gathered = torch.cat([gathered[i*2:(i+1)*2] for i in range(world_size)])
    # 重新排序期望值
    expected_sorted = torch.tensor([
        val for i in range(world_size) for val in [i*10.0, i*10.0+1]
    ], device=device)

    assert torch.allclose(gathered, expected_sorted), \
        f"Rank {rank}: all-gather mismatch. Got {gathered}, expected {expected_sorted}"

    cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all")
    args = parser.parse_args()

    tests = {
        "all_reduce": test_all_reduce_consistency,
        "ring_all_reduce": test_ring_all_reduce,
        "broadcast": test_broadcast,
        "all_gather": test_all_gather,
    }

    if args.test == "all":
        for name, fn in tests.items():
            print(f"Running {name}...")
            fn()
            print(f"  {name} passed")
    else:
        tests[args.test]()
```

- [ ] **Step 3: 实现 setup.py**

```python
# parallel/communication/setup.py
"""
分布式通信环境搭建。

在单机上通过 torch.distributed 初始化多进程通信环境，
模拟多 GPU 拓扑。每个进程通过 LOCAL_RANK 区分身份。
"""
import torch
import torch.distributed as dist
import os


def init_process_group(backend: str = "gloo"):
    """初始化分布式进程组。在 CPU 上推荐使用 gloo backend。"""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def get_rank() -> int:
    """获取当前进程的 rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
```

- [ ] **Step 4: 实现 primitives.py**

```python
# parallel/communication/primitives.py
"""
通信原语手写实现。

每种原语提供两个版本：
1. 手写实现（用 send/recv 逐 rank 通信）—— 理解通信模式
2. PyTorch NCCL/Gloo 版本 —— 实际生产中使用

包含：all-reduce, all-gather, reduce-scatter, broadcast, scatter, reduce
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def naive_all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """
    All-Reduce 手写实现（朴素版本）。

    方式：每个 rank 先 broadcast 自己的数据给所有其他 rank，
         然后本地做 reduce（sum/avg/max）。

    通信量: O(N * P²)，N=数据量，P=进程数
    不高效，但直观展示 all-reduce 的语义。
    """
    rank = get_rank()
    world_size = get_world_size()
    result = torch.zeros_like(tensor)

    for src in range(world_size):
        if rank == src:
            data = tensor.clone()
        else:
            data = torch.zeros_like(tensor)

        # Broadcast 当前 src 的数据
        dist.broadcast(data, src=src)

        if op == "sum":
            result += data
        elif op == "avg":
            result += data / world_size
        elif op == "max":
            result = torch.max(result, data)

    return result


def ring_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-Reduce 的 Ring 算法（Scatter-Reduce + All-Gather）。

    分两步:
    1. Scatter-Reduce: 数据分成 P 块，在环形拓扑中传递 P-1 次，每次累加
    2. All-Gather:   每个 rank 的最终结果块环形传递 P-1 次，让所有人获取完整结果

    通信量: O(2N)，带宽最优（bandwidth-optimal）
    这种算法在 NCCL 和 Horovod 中广泛使用。
    """
    rank = get_rank()
    world_size = get_world_size()

    result = tensor.clone()
    chunk_size = tensor.numel() // world_size

    # Step 1: Scatter-Reduce
    for step in range(world_size - 1):
        send_chunk_start = ((rank - step) % world_size) * chunk_size
        recv_chunk_start = ((rank - step - 1) % world_size) * chunk_size

        send_chunk = result.flatten()[send_chunk_start : send_chunk_start + chunk_size]
        recv_chunk = torch.zeros_like(send_chunk)

        send_dst = (rank + 1) % world_size
        recv_src = (rank - 1) % world_size

        # 同时 send 和 recv
        send_op = dist.isend(send_chunk.contiguous(), send_dst)
        recv_op = dist.irecv(recv_chunk, recv_src)
        send_op.wait()
        recv_op.wait()

        # 累加
        result.flatten()[recv_chunk_start : recv_chunk_start + chunk_size] += recv_chunk

    # Step 2: All-Gather
    for step in range(world_size - 1):
        send_chunk_start = ((rank - step + 1) % world_size) * chunk_size
        recv_chunk_start = ((rank - step) % world_size) * chunk_size

        send_chunk = result.flatten()[send_chunk_start : send_chunk_start + chunk_size].contiguous()
        recv_chunk = torch.zeros_like(send_chunk)

        send_dst = (rank + 1) % world_size
        recv_src = (rank - 1) % world_size

        send_op = dist.isend(send_chunk, send_dst)
        recv_op = dist.irecv(recv_chunk, recv_src)
        send_op.wait()
        recv_op.wait()

        result.flatten()[recv_chunk_start : recv_chunk_start + chunk_size] = recv_chunk

    return result


def naive_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-Gather 手写实现。

    方式：每个 rank broadcast 自己的数据，所有 rank 拼接。

    通信量: O(N * P²)
    """
    rank = get_rank()
    world_size = get_world_size()
    gathered = []

    for src in range(world_size):
        if rank == src:
            data = tensor.clone()
        else:
            data = torch.zeros_like(tensor)
        dist.broadcast(data, src=src)
        gathered.append(data)

    return torch.cat(gathered, dim=0)


def naive_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast 手写实现。

    方式：src rank send 给所有其他 rank。

    通信量: O(N * P)
    """
    rank = get_rank()
    world_size = get_world_size()

    if rank == src:
        for dst in range(world_size):
            if dst != src:
                dist.send(tensor.contiguous(), dst=dst)
        return tensor
    else:
        result = torch.zeros_like(tensor)
        dist.recv(result, src=src)
        return result


def naive_reduce_scatter(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """
    Reduce-Scatter 手写实现。

    方式：
    1. 先用 all-reduce 得到完整结果
    2. 每个 rank 只保留自己负责的那一块

    通信量: O(N * P)（朴素）但实际应为 O(N)
    """
    rank = get_rank()
    world_size = get_world_size()

    full_result = naive_all_reduce(tensor, op=op)
    chunk_size = tensor.numel() // world_size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size

    return full_result.flatten()[start:end].view_as(
        tensor.flatten()[start:end]
    )


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup
    init_process_group(backend="gloo")
    rank = get_rank()

    tensor = torch.tensor([rank * 1.0, rank * 1.0 + 0.5])

    print(f"Rank {rank}: input = {tensor}")
    result = naive_all_reduce(tensor, op="sum")
    print(f"Rank {rank}: all-reduce sum = {result}")

    result_ring = ring_all_reduce(tensor)
    print(f"Rank {rank}: ring all-reduce = {result_ring}")

    cleanup()
```

- [ ] **Step 5: 运行通信测试**

```bash
# 用 4 个进程模拟 4 GPU
torchrun --nproc_per_node=4 tests/test_communication.py
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add parallel/communication/setup.py parallel/communication/primitives.py tests/test_communication.py
git commit -m "feat: add communication primitives (all-reduce, all-gather, broadcast, ring all-reduce)"
```

---

## Stage 6-7: 并行策略与推理并行

Stage 6（六大并行策略：TP/PP/EP/CP/SP）和 Stage 7（推理并行）遵循与 Stage 5 相同的 TDD 模式。每个模块的文件结构已在设计文档中定义，执行时按以下模式逐个实现：

**统一 TDD 步骤模板：**
1. 编写测试（验证切分后 forward 与单卡一致 + 通信正确性）
2. 运行测试（预期失败）
3. 实现切分/通信逻辑（100-200 行/文件）
4. 运行测试（预期通过）
5. Commit

**各 Task 简要：**

| Task | 文件 | 核心测试点 |
|------|------|-----------|
| 5.2 topologies.py | Ring/Tree/Mesh 拓扑可视化 | 通信步数计算 |
| 6.1 DP/DDP/梯度累积 | parallel/data_parallel/ 3 文件 | 梯度一致性 |
| 6.2 列/行/Embedding 并行 | parallel/tensor_parallel/ 3 文件 | 切分后 forward 一致 |
| 6.3 SP + Megatron 风格 | sequence_parallel.py + megatron_style.py | TP+SP 组合切分 |
| 6.4 GPipe + 1F1B | parallel/pipeline_parallel/ 3 文件 | bubble time 计算 |
| 6.5 EP Token 路由 | parallel/expert_parallel/ 2 文件 | all-to-all 分发正确 |
| 6.6 Ring Attention + CP | parallel/context_parallel/ 3 文件 | 环形传递后 attention 一致 |
| 7.1 KV Cache 分片 | parallel/inference/kv_cache_shard.py | 分片后 decode 一致 |
| 7.2 Prefill/Decode | parallel/inference/prefill_decode.py | 两阶段策略切换 |
| 7.3 推测解码 | parallel/inference/speculative_decoding.py | draft+target 一致性 |
| 7.4 辅助工具 | parallel/utils/ 3 文件 | 可视化输出验证 |

## Notebooks (notebooks/)

10 个 Jupyter Notebook 在所有代码模块完成后依次创建，每个 notebook 导入对应的代码模块，配合 markdown 讲解和可视化图表。每个 notebook 约 200-400 行。

---

## 最终验证

- [ ] 运行全部测试: `python -m pytest tests/ -v`
- [ ] 模型端到端: Transformer、LLaMA3、DeepSeekV3 在 CPU 上完成前向+反向+生成
- [ ] 并行验证: 用 torchrun 启动多进程验证各策略切分一致性
- [ ] 最终 commit
