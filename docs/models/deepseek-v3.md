# DeepSeek V3 架构详解

> 上一篇：[LLaMA 3](llama3.md) ｜ 返回：[模型架构总览](overview.md)

## 概述

DeepSeek V3 是 DeepSeek 于 2024 年发布的 MoE（Mixture of Experts）大语言模型，总参数 671B，每次推理仅激活 37B 参数。相比 LLaMA 3，它引入了两项重大创新：MLA（Multi-head Latent Attention）低秩压缩注意力和 MoE 混合专家系统。

**前置知识：** [LLaMA 3](llama3.md)（理解 GQA、RoPE、RMSNorm）、[Attention 机制详解](attention.md)
**代码位置：** [`models/deepseek_v3/`](../../models/deepseek_v3/)

## 核心原理

### MLA (Multi-head Latent Attention)

MLA 的核心思想：将 KV 压缩到一个低秩的 latent 空间，大幅减少 KV Cache。

#### 传统 MHA 的 KV Cache 问题

标准 MHA 需要缓存每个 token、每个头的完整 K 和 V 向量：

$$\text{KV Cache} = 2 \times n_{layers} \times n_{heads} \times seq\_len \times d_{head}$$

对于 128K 上下文长度，这个值非常大。

#### MLA 的低秩压缩

MLA 引入压缩-解压两步：

**压缩（Down-Projection）：** 将输入 $x$ 压缩到低维 latent 向量 $c_{KV}$

$$c_{KV} = W_{DKV} \cdot x, \quad c_{KV} \in \mathbb{R}^{d_c}$$

其中 $d_c \ll n_{heads} \times d_{head}$，$W_{DKV} \in \mathbb{R}^{d_{model} \times d_c}$。

**解压（Up-Projection）：** 从 $c_{KV}$ 恢复出 K 和 V

$$k = W_{UK} \cdot c_{KV}, \quad v = W_{UV} \cdot c_{KV}$$

**KV Cache 只需存储 $c_{KV}$：** 大小从 $2 \times n_{heads} \times d_{head}$ 减少到 $d_c$，压缩比为 $\frac{d_c}{2 \times n_{heads} \times d_{head}}$（通常 5-13 倍）。

#### 解耦 RoPE

MLA 将位置信息与内容信息分离：
- **内容部分**：通过低秩压缩（$c_{KV}$）处理
- **位置部分**：通过单独的 $q_{pe}$, $k_{pe}$ 编码 RoPE

$$q = [q_c; q_{pe}], \quad k = [k_c; k_{pe}]$$

这样做是因为 RoPE 的旋转操作会破坏低秩结构。将 RoPE 应用在独立的位置向量上，既保留了位置信息，又不影响压缩效率。

代码对应（`mla.py:38-130`）：

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        # 低秩压缩投影
        self.W_DKV = nn.Linear(dim, kv_lora_rank, bias=False)       # 压缩
        self.W_UK = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)  # 解压 K
        self.W_UV = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)  # 解压 V
        # 解耦 RoPE
        self.W_QR = nn.Linear(dim, n_heads * qk_rope_head_dim, bias=False)  # Q 位置
        self.W_KR = nn.Linear(dim, n_heads * qk_rope_head_dim, bias=False)  # K 位置
```

### MoE (Mixture of Experts)

MoE 将 FFN 层替换为多个 Expert（专家网络），由 Router 动态决定每个 token 发给哪些 Expert。

#### Router（路由器）

Router 是一个线性层 + softmax，输出每个 token 对每个 expert 的路由分数：

$$g = \text{softmax}(x \cdot W_g), \quad g \in \mathbb{R}^{n_{tokens} \times n_{experts}}$$

然后选择 Top-K 个 expert：

$$\text{indices}, \text{scores} = \text{TopK}(g, K=n_{activated})$$

代码对应（`moe.py:12-56`）：

```python
class Router(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(dim, n_routed_experts, bias=False)
    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)     # (n_tokens, n_experts)
        topk_scores, topk_indices = torch.topk(scores, self.n_activated_experts)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # 归一化
        return topk_indices, topk_scores
```

#### Shared Expert vs Routed Expert

- **SharedExpert**：所有 token 都经过的 Expert，捕获通用知识
- **RoutedExpert**：由 Router 动态选择的 Expert，捕获专业领域知识

最终输出 = Shared Expert 输出 + Routed Expert 加权和：

$$y = \text{SharedExpert}(x) + \sum_{i \in \text{selected}} g_i \cdot \text{RoutedExpert}_i(x)$$

代码对应（`moe.py:80-130`）：

```python
class SharedExpert(nn.Module):
    def forward(self, x):
        return self.swiglu_ffn(x)  # 所有 token 都过

class MoELayer(nn.Module):
    def forward(self, x):
        shared_out = self.shared_expert(x)         # 通用知识
        indices, scores = self.router(x)           # 路由选择
        # scatter-add 实现加权路由
        routed_out = self._dispatch_and_compute(x, indices, scores)
        return shared_out + routed_out
```

## 架构图解

### DeepSeek V3 Transformer Block

```
Input x
  ├→ RMSNorm ─→ MLA (低秩压缩 Attention + 解耦 RoPE) ─→ +
  │                                                        ↓
  └────────────────────────────────────────────────────────→ + → x'
                                                                │
  ┌────────────────────────────────────────────────────────→ + ←─┘
  │                                                           │
  └→ RMSNorm ─→ MoE (SharedExpert + Router → RoutedExperts) → +
                                                              ↓
                                                           Output
```

### MLA 压缩-解压流程

```
Input x: (B, S, dim)
  ├→ W_DKV 压缩: (B, S, d_c)        ← 只存这个到 KV Cache!
  │    ├→ W_UK 解压 K: (B, S, n_heads * v_head_dim)
  │    └→ W_UV 解压 V: (B, S, n_heads * v_head_dim)
  ├→ W_Q 投影 Q 内容: (B, S, n_heads * q_head_dim)
  ├→ W_QR 投影 Q 位置: (B, S, n_heads * rope_dim)  → RoPE
  └→ W_KR 投影 K 位置: (B, S, n_heads * rope_dim)  → RoPE

拼接: Q = [Q_content; Q_pe], K = [K_content; K_pe]
Attention(Q, K, V) → Output
```

### MoE 路由流程

```
Input tokens: tok₁, tok₂, ..., tokₙ
  → Router gate: 每个 token 计算 expert 分数
  → Top-K 选择: tok₁ → [exp₂, exp₅], tok₂ → [exp₁, exp₇], ...
  → Dispatch: 将 token 发送到对应 expert
  → Expert 计算: 各 expert 独立处理
  → Gather: 收集结果并加权求和
  → + SharedExpert 输出 → 最终输出
```

## 代码实现分析

### 关键文件清单

| 文件 | 职责 | 关键类 |
|------|------|--------|
| `config.py` | 超参数 | `DeepSeekV3Config` |
| `mla.py:38-274` | MLA 注意力 | `MultiHeadLatentAttention` |
| `moe.py:12-56` | 路由器 | `Router` |
| `moe.py:58-78` | 共享专家 | `SharedExpert` |
| `moe.py:80-130` | 路由专家 | `RoutedExpert` |
| `moe.py:132-211` | MoE 层 | `MoELayer` |
| `model.py:14-65` | 单层 Block | `DeepSeekV3Block` |
| `model.py:68-145` | 基础模型 | `DeepSeekV3Model` |
| `model.py:148-234` | 带 LM Head | `DeepSeekV3ForCausalLM` |

### DeepSeekV3Config 关键参数

```python
@dataclass
class DeepSeekV3Config:
    dim: int = 128                    # 模型维度
    n_heads: int = 4                  # 注意力头数
    kv_lora_rank: int = 32           # KV 低秩压缩维度 d_c
    qk_rope_head_dim: int = 16       # RoPE 位置维度
    v_head_dim: int = 32             # V 头维度
    n_routed_experts: int = 8        # 路由专家总数
    n_shared_experts: int = 1        # 共享专家数
    n_activated_experts: int = 2     # 每次激活的路由专家数
    moe_intermediate_dim: int = 128  # 每个 expert 的 FFN 隐藏层维度
```

**压缩比计算：** 传统 KV Cache = `2 * n_heads * v_head_dim = 2 * 4 * 32 = 256`，MLA Cache = `kv_lora_rank = 32`，压缩比 = 256/32 = **8 倍**。

## 动手实践

→ [notebook 04: DeepSeek V3 walkthrough](../../notebooks/04_deepseek_v3_walkthrough.ipynb)

推荐练习：
1. 对比 MLA 和 MHA 的 KV Cache 大小
2. 修改 `n_activated_experts` 观察路由分布变化
3. 可视化 Router 的路由分数矩阵，理解 token-to-expert 分配

## 延伸阅读

- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
- DeepSeek-AI, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (2024)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2022) — MoE 基础
