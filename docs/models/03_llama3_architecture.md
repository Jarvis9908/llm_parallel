# LLaMA 3 架构详解

## 概述

LLaMA 3 是 Meta 于 2024 年发布的大语言模型系列，延续了 LLaMA 系列的 Decoder-Only Transformer 架构，但在多个关键组件上进行了优化。LLaMA 3 的设计哲学是"只保留解码器，优化每个组件"——不追求架构上的激进创新，而是对每个已有组件精雕细琢。

LLaMA 3 提供了 8B 和 70B 两种规模，其中 70B 版本使用了 GQA 来提升推理效率。后续还发布了 405B 版本和 LLaMA 3.1/3.2 系列。

## 直觉理解

如果原始 Transformer 是一辆功能齐全的汽车，LLaMA 3 就是对每个零件精心调校的赛车——发动机（注意力）更高效、悬挂（归一化）更稳定、变速箱（激活函数）更顺畅。架构没有根本性变化，但每个组件都换成了当前最优方案。

## 六项关键改进

### 1. Pre-RMSNorm vs Post-LayerNorm

**原始 Transformer（Post-LayerNorm）**：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**LLaMA 3（Pre-RMSNorm）**：

$$\text{output} = x + \text{Sublayer}(\text{RMSNorm}(x))$$

**关键差异**：

| 方面 | Post-LayerNorm | Pre-RMSNorm |
|------|---------------|-------------|
| 归一化位置 | 子层输出之后 | 子层输入之前 |
| 残差路径 | 经过归一化 | 不经过归一化（干净残差） |
| 梯度传播 | 被归一化缩放 | 直接传播，梯度更稳定 |
| 训练稳定性 | 深层网络不稳定 | 深层网络更稳定 |

Pre-Norm 的核心优势在于**干净的残差路径**：梯度可以通过 $x$ 直接回传，不受归一化层的影响。这对深层网络（30+ 层）的训练至关重要。

RMSNorm 相比 LayerNorm 去掉了均值中心化步骤，只做缩放归一化，计算更高效且效果相当。

### 2. SwiGLU vs ReLU/GELU

**ReLU**：$f(x) = \max(0, x)$

**GELU**：$f(x) = x \cdot \Phi(x)$，其中 $\Phi$ 是标准正态分布的 CDF

**SwiGLU**：

$$\text{SwiGLU}(x, W, V, b) = (\text{SiLU}(xW + b) \otimes (xV))$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$ 是 Sigmoid 线性单元，$\sigma$ 是 Sigmoid 函数。

**LLaMA 3 的 FFN 实现**：

$$\text{FFN}(x) = W_2 \cdot (\text{SiLU}(W_{gate} x) \odot W_{up} x)$$

其中 $W_{gate}$ 和 $W_{up}$ 是两个独立的投影矩阵，实现门控机制。

**SwiGLU 的优势**：

- 门控机制提供了更丰富的非线性表达能力
- SiLU 的平滑梯度避免了 ReLU 的"死亡神经元"问题
- 实验表明在相同参数量下，SwiGLU 比 ReLU 和 GELU 效果更好

**代价**：需要三个权重矩阵（gate、up、down）而非两个，参数量增加约 50%。为保持总参数量不变，通常将隐藏维度从 $4d$ 调整为 $\frac{8}{3}d$（向上取整到最近的 256 的倍数）。

### 3. RoPE 位置编码

RoPE（Rotary Position Embedding）将位置信息编码为旋转操作，应用于 Q 和 K：

$$q_m = R_{\Theta,m} W_q x_m$$
$$k_n = R_{\Theta,n} W_k x_n$$

其中 $R_{\Theta,m}$ 是旋转矩阵：

$$R_{\Theta,m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

$\theta_i = 10000^{-2i/d}$ 是频率参数。

**相对位置编码性质**：

$$q_m^T k_n = (R_{\Theta,m} W_q x_m)^T (R_{\Theta,n} W_k x_n) = x_m^T W_q^T R_{\Theta,n-m} W_k x_n$$

点积结果只依赖相对位置 $n - m$，这是 RoPE 的核心优势。

**高效实现**：旋转矩阵可以分解为逐元素的复数乘法：

$$\text{RoPE}(x, m) = x \odot [\cos(m\theta_1), \cos(m\theta_1), \cos(m\theta_2), \cos(m\theta_2), ...] + \text{rotate\_half}(x) \odot [-\sin(m\theta_1), -\sin(m\theta_1), ...]$$

### 4. GQA 参数效率分析

LLaMA 3 70B 使用 GQA，将 KV 头数从 64 减少到 8：

| 指标 | MHA (64 KV头) | GQA (8 KV头) | 倍率 |
|------|-------------|-------------|------|
| KV Cache 大小 | $2 \times 80 \times 64 \times 128 \times L$ | $2 \times 80 \times 8 \times 128 \times L$ | 1/8 |
| KV 投影参数 | $2 \times 8192 \times 64 \times 128$ | $2 \times 8192 \times 8 \times 128$ | 1/8 |
| 注意力计算量 | 不变 | 不变 | 1x |

GQA 的关键洞察：KV 头数的减少只影响 KV Cache 和 KV 投影的参数量，**不影响注意力的计算量**（因为 K/V 在计算前会被扩展到与 Q 相同的头数）。

### 5. KV Cache 优化

LLaMA 3 在 KV Cache 方面的优化策略：

1. **GQA 减少 KV 头数**：如上所述，直接减少 8 倍
2. **训练时梯度检查点**：不保存中间激活，需要时重新计算
3. **推理时连续批处理**：多个请求共享 GPU，提高吞吐量

### 6. 词表大小与训练效率

LLaMA 3 将词表从 LLaMA 2 的 32K 扩展到 128K：

- **更大的词表** → 更短的序列 → 更少的注意力计算步
- **更好的多语言支持**：128K 词表可以更好地编码中文、日文等非拉丁语系
- **训练效率**：虽然 embedding 层参数增加，但序列缩短带来的注意力计算节省更显著

## 算法流程

### LLaMA 3 单层前向传播

```
输入: x ∈ R^{n×d_model}

1. h = RMSNorm(x)                              # Pre-Norm
2. h = CausalGQA(h) + x                        # GQA + 残差
3. h' = RMSNorm(h)                             # Pre-Norm
4. gate = SiLU(h' @ W_gate)                    # 门控
5. up = h' @ W_up                              # 上投影
6. h'' = gate ⊙ up                             # SwiGLU
7. h''' = h'' @ W_down                         # 下投影
8. output = h''' + h                            # 残差
```

### RoPE 应用流程

```
输入: x ∈ R^{n×d_head}, 位置 positions ∈ R^n

1. 将 x 拆分为偶数位和奇数位: x_even, x_odd
2. 构造旋转对: (x_even, x_odd)
3. 计算角度: θ_i = positions * 10000^{-2i/d_head}
4. 应用旋转:
   x_even' = x_even * cos(θ) - x_odd * sin(θ)
   x_odd'  = x_even * sin(θ) + x_odd * cos(θ)
5. 交错合并: x' = interleave(x_even', x_odd')
```

## 代码实现

本项目的 LLaMA 3 实现位于 `models/llama3/` 目录：

```
models/llama3/
├── model.py        # LLaMA3 模型主体
└── config.py       # 模型配置（层数、头数、隐藏维度等）
```

关键实现要点：

- 使用 RMSNorm 替代 LayerNorm
- FFN 使用 SwiGLU 激活函数（gate + up + down 三个投影）
- RoPE 在注意力计算前应用于 Q 和 K
- GQA 通过 `n_kv_heads` 参数配置

详细代码参见：[`models/llama3/`](../../models/llama3/)

## 实践考量

### 模型配置选择

| 配置 | 8B | 70B | 405B |
|------|-----|------|------|
| 层数 | 32 | 80 | 126 |
| 隐藏维度 | 4096 | 8192 | 16384 |
| 注意力头数 | 32 | 64 | 128 |
| KV 头数 | 8 | 8 | 8 |
| FFN 隐藏维度 | 14336 | 28672 | 53248 |
| 词表大小 | 128256 | 128256 | 128256 |

### 训练稳定性

- Pre-RMSNorm 是深层网络训练稳定的关键
- SwiGLU 的平滑梯度有助于避免训练崩溃
- RoPE 的相对位置性质使模型对序列长度更鲁棒

### 推理优化

- GQA 大幅减少 KV Cache 显存
- RoPE 支持动态缩放，可扩展到更长序列
- 128K 词表缩短了序列长度，间接减少注意力计算

### LLaMA 3 与 LLaMA 2 的关键差异

| 方面 | LLaMA 2 | LLaMA 3 |
|------|---------|---------|
| 词表大小 | 32K | 128K |
| KV 头数（70B） | 64（MHA） | 8（GQA） |
| 训练数据量 | 2T tokens | 15T+ tokens |
| 最大上下文长度 | 4K | 8K → 128K（3.1） |
| 分词器 | SentencePiece | Tiktoken（BPE） |

### 训练数据与词表的影响

LLaMA 3 的 128K 词表基于 Tiktoken（BPE 算法），相比 LLaMA 2 的 32K SentencePiece 词表：

- **编码效率**：128K 词表将英文压缩率提升约 1.5x，中文约 2-3x
- **序列长度**：相同文本编码后更短，直接减少注意力计算量
- **多语言能力**：更大的词表可以更好地编码多语言字符
- **Embedding 参数**：128K × 4096 ≈ 512M 参数（8B 模型），占总参数约 6%

### 上下文长度扩展

LLaMA 3 从 8K 上下文扩展到 128K（LLaMA 3.1）的策略：

1. **RoPE 缩放**：使用 NTK-aware 缩放调整频率基数
2. **渐进训练**：先在短序列上训练，再逐步增加序列长度
3. **注意力优化**：Flash Attention 2 支持长序列的高效计算

### LLaMA 3 的训练细节

LLaMA 3 的训练过程体现了"数据驱动"的哲学：

- **训练数据**：超过 15T tokens，来自公开可用的数据源
- **数据质量**：严格的数据清洗管道，包括去重、质量过滤、安全过滤
- **训练效率**：在 16K GPU 上并行训练，使用 BF16 混合精度
- **分词器**：Tiktoken（BPE），128K 词表，支持多语言和代码

### LLaMA 3 的量化与部署

LLaMA 3 支持多种量化方案以适应不同部署场景：

| 量化方案 | 精度 | 显存（70B） | 质量损失 |
|---------|------|-----------|---------|
| BF16 | 16-bit | ~140 GB | 无 |
| INT8 | 8-bit | ~70 GB | 极小 |
| INT4 (GPTQ) | 4-bit | ~35 GB | 小 |
| INT4 (AWQ) | 4-bit | ~35 GB | 小 |

量化后的模型可以在消费级 GPU 上运行，大幅降低了部署门槛。

## 与其他技术的关系

- **注意力机制**：LLaMA 3 使用 GQA，详见 [注意力机制详解](./01_attention_mechanism.md)
- **Transformer**：LLaMA 3 基于 Decoder-Only Transformer，详见 [Transformer 架构详解](./02_transformer_architecture.md)
- **位置编码**：LLaMA 3 使用 RoPE，详见 [位置编码详解](./05_positional_encoding.md)
- **归一化层**：LLaMA 3 使用 Pre-RMSNorm，详见 [归一化层详解](./06_normalization.md)
- **激活函数**：LLaMA 3 使用 SwiGLU，详见 [激活函数详解](./07_activation_functions.md)

## 参考资料

1. Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv 2023.
2. Touvron, H., et al. "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv 2023.
3. Dubey, A., et al. "The Llama 3 Herd of Models." arXiv 2024.
4. Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing 2024.
5. Shazeer, N. "GLU Variants Improve Transformer." arXiv 2020.
6. Zhang, B., & Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS 2019.
