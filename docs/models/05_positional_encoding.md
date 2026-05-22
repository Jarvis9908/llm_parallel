# 位置编码详解

## 概述

Transformer 的注意力机制是位置无关的——它对输入的排列不变（permutation invariant）。这意味着如果不额外提供位置信息，模型无法区分"猫吃鱼"和"鱼吃猫"。位置编码的作用就是**告诉模型"在哪里"而不是"是什么"**，为每个位置注入唯一的位置信号。

本文将系统介绍三种主流位置编码方法：正弦位置编码、RoPE 和 ALiBi，并进行对比分析。

## 直觉理解

### 为什么需要位置编码

考虑以下两个句子：

- "我 喜欢 你"（I like you）
- "你 喜欢 我"（You like me）

如果去掉位置信息，两个句子的词袋表示完全相同（{我, 喜欢, 你}），但语义截然不同。位置编码就是给每个词打上"位置标签"，让模型知道每个词在句子中的位置。

### 类比：座位号

如果把 token 比作观众，注意力机制让观众之间可以互相交流。但没有位置编码，观众不知道自己坐在第几排第几座——位置编码就是给每个观众分配座位号。

## 数学原理

### 正弦位置编码（Sinusoidal Positional Encoding）

原始 Transformer 使用正弦和余弦函数生成固定位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中 $pos$ 是位置索引，$i$ 是维度索引，$d$ 是嵌入维度。

**设计动机**：

1. **有界性**：$\sin$ 和 $\cos$ 的值域为 $[-1, 1]$，不会主导嵌入值
2. **周期性**：不同维度有不同的频率，低维度变化快（捕获局部位置），高维度变化慢（捕获全局位置）
3. **相对位置可计算**：对于任意固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数

**相对位置的线性关系推导**：

设 $\omega_i = 1/10000^{2i/d}$，则：

$$PE_{(pos, 2i)} = \sin(pos \cdot \omega_i)$$
$$PE_{(pos, 2i+1)} = \cos(pos \cdot \omega_i)$$

利用三角恒等式：

$$\sin((pos+k) \cdot \omega_i) = \sin(pos \cdot \omega_i)\cos(k \cdot \omega_i) + \cos(pos \cdot \omega_i)\sin(k \cdot \omega_i)$$

即 $PE_{pos+k}$ 可以通过 $PE_{pos}$ 的线性变换得到，变换矩阵只依赖偏移 $k$。

**局限性**：

- 绝对位置编码，外推性差——训练时未见过的位置无法合理编码
- 加法融合方式，位置信息可能被嵌入信息淹没

### RoPE：旋转位置编码

RoPE（Rotary Position Embedding）将位置信息编码为旋转变换，应用于 Q 和 K。

#### 旋转矩阵推导

对于二维情况，位置 $m$ 的旋转矩阵为：

$$R_{\Theta,m} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$$

将 $d$ 维向量分成 $d/2$ 个二维子空间，每个子空间独立旋转：

$$R_{\Theta,m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & & \\ \sin m\theta_1 & \cos m\theta_1 & & \\ & & \cos m\theta_2 & -\sin m\theta_2 \\ & & \sin m\theta_2 & \cos m\theta_2 \\ & & & \ddots \end{pmatrix}$$

其中 $\theta_i = 10000^{-2(i-1)/d}$。

#### 相对位置编码性质

将 RoPE 应用于 Q 和 K 后，它们的点积：

$$\langle R_{\Theta,m} q, R_{\Theta,n} k \rangle = q^T R_{\Theta,m}^T R_{\Theta,n} k = q^T R_{\Theta,n-m} k$$

由于 $R_{\Theta,m}^T R_{\Theta,n} = R_{\Theta,n-m}$（旋转矩阵的性质），点积只依赖相对位置 $n - m$。

#### 高效实现

完整的旋转矩阵是稀疏的分块对角矩阵，直接计算效率低。实际实现使用逐元素操作：

```python
def apply_rope(x, positions, theta=10000.0):
    d = x.shape[-1]
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2) / d))
    # 计算角度
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    # 拆分偶数位和奇数位
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    # 应用旋转
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    x_even_new = x_even * cos_angles - x_odd * sin_angles
    x_odd_new = x_even * sin_angles + x_odd * cos_angles
    # 交错合并
    return torch.stack((x_even_new, x_odd_new), dim=-1).flatten(-2)
```

#### 长度外推

RoPE 的一个挑战是长度外推——训练时只见过长度 $L_{train}$ 的序列，推理时需要处理 $L_{test} > L_{train}$ 的序列。

常见解决方案：

1. **位置插值（PI）**：将位置 $[0, L_{test})$ 线性映射到 $[0, L_{train})$
2. **NTK-aware 缩放**：调整 $\theta$ 的基数，使高频分量保持不变
3. **YaRN**：结合 PI 和 NTK 缩放，对不同频率分量使用不同策略

### ALiBi：线性偏置注意力

ALiBi（Attention with Linear Biases）不使用显式的位置编码向量，而是在注意力得分上添加与距离成比例的偏置：

$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T + m \cdot \text{bias})V$$

其中偏置矩阵为：

$$\text{bias}_{ij} = -m \cdot |i - j|$$

$m$ 是每个注意力头独有的斜率，按几何级数设置：

$$m_h = 2^{-\frac{8}{n_{heads}} \cdot (h+1)}, \quad h = 0, 1, ..., n_{heads}-1$$

**ALiBi 的直觉**：距离越远的 token，注意力得分受到越大的惩罚。这符合语言学的局部性——相邻词的关系通常比远距离词更紧密。

**外推性**：ALiBi 天然支持长度外推。由于偏置是相对距离的线性函数，无论序列多长，偏置都可以直接计算。实验表明 ALiBi 可以在短序列上训练、长序列上推理，性能几乎不下降。

**局限性**：

- 线性偏置可能过于简单，无法捕获复杂的位置模式
- 长距离依赖被过度惩罚
- 在大规模模型上的效果不如 RoPE

## 三种方法对比

| 特性 | 正弦位置编码 | RoPE | ALiBi |
|------|------------|------|-------|
| 编码类型 | 绝对位置 | 相对位置 | 相对位置 |
| 应用位置 | 输入嵌入 | Q 和 K | 注意力得分 |
| 外推性 | 差 | 中等（需缩放技巧） | 好 |
| 计算开销 | 低（加法） | 中等（旋转） | 低（加偏置） |
| 实现复杂度 | 简单 | 中等 | 简单 |
| 主流采用 | 原始 Transformer | LLaMA、GPT-NeoX | BLOOM、MPT |
| 长距离建模 | 一般 | 好 | 较差 |
| KV Cache 影响 | 无 | 需存储位置 | 无 |

**选择建议**：

- 新项目首选 RoPE：效果好、生态成熟、长度外推有成熟方案
- 需要极长序列外推：考虑 ALiBi
- 正弦位置编码：仅用于复现原始 Transformer

## 算法流程

### 正弦位置编码

```
输入: 序列长度 n, 嵌入维度 d

1. positions = [0, 1, 2, ..., n-1]           # 位置索引
2. dims = [0, 1, 2, ..., d/2-1]              # 维度索引
3. angles = positions[:, None] / (10000^(2*dims[None, :] / d))
4. PE[:, 0::2] = sin(angles)                  # 偶数维度
5. PE[:, 1::2] = cos(angles)                  # 奇数维度
6. output = token_embeddings + PE              # 加法融合
```

### RoPE

```
输入: Q ∈ R^{n×d}, K ∈ R^{n×d}

1. 将 Q, K 拆分为 d/2 个二维子空间
2. 对每个子空间计算旋转角度: angle_i = pos * θ_i
3. 应用旋转: [q_even, q_odd] → [q_even*cos - q_odd*sin, q_even*sin + q_odd*cos]
4. Q' = 旋转后的 Q, K' = 旋转后的 K
5. attention_scores = Q' @ K'^T / sqrt(d)
```

### ALiBi

```
输入: Q ∈ R^{n×d}, K ∈ R^{n×d}

1. attention_scores = Q @ K^T / sqrt(d)
2. 对每个头 h:
   bias[i][j] = -m_h * |i - j|
3. attention_scores += bias
4. attention_weights = softmax(attention_scores)
```

## 代码实现

本项目的位置编码实现位于 `models/common/positional_encoding.py`：

```python
# 核心类结构示意
class SinusoidalPositionalEncoding:
    """正弦余弦位置编码"""

class RotaryPositionalEncoding:
    """RoPE 旋转位置编码"""

class ALiBiPositionalEncoding:
    """ALiBi 线性偏置位置编码"""
```

关键实现要点：

- 正弦位置编码支持预计算和缓存
- RoPE 支持动态序列长度和 NTK 缩放
- ALiBi 的偏置矩阵在首次使用时计算并缓存

详细代码参见：[`models/common/positional_encoding.py`](../../models/common/positional_encoding.py)

## 实践考量

### 长度外推策略

对于 RoPE 模型的长度外推，推荐方案：

1. **训练时使用较长序列**：最直接，但成本高
2. **位置插值微调**：在长序列上用 PI 微调少量步数
3. **NTK-aware 缩放**：无需微调，调整基数即可
4. **YaRN**：当前最优的无微调方案

### 位置编码与 KV Cache

- 正弦位置编码和 ALiBi 不影响 KV Cache
- RoPE 需要在推理时知道每个缓存 token 的位置，通常存储位置索引或通过序列长度推导

### 批处理中的位置编码

- 不同长度的序列在批处理时需要填充（padding）
- RoPE 的位置应基于实际位置而非填充后的位置
- ALiBi 的偏置应忽略填充位置

## 与其他技术的关系

- **注意力机制**：位置编码为注意力提供位置信息，详见 [注意力机制详解](./01_attention_mechanism.md)
- **Transformer**：位置编码是 Transformer 的必要组件，详见 [Transformer 架构详解](./02_transformer_architecture.md)
- **LLaMA 3**：使用 RoPE，详见 [LLaMA 3 架构详解](./03_llama3_architecture.md)
- **DeepSeek V3**：使用 RoPE + 解耦 K，详见 [DeepSeek V3 详解](./04_deepseek_v3_architecture.md)

## 参考资料

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017. (正弦位置编码)
2. Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing 2024. (RoPE)
3. Press, O., et al. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR 2022. (ALiBi)
4. Chen, S., et al. "Extending Context Window of Large Language Models via Positional Interpolation." arXiv 2023. (PI)
5. bloc97. "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation." 2023.
6. Peng, B., et al. "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024.
