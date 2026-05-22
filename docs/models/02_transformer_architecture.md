# Transformer 架构详解

## 概述

Transformer 是 2017 年由 Vaswani 等人在论文《Attention is All You Need》中提出的序列处理架构。它完全基于注意力机制，摒弃了传统 RNN 的循环结构，实现了序列建模的并行化。Transformer 的出现彻底改变了自然语言处理领域，并逐渐扩展到计算机视觉、语音处理等多个领域。

原始 Transformer 采用 Encoder-Decoder 架构用于机器翻译，后续发展出了仅编码器（BERT）、仅解码器（GPT 系列）等变体。当前主流大语言模型（LLaMA、GPT、DeepSeek 等）均采用仅解码器架构。

## 直觉理解

### 用注意力替代循环

传统 RNN 处理序列的方式像"逐字阅读"——读完第一个词才能读第二个，信息沿时间步依次传递。这导致两个根本问题：

1. **无法并行**：第 $t$ 步的计算依赖第 $t-1$ 步的输出
2. **长距离遗忘**：信息经过多步传递后逐渐衰减

Transformer 的思路是"一眼看全文"——用注意力机制直接建立任意两个位置之间的联系，无论它们相距多远。就像读书时不需要从头读到尾才能理解某句话，你可以直接翻到任何一页。

### Encoder-Decoder 的类比

想象翻译工作：翻译官先通读整段原文（Encoder），理解整体含义后，再逐句生成译文（Decoder）。Encoder 看到的是完整的输入，Decoder 则是自回归地逐步生成。

## 数学原理

### Encoder-Decoder 架构数据流

**Encoder**：处理输入序列 $X = (x_1, x_2, ..., x_n)$

$$H^{(l)} = \text{EncoderLayer}(H^{(l-1)})$$

每个 Encoder 层包含：

$$H' = \text{MultiHeadAttention}(H^{(l-1)}, H^{(l-1)}, H^{(l-1)})$$
$$H'' = \text{LayerNorm}(H^{(l-1)} + H')$$
$$H^{(l)} = \text{LayerNorm}(H'' + \text{FFN}(H''))$$

**Decoder**：自回归生成输出序列 $Y = (y_1, y_2, ..., y_m)$

每个 Decoder 层包含三个子层：

1. **掩码自注意力**：只能看到已生成的 token
$$Y' = \text{MaskedMultiHeadAttention}(Y^{(l-1)}, Y^{(l-1)}, Y^{(l-1)})$$

2. **交叉注意力**：关注 Encoder 输出
$$Y'' = \text{MultiHeadAttention}(Y', H^{(L)}, H^{(L)})$$

3. **前馈网络**
$$Y^{(l)} = \text{LayerNorm}(Y'' + \text{FFN}(\text{LayerNorm}(Y' + Y'')))$$

### 残差连接与 LayerNorm

**残差连接**（Residual Connection）：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

残差连接的作用：
- 缓解梯度消失：梯度可以通过短路路径直接回传
- 信息保持：每层只需学习增量变化 $\text{Sublayer}(x)$
- 恒等映射初始化：如果子层输出为零，整体退化为恒等映射

**LayerNorm** 的作用：
- 稳定每层输入的分布
- 缓解内部协变量偏移（Internal Covariate Shift）
- 使训练更稳定，允许使用更大的学习率

### 前馈网络（FFN）

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，通常 $d_{ff} = 4 \times d_{model}$。

FFN 为模型提供了非线性变换能力。注意力机制负责"信息路由"（决定哪些信息流向哪里），FFN 负责"信息加工"（对信息进行非线性变换）。

### 位置前馈网络 vs 逐位置计算

注意：FFN 是逐位置（position-wise）独立应用的，即同一个 FFN 应用于序列的每个位置。不同位置之间不共享的是注意力层，FFN 的参数在所有位置上共享。

## 算法流程

### 完整 Transformer 前向传播

```
输入: 源序列 X_src, 目标序列 X_tgt (训练时)

# Encoder
1. X_src_emb = Embedding(X_src) + PositionalEncoding(X_src)
2. For l = 1 to L:
   a. H = MultiHeadAttention(X_src_emb) + X_src_emb     # 自注意力 + 残差
   b. H = LayerNorm(H)
   c. H = FFN(H) + H                                      # FFN + 残差
   d. H = LayerNorm(H)
   e. X_src_emb = H
3. Encoder输出 = H

# Decoder
4. X_tgt_emb = Embedding(X_tgt) + PositionalEncoding(X_tgt)
5. For l = 1 to L:
   a. Y = MaskedMHA(X_tgt_emb) + X_tgt_emb               # 掩码自注意力
   b. Y = LayerNorm(Y)
   c. Y = CrossAttention(Y, Encoder输出) + Y              # 交叉注意力
   d. Y = LayerNorm(Y)
   e. Y = FFN(Y) + Y
   f. Y = LayerNorm(Y)
   g. X_tgt_emb = Y
6. Logits = Y @ W_vocab                                    # 词表投影
7. Probs = Softmax(Logits)
```

### 仅解码器架构（Decoder-Only）

```
输入: 序列 X

1. X_emb = Embedding(X) + PositionalEncoding(X)
2. For l = 1 to L:
   a. H = CausalMHA(X_emb) + X_emb                        # 因果自注意力
   b. H = LayerNorm(H)
   c. H = FFN(H) + H
   d. H = LayerNorm(H)
   e. X_emb = H
3. Logits = H @ W_vocab
```

## 训练时 vs 推理时的差异

| 方面 | 训练时 | 推理时 |
|------|--------|--------|
| 输入 | 完整目标序列（Teacher Forcing） | 仅已生成的 token |
| 注意力掩码 | 因果掩码（下三角） | 因果掩码（逐步增长） |
| 并行性 | 所有位置并行计算 | 逐步自回归 |
| KV Cache | 不需要（一次计算全部） | 必需（避免重复计算） |
| 计算复杂度 | $O(n^2 \cdot d)$ 一次前向 | $O(n \cdot d)$ 每步，共 $O(n^2 \cdot d)$ |
| 损失计算 | 所有位置同时计算交叉熵 | 仅计算最新 token 的概率 |

**训练时的并行化**：由于 Teacher Forcing，训练时可以将整个目标序列一次性输入，通过因果掩码确保每个位置只能看到之前的 token，从而实现并行计算。这是 Transformer 相比 RNN 的核心优势。

**推理时的自回归**：推理时必须逐 token 生成，每步依赖前一步的输出。KV Cache 是推理优化的关键。

### 三种架构变体对比

| 变体 | 代表模型 | 特点 | 适用场景 |
|------|---------|------|---------|
| Encoder-Only | BERT | 双向注意力，看到完整输入 | 文本理解、分类、NER |
| Decoder-Only | GPT、LLaMA | 单向因果注意力，自回归生成 | 文本生成、对话 |
| Encoder-Decoder | T5、BART | 编码器双向+解码器单向 | 翻译、摘要 |

### 参数量分析

以 $d_{model} = d$，$L$ 层为例：

| 组件 | 参数量 | 占比（典型值） |
|------|--------|--------------|
| 词嵌入 | $V \times d$ | 取决于词表大小 |
| 注意力 Q/K/V/O | $4 \times d^2$ | ~33% |
| FFN | $2 \times d \times d_{ff}$ | ~67% |
| LayerNorm | $4 \times d$（每层2个） | <1% |
| **每层总计** | $4d^2 + 2d \times d_{ff}$ | — |

当 $d_{ff} = 4d$ 时，每层参数量约为 $12d^2$，FFN 占比约 $2/3$。

### Transformer 的局限性

1. **二次复杂度**：注意力计算 $O(n^2)$ 限制了最大序列长度
2. **位置编码外推**：训练长度外的位置编码可能不合理
3. **缺乏层次结构**：所有层使用相同结构，没有显式的多尺度建模
4. **推理延迟**：自回归生成无法并行，KV Cache 增长

### 嵌入层与词表

Transformer 的输入首先经过嵌入层，将离散的 token ID 映射为连续向量：

$$x_i = \text{Embedding}(token_i) + \text{PositionalEncoding}(pos_i)$$

- **词嵌入**：$W_E \in \mathbb{R}^{V \times d}$，$V$ 是词表大小
- **位置编码**：注入位置信息（详见 [位置编码详解](./05_positional_encoding.md)）
- **权重共享**：某些实现中，嵌入矩阵 $W_E$ 与输出投影 $W_{vocab}$ 共享权重

**词表大小的影响**：

| 词表大小 | Embedding 参数 | 序列压缩率 | 适用场景 |
|---------|---------------|-----------|---------|
| 32K | $32K \times d$ | 基线 | 英文为主 |
| 64K | $64K \times d$ | ~1.3x | 多语言 |
| 128K | $128K \times d$ | ~1.5-3x | 多语言、代码 |

### 输出层与损失函数

Transformer 的输出经过线性投影和 softmax 得到词表上的概率分布：

$$P(y_t | y_{<t}) = \text{softmax}(h_L \cdot W_{vocab})$$

训练时使用交叉熵损失：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(y_t | y_{<t})$$

**标签平滑**（Label Smoothing）：将目标分布从 one-hot 调整为：

$$y'_t = (1 - \epsilon) \cdot y_t + \epsilon / V$$

标签平滑可以防止模型过度自信，提升泛化能力。原始 Transformer 使用 $\epsilon = 0.1$。

## 与 RNN/LSTM 的对比

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行性 | 无法并行（顺序依赖） | 训练时完全并行 |
| 长距离依赖 | 梯度消失/爆炸 | 直接连接，$O(1)$ 路径 |
| 计算复杂度 | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| 序列长度限制 | 实际约 100-300 | 理论无限制，实际受显存约束 |
| 位置信息 | 天然有序 | 需要显式位置编码 |
| 参数效率 | 较少参数 | 较多参数 |

**关键洞察**：Transformer 用 $O(n^2)$ 的计算复杂度换来了并行性和长距离建模能力。对于短序列，RNN 可能更高效；对于长序列和大规模训练，Transformer 的并行性使其具有压倒性优势。

## 代码实现

本项目的 Transformer 实现位于 `models/transformer/` 目录：

```
models/transformer/
├── model.py        # 完整 Transformer 模型
├── encoder.py      # Encoder 实现
├── decoder.py      # Decoder 实现
└── config.py       # 模型配置
```

关键实现要点：

- Encoder 和 Decoder 各自包含多个相同结构的层
- 支持 Encoder-Decoder 和 Decoder-Only 两种模式
- 因果掩码通过上三角矩阵实现
- 交叉注意力仅在 Encoder-Decoder 模式下使用

详细代码参见：[`models/transformer/`](../../models/transformer/)

## 与其他技术的关系

- **注意力机制**：Transformer 的核心组件，详见 [注意力机制详解](./01_attention_mechanism.md)
- **位置编码**：Transformer 需要位置编码来感知顺序，详见 [位置编码详解](./05_positional_encoding.md)
- **归一化层**：每个子层后都有 LayerNorm，详见 [归一化层详解](./06_normalization.md)
- **LLaMA 3**：基于 Transformer Decoder-Only 的优化变体，详见 [LLaMA 3 架构详解](./03_llama3_architecture.md)
- **激活函数**：FFN 中的非线性激活，详见 [激活函数详解](./07_activation_functions.md)

## 参考资料

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
3. Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI 2019. (GPT-2)
4. Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020. (GPT-3)
5. Tay, Y., et al. "Efficient Transformers: A Survey." ACM Computing Surveys 2022.
