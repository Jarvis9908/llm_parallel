# DeepSeek V3 架构详解

## 概述

DeepSeek V3 是深度求索（DeepSeek）于 2024 年底发布的大语言模型，采用了 MoE（Mixture of Experts）稀疏架构和多项创新技术。其核心设计理念是"用稀疏性换效率，用低秩换显存"——通过 MoE 实现参数量大但计算量小的模型，通过 MLA（Multi-head Latent Attention）压缩 KV Cache，通过 FP8 量化降低训练和推理成本。

DeepSeek V3 拥有 671B 总参数，但每个 token 仅激活 37B 参数，实现了参数效率与计算效率的平衡。

## 直觉理解

### MoE 的类比

想象一家大医院：有 100 位专科医生（专家），但每位患者只需要看 2-3 位。医院不需要让所有医生都看每个患者，而是由分诊台（路由器）根据症状将患者分配给最合适的专家。这样医院的总能力很大，但每次看诊的成本很低。

### MLA 的类比

KV Cache 就像快递站暂存包裹。标准做法是为每个客户保存完整的包裹（完整 KV 向量），MLA 则是将包裹压缩成一个小盒子（低秩压缩），取件时再解压还原。虽然需要压缩/解压的额外操作，但存储空间大幅减少。

## 数学原理

### MoE：混合专家系统

#### 路由机制

给定输入 $x$，路由器计算每个专家的得分并选择 Top-K 个：

$$g(x) = \text{Softmax}(x \cdot W_g)$$

$$\text{selected\_experts} = \text{TopK}(g(x), K)$$

$$\text{output} = \sum_{i \in \text{TopK}} g(x)_i \cdot E_i(x)$$

其中 $W_g \in \mathbb{R}^{d \times N}$ 是路由权重，$E_i$ 是第 $i$ 个专家网络，$N$ 是专家总数，$K$ 是每个 token 激活的专家数。

DeepSeek V3 的具体配置：$N = 256$ 个路由专家 + 1 个共享专家，$K = 8$。

#### 负载均衡

MoE 的核心挑战是负载不均——路由器可能倾向于选择少数几个专家（"赢者通吃"），导致其他专家得不到训练。

**辅助损失（Auxiliary Loss）**：

$$L_{aux} = \alpha \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中：
- $f_i = \frac{\text{分配给专家 } i \text{ 的 token 数}}{\text{总 token 数}}$（实际负载比例）
- $P_i = \frac{\sum_{x} \text{softmax}(x \cdot W_g)_i}{\text{总 token 数}}$（平均路由概率）
- $\alpha$ 是辅助损失系数（通常很小，如 0.01）

这个损失鼓励 $f_i$ 和 $P_i$ 都趋向均匀分布 $1/N$。

#### 共享专家

DeepSeek V3 额外引入了一个共享专家（Shared Expert），所有 token 都会经过该专家：

$$\text{output} = \text{SharedExpert}(x) + \sum_{i \in \text{TopK}} g(x)_i \cdot E_i(x)$$

共享专家负责捕获通用知识，路由专家负责捕获专业知识，减少了不同路由专家之间的冗余。

#### 无辅助损失的负载均衡

DeepSeek V3 提出了一种新的负载均衡策略：通过给每个专家设置一个可学习的偏置项（bias），动态调整路由概率：

$$g'(x)_i = g(x)_i + \text{bias}_i$$

偏置项不参与梯度计算，仅用于路由决策。当某专家过载时增大其偏置，使其更难被选中；反之亦然。

### MLA：多头潜在注意力

#### 动机

标准 MHA 的 KV Cache 需要存储每个头的完整 K 和 V 向量：

$$\text{KV Cache} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len$$

对于 DeepSeek V3（61 层、128 头、128 维），长序列时 KV Cache 可达数十 GB。

#### 低秩压缩

MLA 的核心思想：将 KV 向量投影到低维潜在空间，只缓存压缩后的向量。

**压缩**：

$$c_{KV} = W_{DKV} \cdot h \quad \in \mathbb{R}^{d_c}$$

其中 $d_c \ll n_{heads} \times d_{head}$，$W_{DKV} \in \mathbb{R}^{d_c \times d_{model}}$ 是下投影矩阵。

**解压**：

$$K = W_{UK} \cdot c_{KV} \quad \in \mathbb{R}^{n_{heads} \times d_{head}}$$
$$V = W_{UV} \cdot c_{KV} \quad \in \mathbb{R}^{n_{heads} \times d_{head}}$$

其中 $W_{UK}$ 和 $W_{UV}$ 是上投影矩阵。

**关键洞察**：$W_{UK}$ 和 $W_{UV}$ 可以在推理时吸收到 $W_Q$ 和 $W_O$ 中，避免显式解压的计算开销。

#### Q 的压缩

MLA 同样对 Q 进行低秩压缩：

$$c_Q = W_{DQ} \cdot h \quad \in \mathbb{R}^{d_c'}$$
$$Q = W_{UQ} \cdot c_Q \quad \in \mathbb{R}^{n_{heads} \times d_{head}}$$

Q 的压缩不减少 KV Cache，但减少了训练时的激活值显存。

#### 与 RoPE 的兼容

RoPE 需要在 K 上应用位置相关的旋转，这与 MLA 的 KV 吸收不兼容。DeepSeek V3 的解决方案：

- 对 Q 应用 RoPE（在解压后）
- 额外保留一小段解耦的 K（不经过压缩），专门用于 RoPE

$$K = \text{concat}(W_{UK} \cdot c_{KV}, \text{RoPE}(W_{KR} \cdot h))$$

其中 $W_{KR}$ 是一个小的投影矩阵，只产生 $d_r$ 维的解耦 K（$d_r \ll d_{head}$）。

### FP8 训练

#### 量化原理

FP8 使用 8 位浮点数表示，相比 BF16 减少 50% 的显存和通信量。FP8 有两种格式：

- **E4M3**：4 位指数 + 3 位尾数，范围较小但精度较高，用于前向传播
- **E5M2**：5 位指数 + 2 位尾数，范围较大但精度较低，用于反向传播

#### 精度损失分析

FP8 的动态范围约为 $[2^{-9}, 2^4] \approx [0.002, 16]$（E4M3），远小于 BF16 的 $[2^{-127}, 2^{127}]$。因此需要细粒度的缩放：

- **逐张量缩放**：整个张量共用一个缩放因子，粒度太粗
- **逐通道缩放**：每个输出通道一个缩放因子，DeepSeek V3 的选择
- **逐 token 缩放**：每个 token 一个缩放因子

DeepSeek V3 的 FP8 策略：
- 注意力计算使用 FP8
- MoE 的专家计算使用 FP8
- 归一化层和路由器保持高精度
- 使用在线缩放因子调整

### Multi-Token Prediction（MTP）

传统语言模型每次只预测下一个 token，MTP 同时预测多个未来 token：

$$L = \sum_{k=1}^{K} L_k(y_{t+k} | x_{\leq t})$$

DeepSeek V3 使用 2 个 MTP 头（预测 $y_{t+1}$ 和 $y_{t+2}$），每个头共享主模型的表示，额外添加一个浅层 Transformer 层。

MTP 的好处：
- **训练信号更丰富**：每个位置提供多个梯度信号
- **更好的规划能力**：模型需要"向前看"多个 token
- **推理加速**：MTP 头可用于推测解码（Speculative Decoding）

## 算法流程

### DeepSeek V3 单层前向传播

```
输入: x ∈ R^{n×d_model}

1. h = RMSNorm(x)                              # Pre-Norm
2. h = MLA(h) + x                              # 多头潜在注意力 + 残差
3. h' = RMSNorm(h)                             # Pre-Norm
4. shared_out = SharedExpert(h')               # 共享专家
5. router_scores = Softmax(h' @ W_g)           # 路由计算
6. top_experts = TopK(router_scores, K=8)      # 选择专家
7. expert_out = Σ g_i * E_i(h')                # 加权专家输出
8. output = shared_out + expert_out + h         # 残差
```

### MLA 注意力流程

```
输入: h ∈ R^{n×d_model}

1. c_Q = h @ W_DQ                              # Q 压缩
2. c_KV = h @ W_DKV                            # KV 压缩（缓存此向量）
3. Q = c_Q @ W_UQ                              # Q 解压
4. K = concat(c_KV @ W_UK, RoPE(h @ W_KR))    # K 解压 + 解耦 RoPE
5. V = c_KV @ W_UV                             # V 解压
6. output = ScaledDotProductAttention(Q, K, V)
```

## 代码实现

本项目的 DeepSeek V3 实现位于 `models/deepseek_v3/` 目录：

```
models/deepseek_v3/
├── model.py        # DeepSeekV3 模型主体
├── moe.py          # MoE 层实现（路由、专家、负载均衡）
├── mla.py          # MLA 注意力实现
└── config.py       # 模型配置
```

关键实现要点：

- MoE 层支持 Top-K 路由和共享专家
- MLA 实现了 KV 的低秩压缩和解压
- 支持 FP8 量化训练和推理
- MTP 头用于训练辅助和推测解码

详细代码参见：[`models/deepseek_v3/`](../../models/deepseek_v3/)

## 实践考量

### MoE 的训练挑战

1. **负载不均**：需要辅助损失或偏置调整来保持专家均衡
2. **通信开销**：分布式训练时，专家可能分布在不同 GPU 上，All-to-All 通信是瓶颈
3. **训练不稳定**：路由的离散性导致梯度估计方差大

### MLA vs GQA

| 方面 | GQA | MLA |
|------|-----|-----|
| KV Cache 压缩 | 减少头数 | 低秩压缩 |
| 压缩比 | 固定（头数比） | 灵活（压缩维度可调） |
| 信息损失 | 丢失部分头的独立信息 | 低秩近似损失 |
| 与 RoPE 兼容性 | 天然兼容 | 需要解耦 K |
| 实现复杂度 | 简单 | 较复杂 |

### FP8 的适用场景

- 大规模训练（数千 GPU）：通信带宽是瓶颈时，FP8 可减少 50% 通信量
- 推理部署：减少显存占用和延迟
- 不适合：小模型训练、精度敏感的任务

### DeepSeek V3 模型配置

| 配置 | 值 |
|------|-----|
| 总参数量 | 671B |
| 激活参数量 | 37B |
| 层数 | 61 |
| 隐藏维度 | 7168 |
| 注意力头数 | 128 |
| KV 压缩维度 | 512 |
| 路由专家数 | 256 |
| 共享专家数 | 1 |
| Top-K | 8 |
| FFN 隐藏维度 | 18432 |
| 词表大小 | 129280 |

### MoE 的分布式训练策略

DeepSeek V3 使用了创新的分布式训练策略：

1. **EP（Expert Parallelism）**：256 个专家分布在不同 GPU 上
2. **All-to-All 通信**：token 需要发送到对应专家所在的 GPU
3. **通信-计算重叠**：在计算当前专家时，预取下一步需要的 token
4. **辅助损失无关的负载均衡**：通过偏置调整替代辅助损失，避免训练损失被污染

### 推理时的推测解码

DeepSeek V3 的 MTP 头可以用于推测解码加速推理：

1. MTP 头同时预测 $y_{t+1}$ 和 $y_{t+2}$
2. 用 MTP 头快速生成候选 token
3. 用主模型验证候选 token 的正确性
4. 接受正确的 token，拒绝错误的 token 并重新生成

推测解码可以在不损失生成质量的前提下，将推理速度提升 2-3 倍。

## 与其他技术的关系

- **注意力机制**：MLA 是注意力机制的变体，详见 [注意力机制详解](./01_attention_mechanism.md)
- **LLaMA 3**：对比 GQA vs MLA 的 KV Cache 优化策略，详见 [LLaMA 3 架构详解](./03_llama3_architecture.md)
- **归一化层**：DeepSeek V3 使用 RMSNorm，详见 [归一化层详解](./06_normalization.md)
- **激活函数**：DeepSeek V3 使用 SwiGLU，详见 [激活函数详解](./07_activation_functions.md)

## 参考资料

1. DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv 2024.
2. Fedus, W., et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR 2022.
3. Shazeer, N., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.
4. Micikevicius, P., et al. "FP8 Formats for Deep Learning." arXiv 2022.
5. Gloeckle, F., et al. "Better & Faster Large Language Models via Multi-token Prediction." ICML 2024.
