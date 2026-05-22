# 归一化层详解

## 概述

归一化层（Normalization Layer）是深度神经网络训练稳定性的关键组件。其核心作用是**让每层的输入分布稳定**，缓解内部协变量偏移（Internal Covariate Shift），使梯度流动更平稳，从而允许使用更大的学习率、加速收敛并提升训练稳定性。

在 Transformer 架构的演进中，归一化层经历了从 BatchNorm → LayerNorm → RMSNorm 的简化，以及从 Post-Norm → Pre-Norm 的位置调整。这些变化看似微小，却对深层网络的训练稳定性有决定性影响。

## 直觉理解

### 为什么需要归一化

想象一条流水线，每个工位（网络层）处理上一步传来的产品。如果某个工位输出过大或过小，下游工位就需要不断调整自己的工作方式来适应——这就是内部协变量偏移。归一化层就像标准化质检站，确保每个工位的输出都在合理范围内，下游工位可以专注于自己的任务。

### 归一化的本质

归一化的通用公式：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

- 减去均值 $\mu$：中心化
- 除以标准差 $\sigma$：缩放到单位方差
- 乘以 $\gamma$（可学习缩放）：恢复表达能力
- 加上 $\beta$（可学习偏移）：恢复表达能力

不同的归一化方法区别在于**在哪个维度上计算 $\mu$ 和 $\sigma$**。

## 数学原理

### LayerNorm

LayerNorm 在特征维度上计算统计量：

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中 $d$ 是特征维度，$\gamma$ 和 $\beta$ 是可学习参数。

**LayerNorm 的特点**：

- 对每个样本独立归一化，不依赖批内其他样本
- 适用于变长序列（不像 BatchNorm 依赖批大小）
- 在 NLP 任务中表现优于 BatchNorm

**计算开销**：需要计算均值和方差，两次遍历数据。

### RMSNorm

RMSNorm 是 LayerNorm 的简化版本，去掉了均值中心化步骤：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

**简化推导**：

从 LayerNorm 出发：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

假设输入已经近似零均值（$\mu \approx 0$），则：

$$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 \approx \frac{1}{d}\sum_{i=1}^{d} x_i^2 = \text{RMS}^2(x)$$

因此：

$$\text{LayerNorm}(x) \approx \frac{x}{\text{RMS}(x)} \cdot \gamma + \beta$$

RMSNorm 进一步去掉了偏移 $\beta$（实验表明影响不大），得到最终形式。

**RMSNorm 的优势**：

- 计算更快：只需一次遍历计算 RMS，无需计算均值
- 效果相当：实验表明 RMSNorm 与 LayerNorm 在大多数任务上效果相当
- 参数更少：只有 $\gamma$，没有 $\beta$

**计算量对比**：

| 操作 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算均值 | 需要 | 不需要 |
| 计算方差 | 需要 | 不需要 |
| 计算 RMS | 不需要 | 需要 |
| 均值中心化 | 需要 | 不需要 |
| 可学习参数 | $\gamma, \beta$ | $\gamma$ |
| 总计算量 | 较高 | 较低（约减少 10-15%） |

### Pre-Norm vs Post-Norm

**Post-Norm**（原始 Transformer）：

$$\text{output} = \text{Norm}(x + \text{Sublayer}(x))$$

**Pre-Norm**（LLaMA、GPT 等现代模型）：

$$\text{output} = x + \text{Sublayer}(\text{Norm}(x))$$

**训练稳定性分析**：

从梯度传播的角度分析。设 $L$ 层网络，对于 Post-Norm：

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{k=l}^{L-1} \frac{\partial x_{k+1}}{\partial x_k}$$

其中 $x_{k+1} = \text{Norm}(x_k + f_k(x_k))$，梯度需要经过 Norm 层，可能被缩放。

对于 Pre-Norm：

$$x_{k+1} = x_k + f_k(\text{Norm}(x_k))$$

残差路径 $x_k$ 不经过 Norm，梯度可以直接回传：

$$\frac{\partial x_{k+1}}{\partial x_k} = I + \frac{\partial f_k(\text{Norm}(x_k))}{\partial x_k}$$

当 $f_k$ 的梯度较小时，$\frac{\partial x_{k+1}}{\partial x_k} \approx I$，梯度不会消失。

**关键区别**：

| 方面 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 残差路径 | 经过归一化 | 不经过归一化 |
| 梯度传播 | 可能被归一化缩放 | 直接传播 |
| 训练稳定性 | 深层网络不稳定 | 深层网络稳定 |
| 需要学习率预热 | 是 | 否 |
| 最终性能 | 略好（如果训练成功） | 略差但更稳定 |
| 收敛速度 | 较慢 | 较快 |

**实践建议**：对于 12 层以下的模型，Post-Norm 和 Pre-Norm 差异不大；对于更深的模型，Pre-Norm 是更安全的选择。

### DeepNorm

DeepNorm 是微软提出的一种针对极深 Post-Norm Transformer 的训练技巧：

$$\text{output} = \text{Norm}(\alpha \cdot x + \text{Sublayer}(x))$$

其中 $\alpha > 1$ 是一个缩放因子，根据网络深度自动设置：

$$\alpha = (2N)^{1/4}$$

$N$ 是 Transformer 层数。

**原理**：通过放大残差连接，使梯度在深层网络中更容易传播。DeepNorm 证明了在适当设置 $\alpha$ 和学习率的情况下，Post-Norm Transformer 可以训练 1000+ 层。

**DeepNorm 的初始化**：配合 $\alpha$，子层的参数初始化需要缩小 $\beta$ 倍：

- 注意力投影：$W_O$ 缩小 $\beta$ 倍
- FFN 输出：$W_2$ 缩小 $\beta$ 倍
- $\beta = (8N)^{1/4}$

## 算法流程

### LayerNorm 前向传播

```
输入: x ∈ R^{...×d}

1. μ = mean(x, dim=-1, keepdim=True)          # 计算均值
2. σ² = var(x, dim=-1, keepdim=True)           # 计算方差
3. x_norm = (x - μ) / sqrt(σ² + ε)            # 归一化
4. output = x_norm * γ + β                     # 缩放和偏移
```

### RMSNorm 前向传播

```
输入: x ∈ R^{...×d}

1. rms = sqrt(mean(x², dim=-1, keepdim=True) + ε)  # 计算 RMS
2. x_norm = x / rms                                  # 归一化
3. output = x_norm * γ                               # 缩放
```

### Pre-Norm Transformer 层

```
输入: x

1. h = Norm(x)                                 # 先归一化
2. h = Attention(h) + x                        # 注意力 + 残差
3. h' = Norm(h)                                # 先归一化
4. output = FFN(h') + h                         # FFN + 残差
```

## 代码实现

本项目的归一化层实现位于 `models/common/normalization.py`：

```python
# 核心类结构示意
class LayerNorm:
    """标准 LayerNorm"""

class RMSNorm:
    """RMS 归一化"""
```

关键实现要点：

- RMSNorm 使用 `torch.rsqrt` 高效计算 $1/\sqrt{\text{RMS}}$
- 支持半精度（FP16/BF16）训练
- $\epsilon$ 默认值 $10^{-6}$，防止除零

详细代码参见：[`models/common/normalization.py`](../../models/common/normalization.py)

## 实践考量

### 选择 LayerNorm 还是 RMSNorm

- **RMSNorm**：新项目的默认选择，计算更快，效果相当
- **LayerNorm**：需要精确控制输出分布时使用（如某些生成任务）
- 实际差异很小，选择主要取决于计算效率

### 选择 Pre-Norm 还是 Post-Norm

- **Pre-Norm**：深层网络（12 层以上）的默认选择
- **Post-Norm**：浅层网络或追求极致性能（配合 DeepNorm）
- 大多数现代 LLM 使用 Pre-Norm

### 归一化层的数值稳定性

- $\epsilon$ 的选择：通常 $10^{-5}$ 或 $10^{-6}$，过大会影响归一化效果
- 混合精度训练：归一化计算应在 FP32 下进行，避免精度损失
- 梯度累积：归一化统计量应基于当前 micro-batch 计算

### 归一化与学习率的关系

- Pre-Norm 允许使用更大的学习率
- Post-Norm 通常需要学习率预热（warmup）
- RMSNorm 的梯度更稳定，对学习率更鲁棒

### BatchNorm vs LayerNorm vs RMSNorm 对比

| 特性 | BatchNorm | LayerNorm | RMSNorm |
|------|-----------|-----------|---------|
| 归一化维度 | 批维度 | 特征维度 | 特征维度 |
| 依赖批大小 | 是（小批不稳定） | 否 | 否 |
| 适用序列数据 | 不适合 | 适合 | 适合 |
| 训练/推理差异 | 有（需运行均值） | 无 | 无 |
| 计算开销 | 中等 | 较高 | 最低 |
| 可学习参数 | $\gamma, \beta$ | $\gamma, \beta$ | $\gamma$ |
| 主流应用 | CNN | 原始 Transformer | 现代 LLM |

### 归一化层的梯度分析

**LayerNorm 的梯度**：

$$\frac{\partial \text{LN}(x_i)}{\partial x_j} = \frac{1}{\sigma}\left[\delta_{ij} - \frac{1}{d} - \frac{(x_i - \mu)(x_j - \mu)}{d\sigma^2}\right]\gamma_i$$

其中 $\delta_{ij}$ 是 Kronecker delta。梯度包含三个部分：
- 自身项 $\delta_{ij}/\sigma$：直接缩放
- 均值项 $-1/(d\sigma)$：均值中心化的影响
- 方差项 $-(x_i-\mu)(x_j-\mu)/(d\sigma^3)$：方差归一化的影响

**RMSNorm 的梯度**（更简单）：

$$\frac{\partial \text{RMSNorm}(x_i)}{\partial x_j} = \frac{1}{\text{RMS}}\left[\delta_{ij} - \frac{x_i x_j}{d \cdot \text{RMS}^2}\right]\gamma_i$$

RMSNorm 的梯度不包含均值项，计算更简单且数值更稳定。

### 混合精度训练中的归一化

在 FP16/BF16 混合精度训练中，归一化层需要特殊处理：

1. **输入转为 FP32**：归一化的统计量计算需要 FP32 精度
2. **归一化后转回**：输出可以转回低精度
3. **梯度计算**：归一化的反向传播在 FP32 下进行

PyTorch 的 `torch.autocast` 会自动处理这些转换，但自定义实现需要注意。

## 与其他技术的关系

- **Transformer**：每个子层后都有归一化层，详见 [Transformer 架构详解](./02_transformer_architecture.md)
- **LLaMA 3**：使用 Pre-RMSNorm，详见 [LLaMA 3 架构详解](./03_llama3_architecture.md)
- **DeepSeek V3**：使用 Pre-RMSNorm，详见 [DeepSeek V3 详解](./04_deepseek_v3_architecture.md)
- **注意力机制**：注意力输出后接归一化层，详见 [注意力机制详解](./01_attention_mechanism.md)

## 参考资料

1. Ba, J.L., et al. "Layer Normalization." arXiv 2016.
2. Zhang, B., & Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS 2019.
3. Wang, H., et al. "DeepNet: Scaling Transformers to 1,000 Layers." arXiv 2022. (DeepNorm)
4. Ioffe, S., & Szegedy, C. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML 2015.
5. Xiong, R., et al. "On Layer Normalization in the Transformer Architecture." ICML 2020. (Pre-Norm vs Post-Norm 理论分析)
