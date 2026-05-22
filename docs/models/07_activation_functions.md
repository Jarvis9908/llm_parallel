# 激活函数详解

## 概述

激活函数是神经网络中引入非线性的关键组件。没有激活函数，无论多少层线性变换的叠加仍然等价于一个线性变换——网络将无法学习复杂的模式。激活函数的核心作用是**引入非线性，让网络能学复杂模式**。

在 Transformer 架构的演进中，激活函数经历了从 ReLU → GELU → SwiGLU 的发展。每次演进都带来了更好的梯度特性和训练效果，但也伴随着计算开销的增加。

## 直觉理解

### 为什么需要非线性

假设一个两层网络没有激活函数：

$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'$$

两层线性变换等价于一层——增加层数毫无意义。激活函数打破了这种线性等价性，使每层都能学到新的特征。

### 类比：音乐均衡器

线性变换像音量旋钮——只能整体放大或缩小。激活函数像均衡器——可以选择性地增强某些频段、抑制其他频段，创造出丰富的音色变化。

## 数学原理

### ReLU

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**梯度**：

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**死亡神经元问题**：

当 $x < 0$ 时，ReLU 的梯度为零。如果一个神经元的输入持续为负，它将永远无法更新——这就是"死亡神经元"（Dead Neuron）。

死亡神经元的原因：
- 学习率过大，权重更新使所有输入变为负
- 初始化不当，偏移量过大
- 梯度累积，使权重持续向负方向移动

**ReLU 的优势**：
- 计算极快（只需比较和乘法）
- 正区间梯度恒为 1，缓解梯度消失
- 产生稀疏激活（负值输出为零），有正则化效果

### GELU

GELU（Gaussian Error Linear Unit）的概率解释：GELU 的输出等于输入乘以输入大于零的概率。

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \leq x)$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）：

$$\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-t^2/2} dt$$

**近似计算**：

精确的 GELU 需要计算积分，实际使用两种近似：

1. **tanh 近似**（PyTorch 默认）：

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot (1 + \tanh[\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)])$$

2. **Sigmoid 近似**（SiLU/Swish）：

$$\text{GELU}(x) \approx x \cdot \sigma(1.702 \cdot x)$$

**概率解释的直觉**：

GELU 可以理解为一种"随机门控"：对于每个输入 $x$，以概率 $\Phi(x)$ 保留它，以概率 $1 - \Phi(x)$ 置零。与 ReLU 的硬门控（正数全保留、负数全丢弃）不同，GELU 是软门控——小的负数也有一定概率被保留。

**GELU vs ReLU**：

| 方面 | ReLU | GELU |
|------|------|------|
| 零点处 | 硬截断 | 平滑过渡 |
| 负区间 | 完全为零 | 有微小输出 |
| 梯度 | 零点不连续 | 处处可导 |
| 死亡神经元 | 容易出现 | 不易出现 |
| 计算开销 | 极低 | 较高（需 tanh 或 sigmoid） |

### SwiGLU

SwiGLU（Swish-Gated Linear Unit）是 GLU（Gated Linear Unit）家族的成员，结合了门控机制和 Swish 激活。

**GLU 家族回顾**：

$$\text{GLU}(x, W, V) = \sigma(xW) \odot (xV)$$

其中 $\sigma$ 是 Sigmoid 函数，$\odot$ 是逐元素乘法，$W$ 和 $V$ 是两个投影矩阵。

**SwiGLU 定义**：

$$\text{SwiGLU}(x, W, V, b) = \text{Swish}(xW + b) \odot (xV)$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$，也称为 SiLU。

**在 Transformer FFN 中的应用**：

标准 FFN（使用 GELU）：

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

SwiGLU FFN：

$$\text{FFN}(x) = W_{down} \cdot (\text{SiLU}(W_{gate} x) \odot W_{up} x)$$

**门控机制推导**：

1. $W_{gate} x$：门控路径，决定"放行多少信息"
2. $\text{SiLU}(W_{gate} x)$：门控信号经过 Swish 激活，平滑地控制信息流量
3. $W_{up} x$：信息路径，提供实际传输的内容
4. $\text{SiLU}(W_{gate} x) \odot W_{up} x$：门控调制，信息路径被门控信号选择性放大或抑制
5. $W_{down}$：将维度投影回原始大小

**SwiGLU 的优势**：

- 门控机制提供了比简单激活函数更丰富的非线性
- SiLU 的平滑梯度优于 ReLU 的硬截断
- 实验表明在相同计算预算下，SwiGLU 比 ReLU 和 GELU 效果更好

**参数量分析**：

标准 FFN 有 2 个权重矩阵（$W_1, W_2$），SwiGLU 有 3 个（$W_{gate}, W_{up}, W_{down}$）。为保持总参数量不变，通常将隐藏维度从 $4d$ 调整为 $\frac{8}{3}d$：

$$2 \times d \times 4d = 8d^2 \quad \text{vs} \quad 3 \times d \times \frac{8d}{3} = 8d^2$$

实际实现中，$\frac{8}{3}d$ 会上取整到最近的 256 的倍数。

## 三者对比

| 特性 | ReLU | GELU | SwiGLU |
|------|------|------|--------|
| 公式 | $\max(0, x)$ | $x \cdot \Phi(x)$ | $\text{SiLU}(W_g x) \odot W_u x$ |
| 平滑性 | 零点不连续 | 处处平滑 | 处处平滑 |
| 死亡神经元 | 有 | 无 | 无 |
| 负区间行为 | 完全为零 | 微小输出 | 门控调制 |
| 梯度特性 | 正区间恒1，负区间为0 | 平滑衰减 | 平滑门控梯度 |
| 计算开销 | 最低 | 中等 | 最高（3个投影） |
| FFN 权重矩阵数 | 2 | 2 | 3 |
| 训练效果 | 基线 | 优于 ReLU | 优于 GELU |
| 主流采用 | 早期模型 | BERT、GPT-2 | LLaMA、Mistral、DeepSeek |

**计算开销详细对比**（以 $d = 4096$ 为例）：

| 激活函数 | FFN 隐藏维度 | 权重矩阵数 | 总参数量 | FLOPs |
|---------|------------|-----------|---------|-------|
| ReLU | $4d = 16384$ | 2 | $2 \times 4096 \times 16384 = 134M$ | $2 \times 4096 \times 16384$ |
| GELU | $4d = 16384$ | 2 | $134M$ | $2 \times 4096 \times 16384 + \text{GELU}$ |
| SwiGLU | $\frac{8}{3}d \approx 10923$ → $11008$ | 3 | $3 \times 4096 \times 11008 = 135M$ | $3 \times 4096 \times 11008 + \text{SiLU}$ |

## 算法流程

### ReLU FFN

```
输入: x ∈ R^{n×d}

1. h = x @ W1 + b1                # 上投影 [n, 4d]
2. h = max(0, h)                   # ReLU
3. output = h @ W2 + b2            # 下投影 [n, d]
```

### GELU FFN

```
输入: x ∈ R^{n×d}

1. h = x @ W1 + b1                # 上投影 [n, 4d]
2. h = x * Φ(x)                   # GELU（使用 tanh 近似）
3. output = h @ W2 + b2            # 下投影 [n, d]
```

### SwiGLU FFN

```
输入: x ∈ R^{n×d}

1. gate = x @ W_gate               # 门控路径 [n, d_ffn]
2. up = x @ W_up                   # 信息路径 [n, d_ffn]
3. gate = SiLU(gate)               # 门控激活
4. h = gate ⊙ up                   # 门控调制
5. output = h @ W_down             # 下投影 [n, d]
```

## 代码实现

本项目的激活函数实现位于 `models/common/activation.py`：

```python
# 核心类结构示意
class ReLU:
    """ReLU 激活函数"""

class GELU:
    """GELU 激活函数（支持精确和近似两种模式）"""

class SiLU:
    """SiLU/Swish 激活函数"""

class SwiGLU:
    """SwiGLU 门控激活函数"""
```

关键实现要点：

- GELU 支持精确计算和 tanh 近似两种模式
- SwiGLU 封装了门控投影和信息投影的乘法
- 所有激活函数支持 FP16/BF16 混合精度训练
- 使用 `torch.nn.functional` 中的优化实现

详细代码参见：[`models/common/activation.py`](../../models/common/activation.py)

## 实践考量

### 激活函数的选择

- **新项目默认 SwiGLU**：当前大模型的事实标准，效果最好
- **GELU**：BERT 类模型的标准选择，生态成熟
- **ReLU**：仅用于轻量级模型或计算受限场景

### 混合精度训练中的注意事项

- GELU 的 tanh 近似在 FP16 下可能精度不足，建议在 FP32 下计算
- SwiGLU 的门控乘法在 FP16 下通常稳定
- 使用 PyTorch 的 `torch.autocast` 可以自动处理精度转换

### FFN 隐藏维度的选择

| 激活函数 | 推荐隐藏维度 | 说明 |
|---------|------------|------|
| ReLU/GELU | $4d$ | 原始 Transformer 的选择 |
| SwiGLU | $\lceil \frac{8d}{3} / 256 \rceil \times 256$ | 保持参数量相当 |

### 梯度裁剪与激活函数

- ReLU 的梯度是 0 或 1，天然有裁剪效果
- GELU 和 SwiGLU 的梯度更平滑，但可能出现梯度爆炸
- 配合梯度裁剪（gradient clipping）使用更安全

## 与其他技术的关系

- **Transformer**：FFN 中使用激活函数，详见 [Transformer 架构详解](./02_transformer_architecture.md)
- **LLaMA 3**：使用 SwiGLU，详见 [LLaMA 3 架构详解](./03_llama3_architecture.md)
- **DeepSeek V3**：使用 SwiGLU，详见 [DeepSeek V3 详解](./04_deepseek_v3_architecture.md)
- **归一化层**：归一化层稳定输入分布，激活函数引入非线性，两者配合使用，详见 [归一化层详解](./06_normalization.md)

## 参考资料

1. Nair, V., & Hinton, G. "Rectified Linear Units Improve Restricted Boltzmann Machines." ICML 2010. (ReLU)
2. Hendrycks, D., & Gimpel, K. "Gaussian Error Linear Units (GELUs)." arXiv 2016.
3. Ramachandran, P., et al. "Searching for Activation Functions." arXiv 2017. (Swish/SiLU)
4. Shazeer, N. "GLU Variants Improve Transformer." arXiv 2020. (SwiGLU)
5. Dauphin, Y.N., et al. "Language Modeling with Gated Convolutional Networks." ICML 2017. (GLU)
