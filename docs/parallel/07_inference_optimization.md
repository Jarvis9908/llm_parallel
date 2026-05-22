# 推理优化详解

## 概述

推理优化关注的是如何用更少的计算资源和时间，生成与原始模型质量相当的输出。与训练阶段不同，推理阶段面临独特的挑战：自回归生成的串行性、KV Cache 的显存压力、以及服务级别的延迟和吞吐量要求。从 KV Cache 管理到 Continuous Batching，从 Speculative Decoding 到量化剪枝，推理优化是一个多层次的系统工程。

## 直觉理解

**推理优化 = 用更少的计算和时间生成相同质量的输出**

想象一家餐厅：
- **朴素推理**：一次只接待一桌客人，做完一桌的菜再做下一桌——效率极低
- **KV Cache**：把客人点过的菜记下来，不用每次重新问——避免重复计算
- **Continuous Batching**：有客人吃完就走，新客人随时入座——提高翻台率
- **Speculative Decoding**：先快速写个草稿，再让大厨审核修改——并行加速
- **量化**：用更少的食材做出差不多味道的菜——降低成本

## 数学原理

### KV Cache 显存管理

#### KV Cache 的显存需求

自回归生成中，每一步需要之前所有位置的 Key 和 Value：

$$\text{KV Cache 大小} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times s \times b$$

其中 $s$ 为序列长度，$b$ 为 batch size。

**示例**（Llama-2 70B, FP16）：
- $n_{\text{layers}} = 80$, $n_{\text{heads}} = 64$, $d_{\text{head}} = 128$
- 单 token KV Cache：$2 \times 80 \times 64 \times 128 \times 2 = 2.5$ MB
- 2048 序列长度：$2.5 \times 2048 = 5$ GB
- Batch size 32：$5 \times 32 = 160$ GB

**关键洞察**：KV Cache 的显存随 batch size 和序列长度线性增长，是推理服务的主要显存瓶颈。

#### PagedAttention

传统 KV Cache 的两个问题：
1. **显存碎片**：预分配连续显存，但实际使用不连续
2. **显存浪费**：预分配最大序列长度，但大部分请求较短

PagedAttention 借鉴操作系统的虚拟内存分页机制：

- 将 KV Cache 分成固定大小的页（page），每页存储若干 token 的 KV
- 维护页表（page table）映射逻辑位置到物理页
- 按需分配页，不需要预分配最大长度
- 不同请求共享未使用的物理页

**显存节省**：
- 传统：预分配 $s_{\max} \times b$ 的连续显存
- PagedAttention：只分配实际使用的页，浪费率 < 4%（一页的浪费）

$$\text{显存浪费率} \leq \frac{\text{page\_size}}{s_{\text{avg}} + \text{page\_size}}$$

### Continuous Batching 调度策略

#### 静态 Batching 的问题

传统静态批处理：
1. 等待 batch 中所有请求完成
2. 短请求完成后等待长请求，浪费算力
3. 新请求必须等当前 batch 全部完成

**浪费分析**：设 batch 中请求长度为 $s_1, s_2, \ldots, s_b$，最大长度为 $s_{\max}$：

$$\text{浪费率} = 1 - \frac{\sum_i s_i}{b \times s_{\max}}$$

#### Continuous Batching

核心思想：请求完成后立即移出 batch，新请求立即加入。

```
时刻 1: [Req A (生成中), Req B (生成中), Req C (生成中), Req D (生成中)]
时刻 2: [Req A (生成中), Req B (完成!),  Req C (生成中), Req D (生成中)]
时刻 3: [Req A (生成中), Req E (新加入),  Req C (生成中), Req D (完成!)]
时刻 4: [Req A (完成!),  Req E (生成中),  Req C (生成中), Req F (新加入)]
```

**调度策略**：
1. **iteration-level scheduling**：每步迭代后检查是否有请求完成
2. **preemption**：当显存不足时，可抢占（swap out）某些请求的 KV Cache
3. **priority scheduling**：根据 SLO 要求优先调度延迟敏感的请求

### Speculative Decoding

#### Draft-Verify 流程

Speculative Decoding 利用一个小模型（draft model）快速生成候选 token，大模型（target model）并行验证：

1. **Draft 阶段**：小模型自回归生成 $K$ 个候选 token：$t_1, t_2, \ldots, t_K$
2. **Verify 阶段**：大模型一次前向传播验证所有候选 token
3. **接受/拒绝**：从第一个 token 开始，按概率决定接受或拒绝
   - 接受：保留该 token，继续检查下一个
   - 拒绝：从该位置重新采样，丢弃后续所有候选

#### 接受概率的数学推导

设大模型对位置 $i$ 的概率分布为 $q(x)$，小模型为 $p(x)$。

**接受规则**：对候选 token $x$，以概率 $\min(1, \frac{q(x)}{p(x)})$ 接受。

**关键性质**：接受-拒绝采样保证最终分布与大模型完全一致（无损加速）。

**期望接受长度**：
$$\mathbb{E}[\text{accepted tokens}] = \sum_{k=1}^{K} P(\text{accept first } k) = \sum_{k=1}^{K} \prod_{i=1}^{k} \alpha_i$$

其中 $\alpha_i = \sum_x \min(p(x), q(x))$ 是位置 $i$ 的接受概率。

当 $p \approx q$ 时，$\alpha \approx 1$，几乎所有候选都被接受。
当 $p$ 和 $q$ 差异大时，$\alpha$ 较低，加速效果有限。

#### 加速比分析

设大模型单步时间为 $T_t$，小模型为 $T_d$，验证时间为 $T_t$（一次前向）。

**朴素自回归**：生成 $K$ 个 token 需要 $K \times T_t$

**Speculative Decoding**：
- Draft：$K \times T_d$
- Verify：$T_t$
- 总时间：$K \times T_d + T_t$

**加速比**：
$$\text{Speedup} = \frac{K \times T_t}{K \times T_d + T_t}$$

当 $T_d \ll T_t$ 且接受率高时，加速比接近 $K$。

### 量化和剪枝对推理的影响

#### 量化

将模型权重从 FP16 量化到低精度：

| 精度 | 每参数比特 | 模型大小（70B） | 质量损失 |
|------|-----------|----------------|---------|
| FP16 | 16 | 140 GB | 基线 |
| INT8 | 8 | 70 GB | 极小 |
| INT4 | 4 | 35 GB | 较小 |
| INT3 | 3 | 26 GB | 中等 |

**量化对推理的影响**：
- 显存减少：线性减少
- 计算加速：INT8/INT4 的矩阵乘法更快（硬件支持）
- 带宽减少：模型加载和 KV Cache 的数据传输更快

**KV Cache 量化**：将 KV Cache 从 FP16 量化到 FP8/INT4，显存减半。

#### 剪枝

移除不重要的权重或注意力头：

- **非结构化剪枝**：将小权重置零，需要稀疏计算硬件支持
- **结构化剪枝**：移除整个注意力头或 FFN 中间维度，无需特殊硬件
- **MoE 天然稀疏**：MoE 模型本身就是一种结构化稀疏

### 推理服务的性能指标

| 指标 | 定义 | 重要性 |
|------|------|--------|
| 吞吐量 (Throughput) | 单位时间生成的 token 数 | 成本效率 |
| 延迟 (Latency) | 单个请求的端到端时间 | 用户体验 |
| 首 token 时间 (TTFT) | 从请求到第一个 token 的时间 | 交互体验 |
| 每 Token 时间 (TPOT) | 生成每个 token 的平均时间 | 流式体验 |
| 并发数 | 同时服务的请求数 | 服务容量 |

**关键权衡**：
- 吞吐量 vs 延迟：增大 batch size 提高吞吐量，但增加延迟
- TTFT vs TPOT：prefill 阶段影响 TTFT，decode 阶段影响 TPOT

## 算法流程

### PagedAttention + Continuous Batching

```
1. 请求到达:
   - 分配新的 page table 条目
   - 加入 waiting queue

2. Prefill 阶段:
   - 从 waiting queue 取请求
   - 执行完整 prompt 的前向传播
   - KV Cache 写入物理页
   - 移入 running queue

3. Decode 阶段 (每步迭代):
   a. 检查 running queue:
      - 已完成的请求: 释放 KV Cache 页, 返回结果
      - 未完成的请求: 继续生成下一个 token
   b. 检查显存:
      - 有空闲页: 从 waiting queue 取新请求加入 prefill
      - 无空闲页: 等待或抢占低优先级请求
   c. 批量执行 decode 步骤
   d. 更新 KV Cache (按需分配新页)
```

### Speculative Decoding 流程

```
1. Draft 阶段:
   for i in range(K):
     t_i = draft_model.generate(prev_tokens)
     draft_tokens.append(t_i)

2. Verify 阶段:
   # 大模型一次前向传播，获取所有位置的概率
   probs = target_model.forward(prompt + draft_tokens)

3. 接受/拒绝:
   accepted = 0
   for i in range(K):
     q_t = probs[i][draft_tokens[i]]    # 大模型概率
     p_t = draft_probs[i][draft_tokens[i]]  # 小模型概率

     if random() < min(1, q_t / p_t):
       accepted += 1     # 接受
     else:
       # 拒绝：从修正分布采样
       adjusted_probs = normalize(max(0, probs[i] - draft_probs[i]))
       new_token = sample(adjusted_probs)
       break

4. 输出: prompt + accepted draft tokens + new_token (如有拒绝)
```

## 代码实现

本项目中的推理优化实现位于 `parallel/inference/` 目录：

| 文件 | 内容 |
|------|------|
| `kv_cache_shard.py` | KV Cache 的分页管理和分片 |
| `prefill_decode.py` | Prefill/Decode 分离和 Continuous Batching |
| `speculative_decoding.py` | Speculative Decoding 的 draft-verify 实现 |

```python
# 示例：使用 Speculative Decoding
from parallel.inference.speculative_decoding import SpeculativeDecoder

# decoder = SpeculativeDecoder(
#     target_model=llama_70b,
#     draft_model=llama_7b,
#     num_speculative_tokens=5,
# )
# output = decoder.generate(prompt, max_tokens=256)
```

详细代码请参考：[`parallel/inference/`](../../parallel/inference/)

## 实践考量

### KV Cache 配置

| 模型大小 | 推荐 GPU | 最大并发序列 | KV Cache 占比 |
|---------|---------|------------|-------------|
| 7B | 1× A100 | ~100 | ~30% |
| 70B | 4× A100 | ~50 | ~40% |
| 70B (INT4) | 1× A100 | ~80 | ~50% |

### Prefill/Decode 分离

将推理分为两个阶段：
- **Prefill**：计算密集，适合大 batch、高算力
- **Decode**：访存密集，适合小 batch、高带宽

**分离部署**：
- Prefill 实例：使用高算力 GPU，大 batch 处理
- Decode 实例：使用高带宽 GPU，低延迟生成
- KV Cache 通过高速网络在两个实例间传递

### Speculative Decoding 的适用场景

| 场景 | 适用性 | 原因 |
|------|--------|------|
| 小 batch、低延迟 | ✅ 非常适合 | 串行瓶颈大，SD 加速明显 |
| 大 batch、高吞吐 | ❌ 不适合 | 大 batch 本身已充分利用算力 |
| Draft 模型质量接近 Target | ✅ 非常适合 | 接受率高，加速比大 |
| Draft 模型质量差 | ❌ 不适合 | 接受率低，反而变慢 |

### 量化选择指南

| 场景 | 推荐精度 | 原因 |
|------|---------|------|
| 追求最高质量 | FP16/BF16 | 无损 |
| 平衡质量和效率 | INT8 | 质量损失极小 |
| 显存受限 | INT4 (GPTQ/AWQ) | 显存减半，质量损失可控 |
| 极端显存受限 | INT3 | 显存最小，质量损失较大 |

### 常见问题

1. **KV Cache OOM**：请求过多导致显存不足
   - 解决：限制最大并发数、使用 PagedAttention、KV Cache 量化
2. **Prefill 延迟高**：长 prompt 的 prefill 耗时长
   - 解决：Chunked Prefill（将长 prompt 分块处理）
3. **Speculative Decoding 反而变慢**：draft 模型太慢或接受率太低
   - 解决：选择更小更快的 draft 模型、减少 $K$

## 与其他技术的关系

| 技术 | 与推理优化的关系 |
|------|----------------|
| 张量并行 | 推理时用 TP 切分大模型到多卡 |
| 流水线并行 | 推理时较少使用（延迟敏感） |
| 专家并行 | MoE 推理时专家按需加载 |
| 数据并行 | 推理时用 DP 提高吞吐量 |
| 量化 | 推理优化的核心手段 |
| Flash Attention | 推理时加速注意力计算 |

## 参考资料

1. **vLLM 论文**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
2. **Speculative Decoding**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023
3. **Continuous Batching**: Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022
4. **GPTQ**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023
5. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023
6. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention", NeurIPS 2022
