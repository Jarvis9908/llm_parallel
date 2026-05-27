# 并行策略总览

分布式并行是训练和推理大模型的核心技术。本项目实现了六大并行策略，覆盖从训练到推理的完整场景。

## 并行策略分类

```
分布式并行策略
├── 训练并行
│   ├── 数据并行 (Data Parallel)    — 切分数据，每卡完整模型
│   ├── 张量并行 (Tensor Parallel)  — 切分单层权重
│   ├── 流水线并行 (Pipeline Parallel) — 切分模型层
│   └── 专家并行 (Expert Parallel)  — 切分 MoE 专家
├── 序列并行
│   └── 上下文并行 (Context Parallel) — 切分长序列
└── 推理优化
    ├── KV Cache 分片
    ├── Prefill/Decode 分离
    └── Speculative Decoding
```

## 各策略对比

| 策略 | 切分对象 | 通信操作 | 适用场景 | 通信量 |
|------|---------|---------|---------|--------|
| 数据并行 | 数据 (Batch) | AllReduce (梯度) | 通用训练 | O(模型参数) |
| 张量并行 | 模型权重 (层内) | AllReduce / AllGather | 单层大、多机互联快 | O(激活值) |
| 流水线并行 | 模型层 (层间) | Send/Recv | 模型层数多 | O(激活值) |
| 专家并行 | MoE 专家 | All-to-All | MoE 模型 | O(token 数) |
| 上下文并行 | 序列长度 | AllGather / Ring | 超长序列 | O(序列长度) |

## 通信基础

所有并行策略都依赖集合通信原语。学习并行之前，务必先理解：

| 原语 | 含义 | 用途 |
|------|------|------|
| Broadcast | 一个 rank 的数据广播到所有 rank | 模型初始化同步 |
| AllReduce | 所有 rank 的数据做归约，结果广播 | 梯度同步 |
| AllGather | 收集所有 rank 的数据拼接 | 张量并行输出收集 |
| ReduceScatter | 归约后分散到各 rank | 张量并行梯度处理 |
| All-to-All | 全交换 | 专家并行 token 分发 |

详细实现：[parallel/communication/README.md](../parallel/communication/README.md)

## 适用场景速查

- **单机多卡，模型放得下** → 数据并行（最简单）
- **单层计算量太大** → 张量并行
- **模型太深，层数太多** → 流水线并行
- **MoE 模型** → 专家并行
- **超长序列（>128K）** → 上下文并行
- **推理显存不够** → KV Cache 分片 + Speculative Decoding
- **多策略组合** → 常见组合如 DP+TP、DP+TP+PP

## 通信拓扑的影响

并行策略的效率高度依赖硬件互联拓扑：

| 拓扑 | 特点 | 适合的策略 |
|------|------|-----------|
| Ring | 带宽最优，延迟与节点数成正比 | 数据并行 |
| Tree | 延迟最优，带宽有瓶颈 | Broadcast |
| Mesh (NVLink/NVSwitch) | 高带宽低延迟 | 张量并行 |

详细分析：[parallel/communication/README.md](../parallel/communication/README.md)

## 学习建议

建议按以下顺序学习，每一步都建立在前一步的基础上：

1. [通信原语](../parallel/communication/README.md) → [notebook 05](../../notebooks/05_communication_primitives.ipynb)
2. [数据并行](../parallel/data_parallel/README.md) → [notebook 06](../../notebooks/06_data_parallel.ipynb)
3. [张量并行](../parallel/tensor_parallel/README.md) → [notebook 07](../../notebooks/07_tensor_parallel.ipynb)
4. [流水线并行](../parallel/pipeline_parallel/README.md) → [notebook 08](../../notebooks/08_pipeline_parallel.ipynb)
5. [专家 & 上下文并行](../parallel/expert_parallel/README.md) → [notebook 09](../../notebooks/09_expert_and_context_parallel.ipynb)
6. [推理并行](../parallel/inference/README.md) → [notebook 10](../../notebooks/10_inference_parallel.ipynb)
