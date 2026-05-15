# LLM 架构与分布式并行学习仓库设计

## 概述

### 目标

构建一个从零学习大语言模型架构与分布式并行原理的代码仓库。两条主线：

1. **模型架构**：Transformer → LLaMA3 → DeepSeek V3，从零手写核心机制
2. **分布式并行**：通信原语 + 六大并行策略，单机多进程模拟

### 约束

- **硬件**：消费级 GPU（8-12GB 显存），模型规模需控制在显存范围内
- **语言**：Python + PyTorch
- **用户背景**：Python/PyTorch 初学者，代码需详细注释和 API 解释
- **实现风格**：核心机制从零手写，不依赖 `nn.MultiheadAttention` 等高层 API
- **并行测试**：单机多进程模拟多 GPU 通信

---

## 仓库结构

```
llm_parallel/
├── models/                          # 模型架构（学习主线1）
│   ├── common/                      # 手写基础组件
│   │   ├── attention.py             #   MHA、MQA、GQA
│   │   ├── embeddings.py            #   Token Embedding
│   │   ├── normalization.py         #   LayerNorm + RMSNorm
│   │   ├── feedforward.py           #   FFN + SwiGLU
│   │   ├── positional_encoding.py   #   Sinusoidal PE + RoPE
│   │   └── activation.py            #   GELU、SiLU 等
│   ├── transformer/                 # 原始 Transformer (Vaswani 2017)
│   │   ├── config.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── model.py
│   ├── llama3/                      # LLaMA3 (Decoder-only, RoPE, RMSNorm, SwiGLU, GQA)
│   │   ├── config.py
│   │   └── model.py
│   └── deepseek_v3/                 # DeepSeek V3 (MoE + MLA)
│       ├── config.py
│       ├── model.py
│       ├── moe.py
│       └── mla.py
│
├── parallel/                        # 分布式并行（学习主线2）
│   ├── communication/               # 通信原语基础
│   │   ├── setup.py                 #   多进程环境搭建
│   │   ├── primitives.py            #   6 种通信原语手写实现
│   │   └── topologies.py            #   通信拓扑分析
│   ├── data_parallel/               # 数据并行
│   │   ├── dp.py                    #   朴素 DP
│   │   ├── ddp.py                   #   DistributedDataParallel
│   │   └── gradient_accumulation.py #   梯度累积
│   ├── tensor_parallel/             # 张量并行 + 序列并行
│   │   ├── column_parallel.py       #   列切分
│   │   ├── row_parallel.py          #   行切分
│   │   ├── embedding_parallel.py    #   Embedding 切分
│   │   ├── sequence_parallel.py     #   序列并行（TP 区域内）
│   │   └── megatron_style.py        #   Megatron 风格完整 TP+SP
│   ├── pipeline_parallel/           # 流水线并行
│   │   ├── layer_partition.py       #   按层切分
│   │   ├── gpiped.py                #   GPipe 调度
│   │   └── f1b1.py                  #   1F1B 调度
│   ├── expert_parallel/             # 专家并行
│   │   ├── expert_partition.py      #   Expert 分布
│   │   └── token_dispatch.py        #   Token all-to-all 路由
│   ├── context_parallel/            # 上下文并行
│   │   ├── ring_attention.py        #   环形注意力
│   │   ├── sequence_partition.py    #   序列维度切分
│   │   └── cp_integration.py        #   CP+TP/EP 组合方案
│   ├── inference/                   # 推理并行
│   │   ├── kv_cache_shard.py        #   KV Cache 分片
│   │   ├── prefill_decode.py        #   Prefill/Decode 策略切换
│   │   └── speculative_decoding.py  #   推测解码
│   └── utils/                       # 辅助工具
│       ├── shard_utils.py           #   权重切分/重组
│       ├── comm_simulator.py        #   通信量模拟器
│       └── visualizer.py            #   可视化工具
│
├── notebooks/                       # 教程 Notebook（10个）
│   ├── 01_attention_basics.ipynb
│   ├── 02_transformer_walkthrough.ipynb
│   ├── 03_llama3_walkthrough.ipynb
│   ├── 04_deepseek_v3_walkthrough.ipynb
│   ├── 05_communication_primitives.ipynb
│   ├── 06_data_parallel.ipynb
│   ├── 07_tensor_parallel.ipynb
│   ├── 08_pipeline_parallel.ipynb
│   ├── 09_expert_and_context_parallel.ipynb
│   └── 10_inference_parallel.ipynb
│
├── tests/                           # 单元测试（7个）
│   ├── test_attention.py
│   ├── test_normalization.py
│   ├── test_transformer.py
│   ├── test_llama3.py
│   ├── test_deepseek_v3.py
│   ├── test_communication.py
│   └── test_parallel.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 学习路径（7 阶段）

### 阶段 1：基础组件 (`models/common/`)

从零手写 Attention（MHA→MQA→GQA）、LayerNorm/RMSNorm、FFN/SwiGLU、Sinusoidal PE/RoPE、激活函数。每个组件一个文件，独立可运行验证。

### 阶段 2：原始 Transformer (`models/transformer/`)

基于 Vaswani et al. (2017)，实现完整的 Encoder-Decoder Transformer。EncoderLayer + Encoder, DecoderLayer + Decoder，完整 forward。

### 阶段 3：LLaMA3 架构 (`models/llama3/`)

Decoder-only 架构，集成 RoPE + RMSNorm + SwiGLU + GQA。实现 TransformerBlock → LLaMA3Model → LLaMA3ForCausalLM（含 LM head）。加入 KV Cache 雏形和文本生成。

### 阶段 4：DeepSeek V3 架构 (`models/deepseek_v3/`)

MoE + Multi-head Latent Attention。Router（top-k gating）、Shared/Routed Experts、auxiliary-loss-free load balancing。MLA 的低秩 KV 压缩与解耦 RoPE。

### 阶段 5：通信基础 (`parallel/communication/`)

多进程环境搭建，6 种通信原语（all-reduce, all-gather, reduce-scatter, broadcast, scatter, reduce）的手写模拟与 PyTorch NCCL 版本对比。Ring/Tree/Mesh 拓扑。

### 阶段 6：六大并行策略

| 并行策略 | 核心内容 |
|----------|----------|
| Data Parallel | DP → DDP → 梯度累积，通信与计算 trade-off |
| Tensor Parallel | 列/行/Embedding 切分 + SP + Megatron 风格完整 TP |
| Pipeline Parallel | 层切分 + GPipe + 1F1B，bubble time 分析 |
| Expert Parallel | Expert 跨卡分布 + Token all-to-all 路由 |
| Context Parallel | Ring Attention + 序列切分 + CP/TP/EP 混合 |
| Sequence Parallel | TP 区域内沿 sequence 维度切分激活值 |

### 阶段 7：推理并行 (`parallel/inference/`)

KV Cache 按 head 维度分片，Prefill 阶段（TP/CP）vs Decode 阶段（DP/EP）策略切换，推测解码。

---

## 技术要点

### 模型规模约束

以 LLaMA3 为例，推荐验证规模：
- dim: 512、n_layers: 8、n_heads: 8、n_kv_heads: 4
- max_seq_len: 2048、vocab_size: 32000
- 估计显存占用（fp32）：~300MB，安全适配 8-12GB 显存

### 并行模拟方式

使用 `torch.distributed` + `spawn` 在单机上启动多个进程，每个进程绑定不同的 `LOCAL_RANK`，通过 NCCL/Gloo backend 进行通信模拟。

### 代码风格

- 核心公式逐行手写，不调用 `nn.MultiheadAttention` 等封装
- 每个文件顶部注明参考论文/来源
- 字符串注释用中文解释关键步骤，PyTorch API 调用加行内说明
- 每个模块文件底部包含 `if __name__ == "__main__":` 快速验证代码块

---

## 文件统计

| 模块 | 文件数 |
|------|--------|
| models/common/ | 6 |
| models/transformer/ | 4 |
| models/llama3/ | 2 |
| models/deepseek_v3/ | 4 |
| parallel/communication/ | 3 |
| parallel/data_parallel/ | 3 |
| parallel/tensor_parallel/ | 5 |
| parallel/pipeline_parallel/ | 3 |
| parallel/expert_parallel/ | 2 |
| parallel/context_parallel/ | 3 |
| parallel/inference/ | 3 |
| parallel/utils/ | 3 |
| notebooks/ | 10 |
| tests/ | 7 |
| 根目录 | 3 |
| **合计** | **~61** |
