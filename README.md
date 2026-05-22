# LLM Parallel

LLM 架构与分布式并行学习仓库。

## 项目简介

本仓库是一个学习项目，旨在深入理解大语言模型（LLM）的架构设计和分布式并行训练/推理策略。通过手写实现核心组件，从零构建对 LLM 技术栈的系统性认知。

## 学习路线

### 路线一：模型架构

从经典 Transformer 出发，逐步演进到现代 LLM 架构：

1. **[注意力机制](docs/models/01_attention_mechanism.md)** - MHA/GQA/MQA、KV Cache、Flash Attention
2. **[Transformer](docs/models/02_transformer_architecture.md)** - 原始 Encoder-Decoder 架构
3. **[LLaMA 3](docs/models/03_llama3_architecture.md)** - 解码器架构、RMSNorm、SwiGLU、RoPE、GQA 等改进
4. **[DeepSeek V3](docs/models/04_deepseek_v3_architecture.md)** - MoE（混合专家）、FP8 训练、多 token 预测等前沿技术
5. **[位置编码](docs/models/05_positional_encoding.md)** - 正弦编码、RoPE、ALiBi 对比
6. **[归一化层](docs/models/06_normalization.md)** - LayerNorm、RMSNorm、Pre-Norm vs Post-Norm
7. **[激活函数](docs/models/07_activation_functions.md)** - ReLU、GELU、SwiGLU 对比

### 路线二：分布式并行

涵盖六大并行策略，从数据并行到推理优化：

1. **[通信基础](docs/parallel/01_communication_primitives.md)** - AllReduce、AllGather、ReduceScatter、Broadcast 等集合通信原语
2. **[数据并行](docs/parallel/02_data_parallel.md)** - DP、DDP、FSDP（ZeRO 系列优化）
3. **[张量并行](docs/parallel/03_tensor_parallel.md)** - 1D/2D/2.5D/3D 张量切分策略
4. **[流水线并行](docs/parallel/04_pipeline_parallel.md)** - GPipe、PipeDream、1F1B 调度策略
5. **[专家并行](docs/parallel/05_expert_parallel.md)** - MoE 负载均衡、专家路由、All-to-All 通信
6. **[上下文并行](docs/parallel/06_context_parallel.md)** - 长序列切分、Ring Attention、序列并行
7. **[推理优化](docs/parallel/07_inference_optimization.md)** - KV-Cache、Continuous Batching、Speculative Decoding

## 快速开始

### CPU 版本安装

```bash
# 克隆仓库
git clone <repo-url>
cd llm_parallel

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### CUDA 版本安装

如需 GPU 加速，请按以下步骤安装：

1. **查看本机 CUDA 版本**

```bash
nvidia-smi
```

输出示例中会显示 CUDA 版本（如 `CUDA Version: 12.4`）。

2. **根据 CUDA 版本安装 PyTorch**

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)，选择与你 CUDA 版本匹配的安装命令。例如：

```bash
# CUDA 12.4 示例
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu124

# 然后安装其他依赖
pip install -r requirements-cuda.txt
```

3. **验证 GPU 可用**

```python
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))  # 显示 GPU 名称
```

## 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
llm_parallel/
├── models/                  # 模型架构实现
│   ├── common/              # 通用组件（激活函数、归一化层、位置编码等）
│   ├── transformer/         # 原始 Transformer
│   ├── llama3/              # LLaMA 3 架构
│   └── deepseek_v3/         # DeepSeek V3 架构
├── parallel/                # 分布式并行实现
│   ├── communication/       # 集合通信原语
│   ├── data_parallel/       # 数据并行
│   ├── tensor_parallel/     # 张量并行
│   ├── pipeline_parallel/   # 流水线并行
│   ├── expert_parallel/     # 专家并行
│   ├── context_parallel/    # 上下文并行
│   ├── inference/           # 推理优化
│   └── utils/               # 并行工具函数
├── notebooks/               # Jupyter Notebook 教程（含可视化与练习题）
├── tests/                   # 测试文件
├── docs/                    # 文档与讲解
│   ├── guide/               # 综合指南（入门、并行策略选择、调试、硬件基础）
│   ├── models/              # 模型架构详解（注意力、Transformer、LLaMA3、DeepSeek V3 等）
│   ├── parallel/            # 分布式并行详解（通信原语、数据/张量/流水线/专家/上下文并行、推理优化）
│   └── superpowers/         # 设计规范与实施计划
├── requirements.txt         # CPU 依赖
└── requirements-cuda.txt    # CUDA 依赖
```

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0（CPU 或 CUDA）
- NumPy >= 1.24.0
- 可选：CUDA Toolkit（如使用 GPU）

## 文档指南

- **[项目入门](docs/guide/getting_started.md)** — 学习路径推荐、环境配置、使用方法
- **[并行策略选择](docs/guide/parallel_strategy_guide.md)** — 六大并行策略对比、决策树、实际案例
- **[分布式训练调试](docs/guide/debugging_guide.md)** — 常见错误排查、NCCL/NaN/OOM 诊断
- **[硬件基础知识](docs/guide/hardware_basics.md)** — GPU 架构、NVLink vs PCIe、显存层次
