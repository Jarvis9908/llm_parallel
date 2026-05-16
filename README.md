# LLM Parallel

LLM 架构与分布式并行学习仓库。

## 项目简介

本仓库是一个学习项目，旨在深入理解大语言模型（LLM）的架构设计和分布式并行训练/推理策略。通过手写实现核心组件，从零构建对 LLM 技术栈的系统性认知。

## 学习路线

### 路线一：模型架构

从经典 Transformer 出发，逐步演进到现代 LLM 架构：

1. **Transformer** - 原始 Transformer 架构（Attention is All You Need）
2. **LLaMA 3** - 解码器架构、RMSNorm、SwiGLU、RoPE、GQA 等改进
3. **DeepSeek V3** - MoE（混合专家）、FP8 训练、多 token 预测等前沿技术

### 路线二：分布式并行

涵盖六大并行策略，从数据并行到推理优化：

1. **通信基础** - AllReduce、AllGather、ReduceScatter、Broadcast 等集合通信原语
2. **数据并行** - DP、DDP、FSDP（ZeRO 系列优化）
3. **张量并行** - 1D/2D/2.5D/3D 张量切分策略
4. **流水线并行** - GPipe、PipeDream、1F1B 调度策略
5. **专家并行** - MoE 负载均衡、专家路由、All-to-All 通信
6. **上下文并行** - 长序列切分、Ring Attention、序列并行
7. **推理优化** - KV-Cache、Continuous Batching、Speculative Decoding

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
├── tests/                   # 测试文件
├── docs/                    # 文档与设计规范
├── requirements.txt         # CPU 依赖
└── requirements-cuda.txt    # CUDA 依赖
```

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0（CPU 或 CUDA）
- NumPy >= 1.24.0
- 可选：CUDA Toolkit（如使用 GPU）
