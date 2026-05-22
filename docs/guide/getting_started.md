# 项目入门指南

## 概述

本仓库是一个 **LLM 架构与分布式并行学习项目**，旨在通过手写实现核心组件，从零构建对大语言模型技术栈的系统性认知。我们相信，真正理解一个系统的最好方式是亲手构建它——不是调包，不是套 API，而是逐行写出注意力机制、逐层搭建 Transformer、逐卡配置分布式通信。

本仓库覆盖从模型架构到分布式并行的完整链路：

- **模型架构线**：Attention → Transformer → LLaMA 3 → DeepSeek V3
- **分布式并行线**：通信原语 → 数据并行 → 张量并行 → 流水线并行 → 专家/上下文并行 → 推理优化

---

## 直觉理解

学习 LLM 就像学开车：

1. **了解原理（模型架构）**——先知道发动机怎么转：注意力机制如何工作、Transformer 如何堆叠、现代 LLM 做了哪些改进
2. **学操作（训练推理）**——再学会踩油门和刹车：如何训练一个模型、如何高效推理、如何管理显存
3. **多人协作（分布式并行）**——最后学车队配合：多张卡怎么分工、数据怎么通信、策略怎么选择

如果你跳过第一步直接学分布式，就像不会开车的人学赛车战术——知道什么时候该超车，但方向盘都握不稳。

---

## 项目背景与目标

### 为什么要从零手写 LLM 组件？

| 学习方式 | 优势 | 劣势 |
|---------|------|------|
| 调用 API（如 HuggingFace） | 快速上手、即插即用 | 黑盒思维、遇到问题无从下手 |
| 阅读论文 | 理论扎实、视野开阔 | 论文与实现之间有巨大鸿沟 |
| **手写实现** | **真正理解每个细节** | **耗时较长** |

手写实现的核心价值：

- **理解原理比使用 API 更重要**：当你知道 `nn.MultiheadAttention` 内部做了什么，才能在遇到显存爆炸时知道该改哪个参数
- **调试能力的根基**：分布式训练中 90% 的 bug 来自对底层机制的不理解
- **从使用者到创造者**：只有理解了现有系统的设计决策，才能做出更好的设计决策

### 本仓库的目标

1. 提供可运行的、逐行注释的核心组件实现
2. 建立从单卡到多卡的完整认知链路
3. 让读者具备独立分析和选择并行策略的能力

---

## 学习路径推荐

本仓库的 11 个 Notebook 按照依赖关系组织，推荐以下三条路径：

### 基础路径（必走）

适合刚接触 LLM 和分布式训练的学习者。

```
01 注意力基础 → 02 Transformer详解 → 03 LLaMA3详解 → 05 通信原语 → 06 数据并行
```

| 编号 | Notebook | 核心内容 | 预计时间 |
|------|----------|---------|---------|
| 01 | `01_attention_basics.ipynb` | Scaled Dot-Product Attention、Multi-Head Attention、Causal Mask | 2-3h |
| 02 | `02_transformer_walkthrough.ipynb` | Encoder-Decoder 架构、位置编码、残差连接 | 3-4h |
| 03 | `03_llama3_walkthrough.ipynb` | RMSNorm、SwiGLU、RoPE、GQA 等现代改进 | 3-4h |
| 05 | `05_communication_primitives.ipynb` | AllReduce、AllGather、ReduceScatter、Broadcast | 2-3h |
| 06 | `06_data_parallel.ipynb` | DP、DDP、FSDP/ZeRO | 3-4h |

### 进阶路径

适合已完成基础路径、想深入并行策略的学习者。

```
04 DeepSeek V3详解 → 07 张量并行 → 08 流水线并行
```

| 编号 | Notebook | 核心内容 | 预计时间 |
|------|----------|---------|---------|
| 04 | `04_deepseek_v3_walkthrough.ipynb` | MoE、MLA、FP8 训练、多 Token 预测 | 4-5h |
| 07 | `07_tensor_parallel.ipynb` | 列并行、行并行、Megatron-LM 风格 | 4-5h |
| 08 | `08_pipeline_parallel.ipynb` | GPipe、1F1B、层划分策略 | 3-4h |

### 高级路径

适合想全面掌握分布式训练和推理优化的学习者。

```
09 专家/上下文并行 → 10 推理优化 → 11 端到端训练
```

| 编号 | Notebook | 核心内容 | 预计时间 |
|------|----------|---------|---------|
| 09 | `09_expert_and_context_parallel.ipynb` | MoE 负载均衡、Ring Attention、序列并行 | 4-5h |
| 10 | `10_inference_parallel.ipynb` | KV-Cache 分片、Continuous Batching、投机解码 | 3-4h |
| 11 | `11_end_to_end_training.ipynb` | 综合实战：组合多种并行策略训练完整模型 | 5-6h |

---

## 环境配置详解

### CPU 版本安装

适合没有 GPU 或只想学习模型架构部分的学习者。

```bash
# 1. 克隆仓库
git clone <repo-url>
cd llm_parallel

# 2. 创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate
# Linux/Mac 激活
# source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### CUDA 版本安装

适合有 NVIDIA GPU、需要运行分布式训练代码的学习者。

```bash
# 1. 查看 CUDA 版本
nvidia-smi
# 输出中找到 "CUDA Version: 12.x" 字样

# 2. 根据 CUDA 版本安装 PyTorch（访问 pytorch.org 获取最新命令）
# CUDA 12.4 示例：
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装其他 CUDA 依赖
pip install -r requirements-cuda.txt

# 4. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 常见问题

#### 问题 1：`ModuleNotFoundError: No module named 'numpy'`

```bash
# 原因：未安装依赖或虚拟环境未激活
pip install -r requirements.txt
```

#### 问题 2：`torch.cuda.is_available()` 返回 `False`

排查步骤：

```python
import torch
print(torch.version.cuda)  # 应输出 CUDA 版本号，如 "12.4"
# 如果输出 None，说明安装的是 CPU 版本的 PyTorch
```

```bash
# 重新安装对应 CUDA 版本的 PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

#### 问题 3：CUDA 版本不匹配

```
RuntimeError: CUDA version mismatch: PyTorch was compiled with CUDA 12.1 but the system has CUDA 12.4
```

解决方法：PyTorch 自带 CUDA 运行时，**不需要**系统安装的 CUDA Toolkit 版本与 PyTorch 编译版本完全一致。只需确保驱动版本足够新即可。如果仍有问题，安装与 PyTorch 编译版本匹配的 CUDA Toolkit。

#### 问题 4：Windows 下多卡训练报错

Windows 对 `torch.distributed` 的支持有限，建议使用 WSL2 或 Linux 环境进行多卡训练实验。单卡实验在 Windows 下可正常运行。

---

## 如何使用本仓库

### 阅读顺序

推荐的三步阅读法：

```
第一步：读文档（docs/）→ 建立概念框架
第二步：跑 Notebook（notebooks/）→ 交互式理解
第三步：看代码（models/ + parallel/）→ 深入实现细节
```

### 代码运行方式

#### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_attention.py -v
pytest tests/test_llama3.py -v
pytest tests/test_communication.py -v
```

#### 运行 Python 脚本

```bash
# 单卡运行
python -m parallel.communication.primitives

# 多卡运行（需要 CUDA 环境）
torchrun --nproc_per_node=4 -m parallel.data_parallel.ddp
```

### Notebook 使用

本仓库的所有 Notebook 都可以在以下环境中运行：

- **Jupyter Notebook / JupyterLab**：本地运行
- **Google Colab**：免费 GPU，适合没有本地 GPU 的学习者
- **VS Code Jupyter 插件**：集成开发体验

> **提示**：分布式训练相关的 Notebook（05-11）需要多卡环境。如果只有单卡，可以先阅读文档理解原理，再在 Colab 等环境实践。

---

## 项目结构速览

```
llm_parallel/
├── models/                    # 模型架构实现
│   ├── common/                # 通用组件
│   │   ├── activation.py      # 激活函数（SwiGLU 等）
│   │   ├── attention.py       # 注意力机制
│   │   ├── embeddings.py      # 嵌入层
│   │   ├── feedforward.py     # 前馈网络
│   │   ├── normalization.py   # 归一化层（RMSNorm 等）
│   │   └── positional_encoding.py  # 位置编码（RoPE 等）
│   ├── transformer/           # 原始 Transformer
│   ├── llama3/                # LLaMA 3 架构
│   └── deepseek_v3/           # DeepSeek V3 架构（MoE、MLA）
├── parallel/                  # 分布式并行实现
│   ├── communication/         # 集合通信原语
│   ├── data_parallel/         # 数据并行（DP、DDP、FSDP）
│   ├── tensor_parallel/       # 张量并行（列并行、行并行）
│   ├── pipeline_parallel/     # 流水线并行（GPipe、1F1B）
│   ├── expert_parallel/       # 专家并行
│   ├── context_parallel/      # 上下文并行（Ring Attention）
│   ├── inference/             # 推理优化（KV-Cache、投机解码）
│   └── utils/                 # 并行工具函数
├── notebooks/                 # 交互式学习 Notebook
├── tests/                     # 测试文件
├── docs/                      # 文档与设计规范
│   └── guide/                 # 综合指南文档
├── requirements.txt           # CPU 依赖
└── requirements-cuda.txt      # CUDA 依赖
```

---

## 与其他技术的关系

| 本仓库内容 | 相关工业工具/框架 | 关系说明 |
|-----------|-----------------|---------|
| Transformer 实现 | HuggingFace Transformers | 本仓库侧重教学实现，HF 侧重工业部署 |
| 数据并行 (DDP/FSDP) | PyTorch Distributed | 本仓库手写实现核心逻辑，PyTorch 提供完整封装 |
| 张量并行 | Megatron-LM | Megatron-LM 是张量并行的工业标准实现 |
| 流水线并行 | DeepSpeed PipeDream | DeepSpeed 提供了更完善的流水线调度 |
| MoE 实现 | DeepSeek-V3 / Mixtral | 本仓库实现核心路由与负载均衡逻辑 |
| 通信原语 | NCCL / Gloo | NCCL 是 GPU 通信的后端，本仓库模拟其语义 |

---

## 参考资料

### 核心论文

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文，一切的开始
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Meta 的开源 LLM，定义了现代解码器架构
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MoE + MLA + FP8 的前沿实践

### 分布式训练

- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053) — 张量并行与 3D 并行的经典论文
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — FSDP 的理论基础
- [PipeDream: Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377) — 流水线并行的 1F1B 调度

### 教程与博客

- [PyTorch 分布式训练教程](https://pytorch.org/tutorials/distributed.html)
- [HuggingFace 分布式训练文档](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#distributed-training)
- [Lil'Log: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
