# 学习文档体系设计

## 背景

本项目是一个 LLM 架构与分布式并行的学习项目，目标是通过手写核心代码帮助读者理解底层原理。当前项目已有 5,500+ 行 Python 代码（含中文注释）和 10 个 Jupyter notebooks，但缺少独立于代码的系统性讲解文档。README 仅 117 行，无法满足"学习项目"的知识传递需求。

## 目标

为项目添加系统性的文字说明和讲解内容，使 Python/PyTorch 初学者能够不读代码就能理解核心原理，然后再通过代码加深理解。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 文档形式 | 混合方案 | docs/ 放原理讲解，模块 README 放导航，notebooks 保持交互 |
| 目标读者 | Python/PyTorch 初学者 | 需要详细的概念解释、公式符号说明、张量形状标注 |
| 文档语言 | 中英混合 | 技术术语保留英文，解释用中文 |
| 实施优先级 | 先框架后细节 | 第一阶段搭结构，第二阶段填内容 |

## 目录结构

### 新增 docs/ 文档

```
docs/
├── getting-started.md          # 入门指南（环境搭建、项目结构、学习路径、知识图谱）
├── faq.md                      # 常见问题与解答
├── models/
│   ├── overview.md             # 模型架构总览（三者演进关系、核心创新对比）
│   ├── attention.md            # Attention 机制详解（MHA/GQA/MQA，跨模块共享）
│   ├── transformer.md          # Transformer 原理详解
│   ├── llama3.md               # LLaMA 3 架构详解
│   └── deepseek-v3.md          # DeepSeek V3 架构详解（MLA + MoE）
└── parallel/
    ├── overview.md             # 并行策略总览（分类体系、适用场景）
    ├── communication.md        # 通信原语详解
    ├── data-parallel.md        # 数据并行详解
    ├── tensor-parallel.md      # 张量并行详解
    ├── pipeline-parallel.md    # 流水线并行详解
    ├── expert-parallel.md      # 专家并行详解
    ├── context-parallel.md     # 上下文并行详解
    └── inference.md            # 推理并行详解
```

### 新增模块 README

每个代码目录一个简短的 README.md：

```
models/common/README.md
models/transformer/README.md
models/llama3/README.md
models/deepseek_v3/README.md
parallel/communication/README.md
parallel/data_parallel/README.md
parallel/tensor_parallel/README.md
parallel/pipeline_parallel/README.md
parallel/expert_parallel/README.md
parallel/context_parallel/README.md
parallel/inference/README.md
parallel/utils/README.md
```

## 内容模板

### docs/ 详细文档模板

每篇文档包含以下部分：

1. **概述** — 一句话定位 + 学习路径中的位置
2. **核心原理** — 直觉解释 → 类比 → 数学公式（每个符号标注含义）→ 代码对应
3. **架构图解** — 整体架构 + 数据流向 + 张量形状变化
4. **代码实现分析** — 关键文件职责 + 重点代码段解读 + 配置参数说明
5. **与其他方案的对比** — 演进关系、取舍分析
6. **动手实践** — 指向对应 notebook + 推荐练习顺序
7. **延伸阅读** — 论文引用 + 相关资源

### 模块 README 模板

每个模块 README 包含：

1. **模块简介** — 一句话说明这个目录实现了什么
2. **文件说明** — 表格列出每个文件的功能和关键类/函数
3. **快速开始** — 最小可运行代码示例
4. **详细文档** — 链接到 docs/ 对应文档
5. **对应 Notebook** — 链接到 notebooks/ 对应文件

### 职责划分

- `docs/` 放"为什么要这么做"和"原理是什么"（侧重理解）
- 模块 README 放"这里有什么、怎么用"（侧重导航）
- 两者通过链接互通，不重复长篇内容

## 学习路径

### 路线一：模型架构（建议 1-2 周）

```
Attention 基础 → Transformer → LLaMA 3 → DeepSeek V3
     ↓                ↓            ↓            ↓
  notebook 01     notebook 02  notebook 03  notebook 04
```

### 路线二：分布式并行（建议 2-3 周，需先完成路线一前两步）

```
通信原语 → 数据并行 → 张量并行 → 流水线并行 → 专家/上下文并行 → 推理并行
    ↓           ↓           ↓           ↓              ↓             ↓
 notebook 05  notebook 06  notebook 07  notebook 08  notebook 09  notebook 10
```

### 知识依赖图谱

在 `getting-started.md` 中用 Mermaid 图展示概念之间的前置依赖关系，让读者一目了然学习顺序和跳转路径。

### 每个主题的学习模式

1. 先读 docs/ 理解原理
2. 再读模块 README 了解代码结构
3. 运行对应 notebook 动手实验
4. 阅读源码深入理解
5. 运行 tests/ 验证理解

## 导航系统

- 每篇 docs/ 文档顶部有"上一篇 / 下一篇"导航
- 每篇 docs/ 文档底部有"相关代码"和"动手实践"链接
- 每个模块 README 有"详细文档"和"对应 notebook"链接
- `getting-started.md` 包含完整学习路径和知识依赖图谱
- `faq.md` 集中解答学习过程中的常见困惑

## 实施分阶段

### 第一阶段（本次实施）

框架搭建，确保读者有清晰的学习路径：

| 文档 | 数量 | 内容 |
|------|------|------|
| `docs/getting-started.md` | 1 | 入门指南 + 学习路径 + 知识图谱（含 Mermaid 图） |
| `docs/models/overview.md` | 1 | 模型架构总览 |
| `docs/parallel/overview.md` | 1 | 并行策略总览 |
| `docs/faq.md` | 1 | 常见问题 |
| 各模块 README | ~12 | 文件说明 + 快速开始 + 文档链接 |

### 第二阶段（后续迭代）

填充详细原理文档：

| 文档 | 数量 |
|------|------|
| `docs/models/attention.md` | 1 |
| `docs/models/transformer.md` | 1 |
| `docs/models/llama3.md` | 1 |
| `docs/models/deepseek-v3.md` | 1 |
| `docs/parallel/communication.md` | 1 |
| `docs/parallel/data-parallel.md` | 1 |
| `docs/parallel/tensor-parallel.md` | 1 |
| `docs/parallel/pipeline-parallel.md` | 1 |
| `docs/parallel/expert-parallel.md` | 1 |
| `docs/parallel/context-parallel.md` | 1 |
| `docs/parallel/inference.md` | 1 |
| Notebook 增强 | 10 |

### 第三阶段（持续完善）

- 根据读者反馈补充 FAQ
- 添加更多图解和可视化
- 补充常见报错和调试技巧

## 写作规范

- 技术术语使用英文（如 Self-Attention、Tensor Parallelism），解释和叙述用中文
- 公式使用 LaTeX 格式，每个符号首次出现时标注含义
- 代码引用使用相对路径 + 行号范围
- 张量形状用 `(batch_size, seq_len, dim)` 格式标注
- 每篇文档控制在合理长度，宁可分节也不要单篇过长
