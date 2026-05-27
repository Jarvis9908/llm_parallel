# 常见问题

## 环境与运行

### Q: 运行 notebook 报 `ModuleNotFoundError: No module named 'models'`

**A:** 需要在项目根目录下启动 Jupyter，或者将项目根目录加入 Python 路径：

```bash
cd llm_parallel
jupyter notebook
```

或在 notebook 开头添加：

```python
import sys
sys.path.insert(0, '..')  # 如果 notebook 在 notebooks/ 目录下
```

### Q: CUDA 相关报错怎么办？

**A:** 本项目默认支持 CPU 运行。如需 GPU：
1. 确认 `nvidia-smi` 能正常输出
2. 安装对应 CUDA 版本的 PyTorch（参考 PyTorch 官网）
3. 验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`

### Q: 测试跑不过？

**A:** 确保依赖已正确安装：`pip install -r requirements.txt`。通信相关测试需要多进程支持，在 Windows 上可能有兼容性问题，建议在 Linux/Mac 上运行。

## 模型架构

### Q: Transformer 的 Encoder 和 Decoder 有什么区别？

**A:** Encoder 处理完整输入序列（双向 Self-Attention），Decoder 逐步生成输出序列（Masked Self-Attention + Cross-Attention）。现代 LLM（如 LLaMA）只用 Decoder 部分，称为 Decoder-only 架构。

### Q: GQA 和 MHA 的区别是什么？为什么 LLaMA 用 GQA？

**A:** MHA（Multi-Head Attention）每个头都有独立的 Q/K/V 权重。GQA（Grouped Query Attention）让多个 Q 头共享一组 K/V 头，减少了 KV Cache 的内存占用和计算量。LLaMA 3 用 GQA 在保持模型质量的同时降低推理成本。

### Q: RoPE 和 Sinusoidal Position Encoding 的区别？

**A:** Sinusoidal PE 将位置信息直接加到 token embedding 上。RoPE（Rotary Positional Embedding）通过对 Q/K 向量做旋转变换来编码位置，具有更好的外推性（能处理训练时没见过的更长序列），是现代 LLM 的标配。

### Q: DeepSeek V3 的 MLA 是怎么减少 KV Cache 的？

**A:** MLA（Multi-head Latent Attention）将 KV 压缩到一个低秩的 latent 空间。传统 MHA 需要缓存每个头的完整 K 和 V，MLA 只需要缓存压缩后的低维 latent 向量，大幅减少了显存占用。

### Q: MoE（混合专家）是怎么工作的？

**A:** MoE 用一个 Router（路由器）根据 token 内容决定将每个 token 发送给哪些 expert（专家网络）。每次只激活少数 expert，模型总参数量很大但每次前向传播只用一小部分，实现了"大模型能力、小模型计算量"。

## 分布式并行

### Q: 数据并行和张量并行的区别？

**A:** 数据并行（DP）将数据切分到不同 GPU，每个 GPU 有完整模型副本，同步梯度。张量并行（TP）将模型权重切分到不同 GPU，每个 GPU 只计算模型的一部分。DP 适合单机多卡，TP 适合单层计算量大的场景。

### Q: AllReduce 的通信量是怎么计算的？

**A:** Ring AllReduce 的通信量为 `2 * (P-1) / P * data_size`，接近 `2 * data_size`，与 GPU 数量 P 基本无关。这是 Ring 拓扑的优势——带宽最优。

### Q: 流水线并行的 Bubble 是什么？

**A:** 在 GPipe 中，由于需要等待所有 micro-batch 完成前向传播才能开始反向传播，部分 GPU 会处于空闲等待状态，这种空闲时间称为 Bubble。1F1B 调度策略通过交替执行前向和反向来减小 Bubble。

### Q: 为什么需要 Sequence Parallel？

**A:** 标准张量并行中，LayerNorm 和 Dropout 的激活值在每个 GPU 上都是完整副本（沿 sequence 维度不切分）。Sequence Parallel 将这些操作的激活值也沿序列维度切分，减少了激活值的显存占用。
