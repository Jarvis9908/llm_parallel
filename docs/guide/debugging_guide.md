# 分布式训练调试手册

## 概述

分布式训练调试比单卡训练复杂得多。单卡训练的 bug 通常是确定性的——同样的输入产生同样的错误。而分布式训练中，bug 可能只在特定 GPU 组合、特定通信时序下才会出现，且错误信息往往指向通信库（如 NCCL）而非真正的根因。

本手册将帮助你系统性地排查和解决分布式训练中的常见问题。

---

## 直觉理解

调试分布式训练像侦探破案：

1. **收集线索（日志）**——仔细阅读错误信息，不要被表面现象迷惑
2. **缩小范围（隔离）**——将多卡问题简化为单卡问题，将多节点简化为单节点
3. **验证假设（单卡对比）**——用单卡结果作为 ground truth，逐步引入分布式组件

> **黄金法则**：如果单卡跑不通，多卡一定也跑不通。先确保单卡正确，再调试分布式。

---

## 常见错误分类

根据实践经验，分布式训练错误的大致分布：

| 错误类型 | 占比 | 典型表现 | 难度 |
|---------|------|---------|------|
| NCCL 通信错误 | 35% | timeout、connection refused | ★★★☆☆ |
| 显存不足 (OOM) | 25% | CUDA out of memory | ★★☆☆☆ |
| 梯度 NaN | 20% | loss 变为 nan、梯度全零 | ★★★★☆ |
| 死锁 | 10% | 训练挂起、无输出 | ★★★★★ |
| 数值不一致 | 10% | 多卡结果与单卡不同 | ★★★☆☆ |

---

## NCCL 错误排查

NCCL 是 NVIDIA 的 GPU 集合通信库，是 PyTorch 分布式训练的默认后端。NCCL 错误通常是**最常见也最令人困惑**的，因为错误信息往往不直接指向根因。

### 常见错误 1：Timeout

```
RuntimeError: NCCL error in: /path/to/nccl.cpp:XXX
Last error: Net: Connection timed out
```

**可能原因与排查步骤**：

| 原因 | 排查方法 | 解决方案 |
|------|---------|---------|
| 网络不通 | `ping <other_node_ip>` | 检查网络配置、防火墙 |
| NCCL_SOCKET_IFNAME 配置错误 | `ifconfig` 查看网卡名 | 设置正确的网卡名 |
| 防火墙阻断 | `telnet <ip> <port>` | 开放 NCCL 使用的端口 |
| GPU 间负载不均导致同步超时 | 检查各 GPU 计算时间 | 均衡各 GPU 工作量 |

```bash
# 设置 NCCL 使用的网络接口
export NCCL_SOCKET_IFNAME=eth0  # 替换为实际网卡名

# 增加超时时间（默认 30 分钟）
export NCCL_COMM_BLOCKING=1
export NCCL_MIN_NCHANNELS=1
```

### 常见错误 2：Connection Refused

```
RuntimeError: NCCL error in: unhandled system error
Last error: Net: Connection refused
```

**可能原因与排查步骤**：

| 原因 | 排查方法 | 解决方案 |
|------|---------|---------|
| MASTER_ADDR/PORT 配置错误 | 检查环境变量 | 设置正确的主节点地址和端口 |
| 主节点进程未启动 | 检查 rank 0 进程是否存活 | 确保所有进程同时启动 |
| 端口被占用 | `netstat -tlnp \| grep <port>` | 更换端口或释放占用 |

```bash
# 正确设置主节点信息
export MASTER_ADDR=192.168.1.100  # rank 0 所在节点 IP
export MASTER_PORT=29500           # 未被占用的端口
```

### NCCL 调试方法

```bash
# 启用 NCCL 详细日志
export NCCL_DEBUG=INFO

# 启用所有子系统日志
export NCCL_DEBUG_SUBSYS=ALL

# 将日志写入文件
export NCCL_DEBUG_FILE=/tmp/nccl_log_rank%h_%p.txt

# 仅记录 WARN 级别（减少日志量）
export NCCL_DEBUG=WARN
```

> **提示**：`NCCL_DEBUG=INFO` 会产生大量日志，建议先在少量 GPU 上测试，确认配置正确后再扩大规模。

---

## 数值问题排查

### 梯度 NaN

**症状**：loss 突然变为 `nan`，或梯度检查发现 NaN 值。

**常见原因与排查**：

| 原因 | 排查方法 | 解决方案 |
|------|---------|---------|
| 学习率过大 | 检查 loss 曲线是否先升后爆 | 降低学习率（通常 10× 降幅） |
| 数据异常 | 检查输入是否包含 NaN/Inf | 数据预处理中过滤异常值 |
| 除零操作 | 检查分母是否可能为零 | 添加 epsilon（如 `x / (y + 1e-8)`） |
| FP16 溢出 | 检查中间值是否超出 FP16 范围 | 使用 BF16 或 FP32 累积 |
| 梯度累积溢出 | 检查累积步数是否过多 | 减少累积步数或使用梯度裁剪 |

**调试代码**：

```python
# 方法 1：使用 torch.autograd.detect_anomaly()
with torch.autograd.detect_anomaly():
    loss = model(inputs)
    loss.backward()

# 方法 2：梯度 hook 检查 NaN
def check_nan_gradient(grad, name):
    if torch.isnan(grad).any():
        print(f"NaN gradient detected in: {name}")
    return grad

for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(lambda grad, n=name: check_nan_gradient(grad, n))

# 方法 3：训练循环中检查
for batch in dataloader:
    loss = model(batch)
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"NaN/Inf loss detected! Input: {batch}")
        break
    loss.backward()

    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in: {name}")
            break
```

### Loss 爆炸

**症状**：loss 突然从正常值飙升至极大值，然后可能变为 NaN。

**排查步骤**：

1. **检查学习率**：是否使用了适合模型规模的 lr？大模型通常需要更小的 lr
2. **检查梯度裁剪**：是否启用了梯度裁剪？推荐 `max_norm=1.0`
3. **检查数据预处理**：输入是否已正确归一化？标签是否正确？
4. **检查 warmup**：是否使用了学习率 warmup？

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 学习率 warmup（线性）
from torch.optim.lr_scheduler import LambdaLR

def warmup_fn(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
```

---

## 显存问题排查

### OOM 分析

显存占用的四大组成部分：

```
┌──────────────────────────────────────┐
│            GPU 显存 (80 GB)           │
│  ┌────────────────────────────────┐  │
│  │     激活值 (Activations)       │  │  ← 与序列长度和批量成正比
│  │     可变，可能占 50%+          │  │
│  ├────────────────────────────────┤  │
│  │     优化器状态 (Adam)          │  │  ← 4× 参数量 (FP32主拷贝+动量+方差)
│  ├────────────────────────────────┤  │
│  │     梯度 (Gradients)           │  │  ← 2× 参数量 (FP16)
│  ├────────────────────────────────┤  │
│  │     模型参数 (Parameters)      │  │  ← 2× 参数量 (FP16)
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

**显存估算公式**：

$$\text{显存} \approx 2\Phi + 2\Phi + 4\Phi + A = 8\Phi + A$$

其中 $\Phi$ 是参数量，$A$ 是激活值显存。

**激活值显存估算**：

$$A \approx \text{batch\_size} \times \text{seq\_len} \times \text{hidden\_dim} \times \text{num\_layers} \times K$$

$K$ 取决于是否使用梯度检查点（约 1-10 不等）。

### 优化技巧

#### 1. 梯度检查点 (Gradient Checkpointing)

```python
# 启用梯度检查点，用计算换显存
model.gradient_checkpointing_enable()

# 自定义检查点策略
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x)

    def _forward(self, x):
        # 实际计算逻辑
        return result
```

**效果**：激活值显存从 $O(n)$ 降至 $O(\sqrt{n})$，代价是额外 33% 的前向计算。

#### 2. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.bfloat16):  # BF16 比 FP16 更稳定
    loss = model(inputs)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**效果**：显存减少约 50%，计算速度提升 2-3×。

#### 3. ZeRO 优化 (FSDP)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# ZeRO-1: 分片优化器状态
# ZeRO-2: 分片优化器状态 + 梯度
# ZeRO-3: 分片优化器状态 + 梯度 + 参数
model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

| ZeRO 阶段 | 分片内容 | 显存节省 | 通信量增加 |
|-----------|---------|---------|-----------|
| ZeRO-1 | 优化器状态 | ~4× | 1× |
| ZeRO-2 | + 梯度 | ~8× | 1× |
| ZeRO-3 | + 参数 | ~N× (N=GPU数) | 1.5× |

#### 4. 激活重计算

```python
# Flash Attention 自动处理激活重计算
# 使用 torch.compile 或 Flash Attention 2
from torch.nn.functional import scaled_dot_product_attention

# PyTorch 2.0+ 自动选择 Flash Attention
output = scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

---

## 死锁排查

死锁是最难调试的问题之一，表现为训练挂起、无输出、无报错。

### 常见原因

| 原因 | 症状 | 解决方案 |
|------|------|---------|
| 集合通信不匹配 | 所有进程挂起 | 确保所有进程调用相同的集合通信 |
| 条件分支中的通信 | 部分进程挂起 | 通信操作不能在条件分支内 |
| barrier 不匹配 | 进程在 barrier 处挂起 | 检查 barrier 调用是否对称 |
| GPU 间负载不均 | 某些 GPU 等待其他 GPU | 均衡各 GPU 计算量 |

### 调试方法

```python
# 方法 1：添加超时检测
import torch.distributed as dist

dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=300))

# 方法 2：在关键点添加 print（需要 flush）
print(f"[Rank {dist.get_rank()}] Before all_reduce", flush=True)
dist.all_reduce(tensor)
print(f"[Rank {dist.get_rank()}] After all_reduce", flush=True)

# 方法 3：使用 signal handler 检测挂起
import signal, traceback

def handler(signum, frame):
    print(f"Timeout on rank {dist.get_rank()}")
    traceback.print_stack(frame)
    exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(60)  # 60秒超时
```

### 避免死锁的编码原则

```python
# ❌ 错误：条件分支中的通信
if dist.get_rank() == 0:
    dist.all_reduce(tensor)  # 只有 rank 0 调用，死锁！

# ✅ 正确：所有进程都调用
dist.all_reduce(tensor)

# ❌ 错误：不同 rank 调用不同操作
if dist.get_rank() < 2:
    dist.all_reduce(tensor)
else:
    dist.broadcast(tensor, src=0)  # 操作不匹配，死锁！

# ✅ 正确：所有进程调用相同操作
dist.all_reduce(tensor)
```

---

## 数值不一致排查

分布式训练中，多卡结果与单卡不完全一致是正常的（浮点运算顺序不同），但差异应在合理范围内。

### 可接受的差异

| 精度 | 单卡 vs 多卡差异 | 判断标准 |
|------|-----------------|---------|
| FP32 | < 1e-6 | 几乎完全一致 |
| FP16 | < 1e-3 | 小幅差异正常 |
| BF16 | < 1e-2 | 差异较大但正常 |

### 异常差异的排查

```python
# 检查各 rank 的梯度是否一致
def check_gradient_consistency(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"[Rank {dist.get_rank()}] {name}: grad_norm={grad_norm}")

            # 收集所有 rank 的梯度范数
            grad_norms = [torch.zeros(1) for _ in range(dist.get_world_size())]
            dist.all_gather(grad_norms, torch.tensor([grad_norm]))

            if dist.get_rank() == 0:
                max_diff = max(grad_norms) - min(grad_norms)
                if max_diff > 1e-3:
                    print(f"WARNING: Large gradient difference in {name}: {max_diff}")
```

---

## 调试工具清单

| 工具 | 用途 | 使用方式 |
|------|------|---------|
| `torch.distributed.launch` | 启动分布式训练 | `torchrun --nproc_per_node=N train.py` |
| `NCCL_DEBUG=INFO` | NCCL 通信日志 | 环境变量设置 |
| `torch.profiler` | 性能分析 | 分析通信/计算比例 |
| `py-spy` | Python 进程采样 | `py-spy dump --pid <pid>` |
| `nvidia-smi` | GPU 状态监控 | `watch -n 1 nvidia-smi` |
| `gpustat` | GPU 状态（更友好） | `gpustat -i 1` |
| `nsys` | Nsight Systems 性能分析 | `nsys profile python train.py` |
| `torch.cuda.memory_stats()` | 显存详细统计 | 代码中调用 |

### torch.profiler 使用示例

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in dataloader:
        with record_function("model_forward"):
            loss = model(batch)
        with record_function("model_backward"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()
        prof.step()
```

---

## 通用排查流程

```
┌─────────────────────────────────────────────┐
│              1. 复现 (Reproduce)              │
│  确认 bug 可稳定复现，记录触发条件            │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│              2. 隔离 (Isolate)               │
│  缩小范围：多节点→单节点→多卡→单卡           │
│  如果单卡正常，问题在分布式组件               │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│              3. 定位 (Locate)                │
│  根据错误类型选择排查方向：                   │
│  - 通信错误 → NCCL 调试                      │
│  - 数值错误 → 梯度 hook + detect_anomaly     │
│  - 显存错误 → 显存分析                       │
│  - 挂起 → 死锁排查                           │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│              4. 修复 (Fix)                   │
│  针对性修复，避免过度修改                     │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│              5. 验证 (Verify)                │
│  确认修复有效，且未引入新问题                 │
│  在完整规模上验证                             │
└─────────────────────────────────────────────┘
```

### 快速排查清单

遇到问题时，按以下顺序检查：

- [ ] 单卡训练是否正常？
- [ ] 环境变量是否正确？（MASTER_ADDR、MASTER_PORT、RANK、WORLD_SIZE）
- [ ] NCCL 版本是否与 CUDA 版本兼容？
- [ ] 所有进程是否同时启动？
- [ ] 学习率是否适合当前模型规模？
- [ ] 是否启用了梯度裁剪？
- [ ] 显存使用是否接近上限？
- [ ] 数据加载是否均衡？

---

## 与其他技术的关系

| 调试技术 | 相关并行策略 | 关系说明 |
|---------|------------|---------|
| NCCL 调试 | 所有分布式策略 | NCCL 是所有 GPU 通信的基础 |
| 梯度检查 | DP、TP | 梯度一致性验证 |
| 显存分析 | 所有策略 | OOM 是最常见的显存问题 |
| Profiler | 所有策略 | 分析通信/计算比例，指导策略选择 |
| 数值一致性 | TP、PP | 模型并行更容易引入数值差异 |

---

## 参考资料

### 官方文档

- [PyTorch 分布式调试文档](https://pytorch.org/docs/stable/distributed.html#debugging)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch Profiler 文档](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### 调试指南

- [PyTorch 分布式训练故障排除](https://pytorch.org/docs/stable/notes/faq.html)
- [NCCL 环境变量参考](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [DeepSpeed 调试指南](https://www.deepspeed.ai/troubleshooting/)

### 社区资源

- [PyTorch 论坛 - 分布式训练板块](https://discuss.pytorch.org/c/distributed/42)
- [NVIDIA 开发者论坛 - NCCL](https://forums.developer.nvidia.com/c/accelerated-computing/nccl/)
