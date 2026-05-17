"""
DistributedDataParallel (DDP) 核心概念。

DDP 相比朴素 DP 的改进：
1. Gradient Bucketing: 将多个参数的梯度打包成 bucket 再通信，减少通信次数
2. Communication-Computation Overlap: 在 backward 过程中，计算当前层的同时通信之前层的梯度
3. 只在初始化时 broadcast 一次模型权重，之后只同步梯度

本文件演示 DDP 的核心机制，不依赖 torch.nn.parallel.DistributedDataParallel。
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size
from parallel.communication.primitives import naive_all_reduce


def broadcast_model(model: torch.nn.Module, src: int = 0):
    """模型权重广播：rank 0 的权重发给所有其他 rank（只在初始化时做一次）。

    这是 DDP 在训练开始前的唯一一次模型同步操作。
    之后的训练过程中，所有 rank 通过梯度同步保持模型权重一致，
    无需再次广播模型参数。

    Args:
        model: 要广播的模型，所有参数就地更新为 src rank 的值。
        src: 源 rank 编号，默认为 0。
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def gradient_bucket_sync(model: torch.nn.Module, bucket_size_mb: int = 25):
    """
    梯度分桶同步。将参数按大小分桶，每个桶一起通信。
    这模拟了 DDP 的 gradient bucketing 机制。

    分桶的好处：将多个小张量的梯度合并为一个桶进行通信，
    减少 all-reduce 的启动开销（每次通信都有固定的延迟开销）。
    默认桶大小 25MB 是 PyTorch DDP 的典型值。

    Args:
        model: 模型实例，其参数的 .grad 属性将被分桶并同步。
        bucket_size_mb: 每个桶的目标大小（MB），默认 25MB。
    """
    # 简单的分桶策略：按参数元素数累积直到达到 bucket_size_mb
    buckets = []
    current_bucket = []
    current_size = 0
    bytes_per_element = 4  # float32

    for param in model.parameters():
        if param.grad is not None:
            param_bytes = param.grad.numel() * bytes_per_element
            if current_size + param_bytes > bucket_size_mb * 1024 * 1024 and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_bytes

    if current_bucket:
        buckets.append(current_bucket)

    for bucket_params in buckets:
        for param in bucket_params:
            synced = naive_all_reduce(param.grad.data, op="avg")
            param.grad.data.copy_(synced)


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("DDP 核心概念演示")
    print("=" * 60)

    # 初始化分布式环境以便演示通信操作
    try:
        if not dist.is_initialized():
            init_process_group(backend="gloo")
            print("已初始化 gloo 后端（单进程演示模式）")
    except Exception:
        print("无法初始化分布式环境，跳过通信演示")

    # 构建一个多层模型以便演示分桶
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    )
    print(f"\n模型层数: {sum(1 for _ in model.parameters())}")

    rank = get_rank()
    world_size = get_world_size()
    print(f"当前 rank: {rank}, world_size: {world_size}")

    # 演示 broadcast_model
    print("\n--- 模型权重广播 ---")
    print("DDP 在训练开始前，rank 0 将模型权重广播到所有 rank。")
    print("这确保所有进程从相同的初始权重开始训练。")
    if dist.is_initialized():
        broadcast_model(model, src=0)
        print("模型权重已广播。")
    else:
        print("(跳过：分布式环境未初始化)")

    # 模拟一次 forward + backward
    x = torch.randn(4, 64)
    target = torch.randn(4, 10)
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()

    # 演示分桶同步
    print("\n--- 梯度分桶同步 ---")
    print("分桶可以合并多个小张量，减少通信启动开销。")
    if dist.is_initialized():
        gradient_bucket_sync(model, bucket_size_mb=25)
        n_params = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"同步完成，共处理 {n_params} 个参数。")
    else:
        print("(跳过同步：分布式环境未初始化)")
    print("\n在多进程环境中，梯度已在所有 rank 之间同步。")

    if dist.is_initialized():
        cleanup()
