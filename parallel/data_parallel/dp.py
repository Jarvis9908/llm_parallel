"""
朴素数据并行 (Data Parallel, DP)。

原理：每张 GPU 持有完整模型副本，各自处理不同的 mini-batch，
      forward 后 all-reduce 同步梯度，再各自更新参数。

缺点：每张卡都要存完整模型+优化器状态，显存压力大。
"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size
from parallel.communication.primitives import naive_all_reduce


def sync_gradients_naive(model: torch.nn.Module, op: str = "avg"):
    """
    同步所有 rank 的梯度。对每个参数的 .grad 做 all-reduce。
    这是 DataParallel 的核心操作。

    Args:
        model: 模型实例，其参数的 .grad 属性将被同步。
        op: 规约操作，"sum" 表示求和，"avg" 表示平均（默认）。
            训练时通常使用 "avg"，因为每个 rank 的 loss 已经做了平均。
    """
    for param in model.parameters():
        if param.grad is not None:
            synced = naive_all_reduce(param.grad.data, op=op)
            param.grad.data.copy_(synced)


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("朴素数据并行 (DP) 演示")
    print("=" * 60)

    # 初始化分布式环境以便演示通信操作
    try:
        if not dist.is_initialized():
            init_process_group(backend="gloo")
            print("已初始化 gloo 后端（单进程演示模式）")
    except Exception:
        print("无法初始化分布式环境，跳过通信演示")

    # 构建一个简单模型
    model = torch.nn.Linear(4, 2)
    print(f"\n模型: {model}")
    print(f"当前 rank: {get_rank()}")
    print(f"world_size: {get_world_size()}")

    # 模拟一次 forward + backward
    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()

    print(f"\n同步前 grad 示例 (weight[0,0]): {model.weight.grad[0, 0].item():.6f}")

    # 核心操作：同步梯度
    if dist.is_initialized():
        sync_gradients_naive(model, op="avg")
        print(f"同步后 grad 示例 (weight[0,0]): {model.weight.grad[0, 0].item():.6f}")
    else:
        print("(跳过同步：分布式环境未初始化)")

    print("\n在 world_size=1 时，同步前后梯度相同（只有一个 rank 参与规约）。")
    print("多进程运行时，每个 rank 的梯度会被求和/平均后写回。")

    if dist.is_initialized():
        cleanup()
