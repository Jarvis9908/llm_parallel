"""通信量与延迟模拟器：估算不同并行策略下集合通信操作的耗时与数据量。"""
import math


def simulate_all_reduce(
    tensor_size_bytes: int, world_size: int, bandwidth_gb_s: float = 50.0
) -> float:
    """
    模拟 ring all-reduce 通信时间。

    Ring all-reduce 分两步完成：第一步 scatter-reduce（每个 rank 聚合数据），
    第二步 all-gather（每个 rank 收集完整结果）。
    每个 rank 发送和接收的总数据量为 2 * (P-1)/P * N bytes，
    其中 P = world_size, N = tensor_size_bytes。

    Args:
        tensor_size_bytes: 待归约的张量大小（字节）
        world_size: 参与通信的 GPU 数量
        bandwidth_gb_s: GPU 间通信带宽（GB/s），默认 50 GB/s（NVLink 3.0 级别）

    Returns:
        float: all-reduce 操作耗时（秒）
    """
    data_transferred = 2 * (world_size - 1) / world_size * tensor_size_bytes
    bandwidth_bytes_s = bandwidth_gb_s * 1e9
    return data_transferred / bandwidth_bytes_s


def simulate_all_to_all(
    tensor_size_bytes: int, world_size: int, bandwidth_gb_s: float = 50.0
) -> float:
    """
    模拟 all-to-all 通信时间。

    在专家并行（EP）中，all-to-all 用于将 token 按路由决策分发到
    各专家的所在 GPU。每个 rank 向所有其他 rank 发送 1/P 的数据，
    总通信量约为 (P-1)/P * N bytes。

    Args:
        tensor_size_bytes: 待分发的张量大小（字节）
        world_size: 参与通信的 GPU 数量
        bandwidth_gb_s: GPU 间通信带宽（GB/s），默认 50 GB/s

    Returns:
        float: all-to-all 操作耗时（秒）
    """
    data_transferred = (world_size - 1) / world_size * tensor_size_bytes
    bandwidth_bytes_s = bandwidth_gb_s * 1e9
    return data_transferred / bandwidth_bytes_s


def compare_parallel_strategies(
    model_size_gb: float, seq_len: int, n_gpus: int
) -> dict:
    """
    对比四种主流并行策略的通信开销特征。

    分析数据并行（DP）、张量并行（TP）、流水线并行（PP）、
    专家并行（EP）的通信量和通信频率差异，帮助选择最优策略组合。

    Args:
        model_size_gb: 模型参数量对应的存储大小（GB，float16 下约为 2 * 参数量）
        seq_len: 序列长度
        n_gpus: 可用 GPU 数量

    Returns:
        dict: 包含四种策略的通信量（GB）和通信频率描述
    """
    return {
        "Data Parallel": {
            "comm_volume_gb": model_size_gb,
            "frequency": "每步（all-reduce 同步梯度）",
        },
        "Tensor Parallel": {
            "comm_volume_gb": model_size_gb * 2,
            "frequency": "每层（all-reduce / all-gather）",
        },
        "Pipeline Parallel": {
            "comm_volume_gb": seq_len * 0.001,
            "frequency": "每 micro-batch（P2P 传递激活）",
        },
        "Expert Parallel": {
            "comm_volume_gb": model_size_gb * 0.1,
            "frequency": "每 MoE 层（all-to-all 分发 token）",
        },
    }


if __name__ == "__main__":
    print("=== All-Reduce 通信模拟 ===")
    # 模拟 100MB 张量在 8 卡上的 all-reduce 耗时
    tensor_bytes = 100 * 1024 * 1024  # 100 MB
    time_s = simulate_all_reduce(tensor_bytes, world_size=8, bandwidth_gb_s=50.0)
    print(f"  100MB all-reduce (8 GPUs, 50GB/s): {time_s * 1000:.3f} ms")

    print("\n=== All-to-All 通信模拟 ===")
    time_s = simulate_all_to_all(tensor_bytes, world_size=8, bandwidth_gb_s=50.0)
    print(f"  100MB all-to-all (8 GPUs, 50GB/s): {time_s * 1000:.3f} ms")

    print("\n=== 并行策略通信对比 ===")
    comparison = compare_parallel_strategies(model_size_gb=10.0, seq_len=8192, n_gpus=8)
    for strategy, info in comparison.items():
        print(f"  {strategy}: 通信量 {info['comm_volume_gb']:.2f} GB, 频率: {info['frequency']}")
