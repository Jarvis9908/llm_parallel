"""CP 与其他并行策略的混合方案。"""
import torch


def analyze_cp_tp_memory(
    seq_len: int, dim: int, n_heads: int, tp_size: int, cp_size: int
) -> dict:
    """
    分析 CP+TP 混合下的显存占用。

    TP 切分：每卡显存 ≈ total_memory / tp_size
    CP 切分：每卡序列长度 ≈ seq_len / cp_size
    CP+TP：每卡显存 ≈ total_memory / (tp_size * cp_size)
    """
    total_activation = seq_len * dim  # 简化
    per_device_activation = total_activation / (tp_size * cp_size)
    return {
        "total_activation_memory": total_activation,
        "per_device_activation": per_device_activation,
        "reduction_ratio": 1 / (tp_size * cp_size),
    }


def recommend_parallel_config(
    model_size_gb: float, seq_len: int, n_gpus: int
) -> str:
    """
    根据模型大小和 GPU 数量推荐并行配置。
    启发式规则：
    - 模型 < 1GB: DP only
    - 模型 1-10GB: TP + DP
    - 模型 > 10GB: TP + PP + DP
    - 长序列 (>8K): 额外加 CP
    """
    if model_size_gb < 1:
        return "DP only"
    elif model_size_gb < 10:
        base = "TP + DP"
    else:
        base = "TP + PP + DP"
    if seq_len > 8192:
        base += " + CP"
    return base


if __name__ == "__main__":
    print("=== cp_integration demo ===")

    # CP+TP 显存分析
    configs = [
        (4096, 1024, 32, 4, 2),   # 标准 Llama 风格
        (8192, 2048, 64, 8, 4),   # 大模型
        (32768, 4096, 128, 8, 8), # 长序列
    ]
    for seq_len, dim, n_heads, tp, cp in configs:
        mem = analyze_cp_tp_memory(seq_len, dim, n_heads, tp, cp)
        print(f"seq={seq_len:5d}, dim={dim:4d}, TP={tp}, CP={cp}: "
              f"per_device={mem['per_device_activation']:8.1f} "
              f"(ratio={mem['reduction_ratio']:.4f})")

    # 并行策略推荐
    print("\nParallel configuration recommendations:")
    scenarios = [
        (0.5, 2048, 4),     # 小模型，短序列
        (7.0, 4096, 8),     # 中等模型
        (30.0, 8192, 16),   # 大模型
        (50.0, 16384, 32),  # 大模型+长序列
    ]
    for model_gb, seq_len, n_gpus in scenarios:
        config = recommend_parallel_config(model_gb, seq_len, n_gpus)
        print(f"  Model={model_gb:5.1f}GB, seq={seq_len:5d}, GPUs={n_gpus:2d} -> {config}")
