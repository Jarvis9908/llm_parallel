"""CP 与其他并行策略的混合方案。

直觉
----
TP（张量并行）切权重，CP（上下文并行）切序列——两者互补。
TP 解决单层太大放不下一张卡的问题，CP 解决序列太长显存不够的问题。
两者可以正交组合：TP 在 head 维度切分注意力，CP 在序列维度切分，
每张卡只需处理 B × (S/P_cp) × D/P_tp 的激活。

数学
----
1. CP+TP 显存分析：
   单卡激活大小 ∝ B × S × D / (P_tp × P_cp)
   - P_tp: TP 并行度（切 head 维度）
   - P_cp: CP 并行度（切序列维度）
   两者乘积越大，每卡激活越小。

2. 并行策略选择启发式规则：
   - 小模型（<1GB）：DP 即可，无需切分
   - 中模型（1-10GB）：TP + DP，TP 切权重降低单卡显存
   - 大模型（>10GB）：TP + PP + DP，PP 进一步切层
   - 长序列（>8K）：在上述基础上额外加 CP，切序列降低激活显存

代码流程
--------
1. ``analyze_cp_tp_memory`` —— 分析 CP+TP 混合并行的显存节省
2. ``recommend_parallel_config`` —— 根据模型规模推荐并行策略
"""
import torch


def analyze_cp_tp_memory(
    seq_len: int, dim: int, n_heads: int, tp_size: int, cp_size: int
) -> dict:
    """
    分析 CP+TP 混合下的显存占用。

    直觉：TP 把权重按 head 切，CP 把序列按长度切，两者正交组合，
    每张卡的激活只需要原来的 1/(P_tp × P_cp)。

    数学：
        total_activation = S × D（简化模型，忽略 batch 和 head）
        per_device_activation = total_activation / (P_tp × P_cp)
        reduction_ratio = 1 / (P_tp × P_cp)

    Args:
        seq_len: 序列长度 S
        dim: 隐层维度 D
        n_heads: 注意力头数（当前简化模型未使用）
        tp_size: TP 并行度 P_tp
        cp_size: CP 并行度 P_cp

    Returns:
        dict: 包含 total_activation_memory、per_device_activation、
              reduction_ratio 三项指标
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

    直觉：小模型用 DP 就够，大模型需要 TP 切权重，超大模型还要 PP 切层，
    长序列则额外需要 CP 切序列——按需叠加，避免过度切分导致通信开销过大。

    数学（启发式规则）：
        model_size < 1GB  → DP only
        1GB ≤ model_size < 10GB → TP + DP
        model_size ≥ 10GB → TP + PP + DP
        seq_len > 8192 → 在上述基础上 + CP

    Args:
        model_size_gb: 模型参数量对应的显存大小（GB）
        seq_len: 输入序列长度
        n_gpus: 可用 GPU 数量

    Returns:
        str: 推荐的并行策略描述
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
