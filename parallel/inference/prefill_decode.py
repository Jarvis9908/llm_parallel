"""Prefill vs Decode 阶段的并行策略切换分析。

在大语言模型推理过程中，Prefill（预填充）和 Decode（逐 token 解码）
两个阶段的计算特征截然不同，因此最优并行策略也应当随之切换。
"""
import torch


def analyze_prefill_characteristics(seq_len: int, dim: int) -> dict:
    """
    分析 Prefill 阶段的计算特征。

    Prefill 阶段一次性处理整个 prompt（seq_len 个 token），
    涉及大矩阵乘法，属于计算密集型（compute-bound）阶段。
    适合使用张量并行（TP）将大矩阵运算分布到多张 GPU 上同时计算，
    输出 KV Cache 需要保存下来供后续 decode 阶段使用。

    Args:
        seq_len: 输入 prompt 的 token 数量
        dim: 模型隐藏维度

    Returns:
        dict: 包含 phase、compute_bound、recommended_strategy、total_flops 等字段
    """
    flops_per_token = 2 * dim * dim  # 单 token 约需 2*d^2 次浮点运算（简化估算）
    total_flops = seq_len * flops_per_token
    return {
        "phase": "prefill",
        "compute_bound": True,
        "recommended_strategy": "TP (Tensor Parallel)",
        "total_flops": total_flops,
    }


def analyze_decode_characteristics(seq_len: int, dim: int) -> dict:
    """
    分析 Decode 阶段的计算特征。

    Decode 阶段每步只处理 1 个新 token，计算量小但需要访问完整的 KV Cache
    进行自注意力计算，属于访存密集型（memory-bound）阶段。
    适合使用数据并行（DP）将多个请求 batch 处理，或使用专家并行（EP）
    对 MoE 模型进行路由。KV Cache 在此阶段读多写少。

    Args:
        seq_len: 已缓存的序列长度（即 KV Cache 大小）
        dim: 模型隐藏维度

    Returns:
        dict: 包含 phase、memory_bound、recommended_strategy、kv_cache_intensive 等字段
    """
    return {
        "phase": "decode",
        "memory_bound": True,
        "recommended_strategy": "DP or EP (Expert Parallel for MoE)",
        "kv_cache_intensive": True,
    }


def recommend_strategy(seq_len: int, dim: int, n_gpus: int) -> str:
    """
    根据序列长度和可用 GPU 数推荐两阶段并行策略。

    长序列（>4096）场景：Prefill 用 TP 分工计算完整 prompt 并生成 KV Cache，
    然后切换到 DP 各自处理后续 decode，KV Cache 已在各自 GPU 上就位。
    短序列场景：直接使用 DP 即可，无需在两阶段之间切换策略。

    Args:
        seq_len: 输入 prompt 的 token 数量
        dim: 模型隐藏维度
        n_gpus: 可用 GPU 数量

    Returns:
        str: 推荐的并行策略描述
    """
    if seq_len > 4096:
        return "Prefill: TP(4) → Decode: DP + KV Cache sharding"
    else:
        return "Prefill: DP → Decode: DP (no strategy switch needed)"


if __name__ == "__main__":
    # 演示：分析长序列和短序列下的推荐策略
    print("=== Prefill 阶段分析 ===")
    prefill_info = analyze_prefill_characteristics(seq_len=8192, dim=4096)
    for k, v in prefill_info.items():
        print(f"  {k}: {v}")

    print("\n=== Decode 阶段分析 ===")
    decode_info = analyze_decode_characteristics(seq_len=8192, dim=4096)
    for k, v in decode_info.items():
        print(f"  {k}: {v}")

    print("\n=== 策略推荐 ===")
    print(f"长序列 (8192): {recommend_strategy(seq_len=8192, dim=4096, n_gpus=8)}")
    print(f"短序列 (2048): {recommend_strategy(seq_len=2048, dim=4096, n_gpus=8)}")
