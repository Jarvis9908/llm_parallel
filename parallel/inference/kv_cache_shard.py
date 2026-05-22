"""KV Cache 分片：在推理时将 KV Cache 按 head 维度分布到多卡，减少单卡显存。

直觉
----
KV Cache 就像一个大文件，按 head 维度切分到不同硬盘上存储——
每张卡只存自己负责的那部分 head 的 K 和 V，需要完整结果时再拼接。
这样单卡显存只需原来的 1/P（P 为 GPU 数），支持更长序列或更大 batch。

数学
----
1. 总 KV Cache 显存（float16）：
        total = 4 × B × n_heads × S × d_head  bytes
   其中系数 4 = 2（K+V 各一份）× 2（float16 每元素 2 字节）

2. 分片后单卡显存：
        per_gpu = total / P
   节省比例 = 1 - 1/P

3. 分片策略对比：
   - 按 head 切（TP 风格）：每卡持有 n_heads/P 个 head 的完整 KV
     优点：无需通信即可独立计算局部注意力
     缺点：需要 All-Reduce 汇总各 head 的注意力输出
   - 按 batch 切（DP 风格）：每卡持有 B/P 个样本的完整 KV
     优点：完全独立，无需通信
     缺点：无法扩大单样本的序列长度

代码流程
--------
1. ``shard_kv_cache_by_heads`` —— 按 head 维度切分 KV Cache
2. ``gather_kv_cache`` —— 收集所有分片恢复完整 KV Cache
3. ``kv_cache_memory_analysis`` —— KV Cache 显存占用分析
"""
import torch


def shard_kv_cache_by_heads(
    k: torch.Tensor, v: torch.Tensor, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    按 head 维度切分 KV Cache。

    将多头注意力中的 K 和 V 按 head 维度均匀分片到 world_size 个 GPU 上，
    每个 rank 只持有自己负责的那部分 heads 的缓存，从而降低单卡显存占用。

    Args:
        k: Key 张量，形状 (batch, n_heads, seq_len, head_dim)
        v: Value 张量，形状 (batch, n_heads, seq_len, head_dim)
        rank: 当前设备的序号，取值范围 [0, world_size)
        world_size: 并行 GPU 数量（总设备数）

    Returns:
        (k_local, v_local): 当前 rank 对应的本地 KV Cache 分片
    """
    heads_per_rank = k.shape[1] // world_size
    start = rank * heads_per_rank
    end = start + heads_per_rank
    return k[:, start:end], v[:, start:end]


def gather_kv_cache(
    k_local: torch.Tensor, v_local: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    收集所有 rank 的 KV Cache，恢复完整的 heads。

    在需要完整上下文时（如 beam search 重排序），将所有分片沿 head 维度拼接，
    还原为完整的 K 和 V 张量。当前为单机演示实现，直接返回本地副本。

    Args:
        k_local: 本地的 Key 分片
        v_local: 本地的 Value 分片
        world_size: 并行 GPU 数量

    Returns:
        (k_full, v_full): 拼接后的完整 KV Cache
    """
    return k_local, v_local


def kv_cache_memory_analysis(
    batch_size: int, n_heads: int, seq_len: int, head_dim: int, n_gpus: int
) -> dict:
    """
    KV Cache 显存占用分析。

    计算 KV Cache 在 float16 精度下的总显存占用，以及分片后每张 GPU
    的显存占用和节省比例。KV Cache 需要同时存储 K 和 V（各一份），
    所以乘以 2。

    Args:
        batch_size: 批次大小
        n_heads: 注意力头数
        seq_len: 序列长度（已缓存的 token 数）
        head_dim: 每个注意力头的维度
        n_gpus: 用于分片的 GPU 数量

    Returns:
        dict: 包含 total_kv_cache_mb（总显存 MB）、per_gpu_sharded_mb（单卡 MB）、
              savings_ratio（节省比例）三项指标
    """
    bytes_per_element = 2  # float16 占用 2 字节
    total_bytes = 2 * batch_size * n_heads * seq_len * head_dim * bytes_per_element
    per_gpu_sharded = total_bytes / n_gpus
    return {
        "total_kv_cache_mb": total_bytes / (1024 ** 2),
        "per_gpu_sharded_mb": per_gpu_sharded / (1024 ** 2),
        "savings_ratio": 1 - (1 / n_gpus),
    }


if __name__ == "__main__":
    # 演示：4 个头，2 张 GPU，每个 rank 获得 2 个头的 KV Cache
    batch, n_heads, seq_len, head_dim = 2, 4, 128, 64
    k = torch.randn(batch, n_heads, seq_len, head_dim)
    v = torch.randn(batch, n_heads, seq_len, head_dim)

    for rank in range(2):
        k_local, v_local = shard_kv_cache_by_heads(k, v, rank, world_size=2)
        print(f"[Rank {rank}] k_local shape: {k_local.shape}, v_local shape: {v_local.shape}")

    analysis = kv_cache_memory_analysis(batch, n_heads, seq_len, head_dim, n_gpus=4)
    print(f"\nKV Cache 显存分析: {analysis}")
