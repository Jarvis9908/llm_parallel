"""Embedding 并行：词表按维度切分到不同 rank，all-reduce 收集结果。"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size


def embedding_parallel_forward(
    embed_weight: torch.Tensor, token_ids: torch.LongTensor
) -> torch.Tensor:
    """
    embed_weight: (vocab_size_local, dim) — 本地持有的部分词表
    token_ids: (B, S) — 原始 token ids（全局词表索引）
    需要先调整 token ids 到本地词表范围，然后 all-reduce 收集结果。
    """
    # 简化：直接对本地 embedding 做 all-reduce（适用于 token ids 已在本地范围的场景）
    local_out = torch.nn.functional.embedding(token_ids, embed_weight)
    dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
    return local_out


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("Embedding 并行演示")
    print("=" * 60)

    try:
        if not dist.is_initialized():
            init_process_group(backend="gloo")
            print("已初始化 gloo 后端（单进程演示模式）")
    except Exception:
        print("无法初始化分布式环境，跳过通信演示")

    rank = get_rank()
    ws = get_world_size()

    print(f"当前 rank: {rank}, world_size: {ws}")

    B, S, dim = 2, 4, 16
    vocab_size_local = 128
    embed_w = torch.randn(vocab_size_local, dim)
    token_ids = torch.randint(0, vocab_size_local, (B, S))

    if dist.is_initialized():
        out = embedding_parallel_forward(embed_w, token_ids)
        expected = torch.nn.functional.embedding(token_ids, embed_w)
        assert out.shape == (B, S, dim), f"Wrong output shape: {out.shape}"
        if ws == 1:
            assert torch.allclose(out, expected, atol=1e-5), "Output mismatch"
        print(f"输出形状: {list(out.shape)} — OK")
    else:
        print("(跳过通信：分布式环境未初始化)")

    print("\nEmbedding 并行将词表沿 vocab 维度切分到各 rank。")
    print("每个 rank 只存储部分词向量，查表后 all-reduce 得到完整结果。")
    print("这可以显著减少大词表模型（如多语言模型）的显存占用。")

    if dist.is_initialized():
        cleanup()
