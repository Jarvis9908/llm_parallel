"""环形注意力 (Ring Attention)。序列沿长度切分，KV block 在环形拓扑中轮转传递。"""
import torch


def ring_attention_step(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, step: int
) -> torch.Tensor:
    """
    环形注意力的单步：用本地 Q 和当前持有的 KV 计算 partial attention。

    Ring Attention 的核心思想：
    1. 将输入序列沿 seq_len 切分到各 GPU
    2. 各 GPU 计算本地 Q @ 本地 KV → 得到 partial attention
    3. 将 KV block 传给下一个 GPU（环形传递）
    4. 重复直到 KV 绕行一圈，每步累积 softmax 结果

    通信量: O(N * P) 而非 O(N² * P)，长序列友好。
    """
    scale = q.shape[-1] ** 0.5
    scores = (q @ k.transpose(-2, -1)) / scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def rotate_kv(
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]], direction: int = 1
):
    """在环形拓扑中传递 KV block。direction=1 表示向前传递。"""
    # 在单机演示中直接返回
    return kv_cache


if __name__ == "__main__":
    print("=== ring_attention demo ===")

    # 模拟 2 个头，seq_len=8，d_head=4
    B = 1
    n_heads = 2
    seq_len = 8
    d_head = 4
    q = torch.randn(B, n_heads, seq_len, d_head)
    k = torch.randn(B, n_heads, seq_len, d_head)
    v = torch.randn(B, n_heads, seq_len, d_head)

    # 模拟环形传递多步
    num_steps = 4
    print(f"Sequence length: {seq_len}, d_head: {d_head}, steps: {num_steps}")
    for step in range(num_steps):
        output = ring_attention_step(q, k, v, step)
        print(f"  Step {step}: output shape {output.shape}")

    # KV 旋转
    print(f"\nKV rotation demo:")
    kv_cache = [(k, v)]
    rotated = rotate_kv(kv_cache, direction=1)
    print(f"  Returned {len(rotated)} KV pairs (no-op on single machine)")
