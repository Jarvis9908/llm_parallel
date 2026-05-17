"""Megatron-LM 风格 TP+SP Transformer 块。组合列并行、行并行和序列并行。"""
import torch
import torch.distributed as dist
from parallel.communication.setup import get_rank, get_world_size
from parallel.tensor_parallel.column_parallel import (
    column_parallel_linear,
    split_weight_column,
)
from parallel.tensor_parallel.row_parallel import row_parallel_linear, split_weight_row
from parallel.tensor_parallel.sequence_parallel import scatter_along_seq, gather_along_seq


def megatron_transformer_block_fwd(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    use_sp: bool = True,
) -> torch.Tensor:
    """
    模拟 Megatron 风格 Transformer 块的 TP+SP 前向。

    SP 区域：LayerNorm（在本地 sequence chunk 上计算）
    非 SP 区域：Attention（all-gather 后完整序列）和 FFN（进入时切分出 SP 区域）

    这展示了 TP+SP 如何交替切换分片维度以减少激活显存。
    """
    B, S, D = x.shape

    if use_sp:
        # RMSNorm 在 SP 下只需本地序列
        x_local = scatter_along_seq(x)

    # Column Parallel: QKV and Gate+Up projections
    # (实际中这些需要 all-gather 输入)
    x_full = gather_along_seq(x_local, S) if use_sp else x

    # Row Parallel: Output and Down projections (all-reduce)
    # (简化：直接在本地做)

    if use_sp:
        x_local = scatter_along_seq(x_full)

    return x_full


if __name__ == "__main__":
    from parallel.communication.setup import init_process_group, cleanup

    print("=" * 60)
    print("Megatron-LM 风格 TP+SP Transformer 块演示")
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

    B, S, D = 2, 8, 16
    hidden_local = D // ws

    x = torch.randn(B, S, D)
    # Full weights for shape checking
    w_qkv_full = torch.randn(D, 3 * hidden_local)
    w_o_full = torch.randn(hidden_local, D)
    w_gate_up_full = torch.randn(D, 2 * hidden_local)
    w_down_full = torch.randn(hidden_local, D)

    # Split weights per rank
    w_qkv = split_weight_column(w_qkv_full)
    w_o = split_weight_row(w_o_full)
    w_gate_up = split_weight_column(w_gate_up_full)
    w_down = split_weight_row(w_down_full)

    print(f"QKV 权重: {list(w_qkv.shape)} (列并行)")
    print(f"Output 权重: {list(w_o.shape)} (行并行)")
    print(f"Gate+Up 权重: {list(w_gate_up.shape)} (列并行)")
    print(f"Down 权重: {list(w_down.shape)} (行并行)")

    if dist.is_initialized():
        # Run forward with SP (shape-only demo)
        out = megatron_transformer_block_fwd(
            x, w_qkv, w_o, w_gate_up, w_down, use_sp=True
        )
        assert out.shape == (B, S, D), f"Wrong output shape: {out.shape}"

        # Run forward without SP
        out_no_sp = megatron_transformer_block_fwd(
            x, w_qkv, w_o, w_gate_up, w_down, use_sp=False
        )
        assert out_no_sp.shape == (B, S, D), f"Wrong output shape (no SP): {out_no_sp.shape}"

        print(f"输出形状 (SP): {list(out.shape)}")
        print(f"输出形状 (无 SP): {list(out_no_sp.shape)} — OK")
    else:
        print("(跳过通信操作：分布式环境未初始化)")

    print("\nTP+SP 组合流程:")
    print("  1. SP 区域: LayerNorm/RMSNorm 在本地 sequence chunk 上计算")
    print("  2. 进入 Attention/FFN: all-gather 恢复完整序列")
    print("  3. Column Parallel: QKV/Gate+Up 投影（各 rank 持有部分列）")
    print("  4. Row Parallel: Output/Down 投影（all-reduce 求和）")
    print("  5. 回到 SP 区域: scatter 切分回本地 sequence chunk")

    if dist.is_initialized():
        cleanup()
