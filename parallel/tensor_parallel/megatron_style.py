"""Megatron-LM 风格 TP+SP Transformer 块 —— 列并行与行并行配对，结合序列并行优化激活显存。

直觉理解
--------
Transformer 中的 Attention 和 FFN 天然可以拆成「列并行 → 行并行」的对子：
QKV 投影是列并行（按列切权重），Output 投影是行并行（按行切权重）；
Gate+Up 投影是列并行，Down 投影是行并行。每对只需一次 all-reduce 通信。
序列并行 (SP) 则在 LayerNorm/Dropout 区域沿序列维度切分激活值，进一步节省显存。

数学原理
--------
一个完整 Transformer 块的数据流（含 SP）：

    ┌─────────────────────────────────────────────────────────────────┐
    │  SP 区域: x_local ∈ ℝ^(B×(S/P)×D)                            │
    │    ↓ All-Gather (沿 seq 维度)                                  │
    │  x_full ∈ ℝ^(B×S×D)                                           │
    │    ↓ Column Parallel (QKV 投影)                                │
    │  qkv_local ∈ ℝ^(B×S×(3H/P))                                   │
    │    ↓ Attention 计算                                             │
    │  attn_out_local ∈ ℝ^(B×S×(H/P))                                │
    │    ↓ Row Parallel + All-Reduce (Output 投影)                    │
    │  o_full ∈ ℝ^(B×S×D)                                            │
    │    ↓ Reduce-Scatter (沿 seq 维度)                               │
    │  SP 区域: x_local ∈ ℝ^(B×(S/P)×D)                             │
    │    ↓ All-Gather                                                 │
    │  x_full ∈ ℝ^(B×S×D)                                            │
    │    ↓ Column Parallel (Gate+Up 投影)                             │
    │  gate_up_local ∈ ℝ^(B×S×(2H/P))                                │
    │    ↓ FFN 计算                                                   │
    │  ffn_out_local ∈ ℝ^(B×S×(H/P))                                 │
    │    ↓ Row Parallel + All-Reduce (Down 投影)                      │
    │  out_full ∈ ℝ^(B×S×D)                                          │
    │    ↓ Reduce-Scatter                                             │
    │  SP 区域: out_local ∈ ℝ^(B×(S/P)×D)                           │
    └─────────────────────────────────────────────────────────────────┘

通信次数：每个 Transformer 块 4 次通信（2 次 All-Gather + 2 次 Reduce-Scatter），
其中 All-Gather 和 Reduce-Scatter 是 SP 区域与非 SP 区域之间的转换，
列并行→行并行的 all-reduce 已包含在 Row Parallel 步骤中。

代码流程
--------
1. megatron_transformer_block_fwd: 模拟完整 Transformer 块的 TP+SP 前向流程
"""
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
    """模拟 Megatron 风格 Transformer 块的 TP+SP 前向传播。

    直觉
    ----
    一个 Transformer 块 = 两对「列并行→行并行」+ SP 区域切换。
    Attention 是第一对（QKV 列并行 → Output 行并行），
    FFN 是第二对（Gate+Up 列并行 → Down 行并行）。
    SP 在 LayerNorm 处沿序列维度切分激活值，进出时做通信切换。

    数学
    ----
    数据流与通信量（use_sp=True 时）：

    1. SP→TP: All-Gather, 通信量 B×S×D
       x_local (B, S/P, D) → x_full (B, S, D)
    2. Column Parallel QKV: x_full @ w_qkv → qkv_local (B, S, 3H/P)
    3. Attention 计算 → attn_out_local (B, S, H/P)
    4. Row Parallel Output + All-Reduce: attn_out_local @ w_o → o_full (B, S, D)
       通信量 B×S×D
    5. TP→SP: Reduce-Scatter, 通信量 B×S×D
       o_full (B, S, D) → x_local (B, S/P, D)
    6. SP→TP: All-Gather, 通信量 B×S×D
    7. Column Parallel Gate+Up: x_full @ w_gate_up → gate_up_local (B, S, 2H/P)
    8. FFN 计算 → ffn_out_local (B, S, H/P)
    9. Row Parallel Down + All-Reduce: ffn_out_local @ w_down → out_full (B, S, D)
       通信量 B×S×D
    10. TP→SP: Reduce-Scatter, 通信量 B×S×D

    每个 Transformer 块总通信量：4 × B×S×D（2 次 All-Gather + 2 次 Reduce-Scatter，
    行并行的 all-reduce 已内含在步骤 4、9 中）。

    Args:
        x: 输入张量，形状 (B, S, D)，SP 模式下为完整序列（函数内部做 scatter）。
        w_qkv: QKV 投影权重（列并行切分后），形状 (D, 3H/P)。
        w_o: Output 投影权重（行并行切分后），形状 (H/P, D)。
        w_gate_up: Gate+Up 投影权重（列并行切分后），形状 (D, 2H/P)。
        w_down: Down 投影权重（行并行切分后），形状 (H/P, D)。
        use_sp: 是否启用序列并行。默认 True。

    Returns:
        输出张量，形状 (B, S, D)。

    Note:
        当前为简化演示实现，主要展示数据流和通信模式，
        未包含完整的 Attention 和 FFN 计算逻辑。
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
