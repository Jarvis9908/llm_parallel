"""GPipe 流水线调度。所有 micro-batch 先 forward 完毕，再统一 backward。

直觉
----
想象一条汽车装配线：焊接→喷漆→组装，每个工位是一个 stage（一组层）。
GPipe 的做法是先把所有 mini-car 全部推过焊接站，再全部推过喷漆站，
最后全部推过组装站——即所有 micro-batch 先做完 forward，再统一做 backward。

数学
----
1. GPipe 调度时间线（P 个 stage，M 个 micro-batch）：

    Stage 0: [F0 F1 F2 ... FM-1] [B0 B1 B2 ... BM-1]
    Stage 1:  . [F0 F1 F2 ... FM-1] [B0 B1 B2 ... BM-1]
    Stage 2:  .  . [F0 F1 F2 ... FM-1] [B0 B1 B2 ... BM-1]

   其中 Fi = forward micro-batch i，Bi = backward micro-batch i

2. Bubble ratio（空闲时间占比）：
    bubble_ratio = (P - 1) / (P - 1 + M)
   当 M >> P 时，bubble 趋近于 0；但 M 受显存限制。

3. 激活显存峰值：
    峰值 = M × 单个 micro-batch 的激活大小
   因为所有 M 个 micro-batch 的 forward 激活都需要保存到 backward 阶段使用，
   这是 GPipe 的主要缺点。

代码流程
--------
1. ``gpiped_forward`` —— 对所有 micro-batch 依次执行 forward
2. ``compute_gpipe_bubble_time`` —— 计算 bubble 时间占比
"""
import torch
from typing import Callable


def gpiped_forward(
    micro_batches: list[torch.Tensor],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """
    GPipe 风格：所有 micro-batch 依次 forward。

    直觉：像把 M 辆小车依次推过当前工位，全部推完之后再统一做 backward。

    数学：
        outputs = [forward(mb₀), forward(mb₁), ..., forward(mb_{M-1})]
        激活显存峰值 = M × 单个 micro-batch 的激活大小

    注意：此函数只做 forward，所有中间激活都会被保留到 backward 阶段，
    因此激活显存峰值与 micro-batch 数量 M 成正比，这是 GPipe 的主要瓶颈。

    Args:
        micro_batches: 将一个 mini-batch 切分后的 M 个 micro-batch 列表
        forward_fn: 单个 micro-batch 的 forward 函数

    Returns:
        list[torch.Tensor]: 每个 micro-batch 的 forward 输出列表
    """
    return [forward_fn(mb) for mb in micro_batches]


def compute_gpipe_bubble_time(n_micro_batches: int, n_stages: int) -> float:
    """计算 GPipe 的 bubble 时间占比。

    直觉：流水线在启动（填入 micro-batch）和排空（等待最后一个 micro-batch
    走完全部 stage）期间，部分 stage 处于空闲状态，这段空闲就是 bubble。

    数学：
        bubble_ratio = (P - 1) / (P - 1 + M)
        其中 P = n_stages（流水线深度），M = n_micro_batches

        推导：理想情况下 M 个 micro-batch 经过 P 个 stage 需要 M+P-1 个
        时间步，而实际计算量为 M×P 个 stage-step，因此空闲时间 =
        (M+P-1)×P - M×P = P(P-1)，但归一化后 bubble_ratio = (P-1)/(P-1+M)。

    Args:
        n_micro_batches: micro-batch 数量 M
        n_stages: 流水线 stage 数量 P

    Returns:
        float: bubble 时间占比，取值 [0, 1)
    """
    return (n_stages - 1) / (n_stages - 1 + n_micro_batches)


if __name__ == "__main__":
    print("=== GPipe demo ===")

    # 模拟 forward
    forward_fn = lambda x: x * 2
    micro_batches = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    outputs = gpiped_forward(micro_batches, forward_fn)
    print(f"GPipe forward outputs: {outputs}")

    # Bubble time 分析
    for n_mb in [4, 8, 16, 32]:
        for n_stages in [2, 4, 8]:
            bubble = compute_gpipe_bubble_time(n_mb, n_stages)
            print(f"n_mb={n_mb:2d}, n_stages={n_stages:2d} -> bubble_time={bubble:.4f}")
