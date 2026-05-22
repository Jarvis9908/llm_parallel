"""1F1B 流水线调度。交错 forward 和 backward，减少 activation 显存峰值。

直觉
----
GPipe 的问题是所有 forward 做完才做 backward——这导致 M 个 micro-batch
的激活全部驻留显存。1F1B 的改进思路：forward 完一个 micro-batch 后，
如果流水线已经填满，就立刻做 backward 释放该 micro-batch 的激活，
从而将峰值从 M 降到 P（stage 数）。

数学
----
1F1B 调度分为三个阶段：

1. Warmup 阶段：连续做 P-1 个 forward（逐步填满流水线）
   - 激活累积：1, 2, ..., P-1

2. Steady 阶段：交替做 1F + 1B（保持流水线满载，同时释放旧激活）
   - 激活维持在 P 个（每做 1 个 B 就释放 1 个，同时新增 1 个 F）
   - 这是 1F1B 的核心：激活峰值 = P，远小于 GPipe 的 M

3. Cooldown 阶段：处理剩余 P-1 个 backward（流水线排空）
   - 激活逐步减少：P-1, P-2, ..., 1, 0

关键结论：
- Bubble ratio 与 GPipe 相同：(P-1)/(P-1+M)
- 激活峰值：P（1F1B）vs M（GPipe），当 M >> P 时优势显著

代码流程
--------
1. ``f1b1_schedule`` —— 执行 1F1B 调度（简化演示版）
2. ``compute_1f1b_bubble_time`` —— 计算 bubble 时间占比
"""
import torch


def f1b1_schedule(
    micro_batches: list[torch.Tensor],
    forward_fn,
    backward_fn,
    n_warmup: int = 3,
) -> list[torch.Tensor]:
    """
    1F1B (One Forward One Backward) 调度。

    直觉：先填满流水线（Warmup），然后边进边出（Steady），最后排空（Cooldown）。
    就像高速公路收费站：先让车一辆辆进入（Warmup），然后一辆进一辆出（Steady），
    最后等里面的车全部离开（Cooldown）。

    数学：
        Warmup 阶段：执行 P-1 个 forward，激活累积到 P-1
        Steady 阶段：执行 M-P+1 轮 (1F + 1B)，激活维持在 P
        Cooldown 阶段：执行 P-1 个 backward，激活从 P 降到 0
        激活峰值 = P（vs GPipe 的 M）

    Args:
        micro_batches: 将一个 mini-batch 切分后的 M 个 micro-batch 列表
        forward_fn: 单个 micro-batch 的 forward 函数
        backward_fn: 单个 loss 的 backward 函数
        n_warmup: Warmup 阶段的 forward 次数，默认为 stage 数 - 1

    Returns:
        list[torch.Tensor]: 所有 micro-batch 的 loss 列表
    """
    losses = []
    # Simplified demo: just do F then B for each micro-batch
    for mb in micro_batches:
        out = forward_fn(mb)
        loss = out.sum()
        backward_fn(loss)
        losses.append(loss)
    return losses


def compute_1f1b_bubble_time(n_micro_batches: int, n_stages: int) -> float:
    """计算 1F1B 调度的 bubble 时间占比。

    直觉：1F1B 的 bubble 与 GPipe 相同——都来自流水线的启动和排空。
    1F1B 的优势不在减少 bubble，而在降低激活显存峰值（P vs M）。

    数学：
        bubble_ratio = (P - 1) / (P - 1 + M)
        与 GPipe 相同，但激活峰值从 M 降到 P。

    Args:
        n_micro_batches: micro-batch 数量 M
        n_stages: 流水线 stage 数量 P

    Returns:
        float: bubble 时间占比
    """
    n_warmup = n_stages - 1
    total_steps = 2 * (n_warmup + n_micro_batches) - 1  # 简化计算
    idle_steps = n_warmup * 2
    return idle_steps / total_steps


def simulate_1f1b_timeline(
    n_micro_batches: int, n_stages: int, forward_time: float = 1.0, backward_time: float = 2.0
) -> dict:
    """
    模拟 1F1B 调度的完整时间线。

    直觉：把 1F1B 调度的三个阶段（warmup、steady、cooldown）的
    每个操作的时间戳精确计算出来，用于可视化和分析。

    数学：
        Warmup 阶段（stage s 做 n_warmup - s 个 forward）：
            n_warmup = n_stages - 1
            Stage s 的 warmup forward 数 = n_warmup - s

        Steady 阶段（1F1B 交替）：
            每个 stage 做 M - n_warmup + s 轮 1F+1B

        Cooldown 阶段（处理剩余 backward）：
            Stage s 的 cooldown backward 数 = s + 1

    Args:
        n_micro_batches: micro-batch 数量 M
        n_stages: 流水线 stage 数量 P
        forward_time: 单次 forward 耗时
        backward_time: 单次 backward 耗时

    Returns:
        包含 timeline（每个 stage 的操作列表）、total_time、bubble_ratio 的字典
    """
    n_warmup = n_stages - 1
    timeline = {s: [] for s in range(n_stages)}

    for stage in range(n_stages):
        current_time = stage * forward_time  # 每个 stage 的启动延迟
        warmup_count = n_warmup - stage
        steady_count = n_micro_batches - warmup_count

        # Warmup: 连续 forward
        for i in range(warmup_count):
            mb_idx = i
            timeline[stage].append({
                'type': 'forward',
                'micro_batch': mb_idx,
                'start': current_time,
                'end': current_time + forward_time,
            })
            current_time += forward_time

        # Steady: 1F1B 交替
        for i in range(steady_count):
            mb_fwd = warmup_count + i
            mb_bwd = i
            timeline[stage].append({
                'type': 'forward',
                'micro_batch': mb_fwd,
                'start': current_time,
                'end': current_time + forward_time,
            })
            current_time += forward_time
            timeline[stage].append({
                'type': 'backward',
                'micro_batch': mb_bwd,
                'start': current_time,
                'end': current_time + backward_time,
            })
            current_time += backward_time

        # Cooldown: 剩余 backward
        for i in range(stage + 1):
            mb_bwd = steady_count + i
            if mb_bwd < n_micro_batches:
                timeline[stage].append({
                    'type': 'backward',
                    'micro_batch': mb_bwd,
                    'start': current_time,
                    'end': current_time + backward_time,
                })
                current_time += backward_time

    # 计算总时间和 bubble
    total_time = max(
        max(op['end'] for op in ops) for ops in timeline.values()
    )
    ideal_time = n_micro_batches * (forward_time + backward_time)
    bubble_ratio = 1 - ideal_time / (total_time * n_stages)

    return {
        'timeline': timeline,
        'total_time': total_time,
        'ideal_time': ideal_time,
        'bubble_ratio': bubble_ratio,
    }


if __name__ == "__main__":
    print("=== 1F1B demo ===")

    # 模拟 1F1B 调度
    micro_batches = [torch.ones(3, 3) for _ in range(5)]
    backward_called = []

    def forward_fn(x):
        return x + 1

    def backward_fn(loss):
        backward_called.append(loss.item())

    losses = f1b1_schedule(micro_batches, forward_fn, backward_fn, n_warmup=2)
    print(f"Number of micro-batches: {len(losses)}")
    print(f"Backward calls: {len(backward_called)}")

    # Bubble time 对比
    print("\n1F1B vs GPipe bubble time comparison:")
    for n_mb in [4, 8, 16, 32]:
        for n_stages in [2, 4, 8]:
            b1f1b = compute_1f1b_bubble_time(n_mb, n_stages)
            # Re-use GPipe公式
            b_gpipe = (n_stages - 1) / (n_stages - 1 + n_mb)
            print(f"n_mb={n_mb:2d}, stages={n_stages}: 1F1B={b1f1b:.4f}, GPipe={b_gpipe:.4f}")

    # 时间线模拟
    print("\n1F1B Timeline Simulation (4 stages, 8 micro-batches):")
    result = simulate_1f1b_timeline(n_micro_batches=8, n_stages=4, forward_time=1.0, backward_time=2.0)
    for stage in range(4):
        ops = result['timeline'][stage]
        print(f"  Stage {stage}: {len(ops)} ops, total time = {max(op['end'] for op in ops):.1f}")
    print(f"  Total time: {result['total_time']:.1f}")
    print(f"  Bubble ratio: {result['bubble_ratio']:.4f}")
