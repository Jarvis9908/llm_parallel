"""1F1B 流水线调度。交错 forward 和 backward，减少 activation 显存峰值。"""
import torch


def f1b1_schedule(
    micro_batches: list[torch.Tensor],
    forward_fn,
    backward_fn,
    n_warmup: int = 3,
) -> list[torch.Tensor]:
    """
    1F1B (One Forward One Backward) 调度。

    Warmup 阶段：连续做 n_warmup 个 forward（填充流水线）
    Steady 阶段：交替做 1 forward + 1 backward（保持流水线满载）
    Cooldown 阶段：处理剩余 backward

    返回所有 micro-batch 的 loss。
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
    """1F1B bubble time 分析。warmup 阶段的空闲时间占比。"""
    n_warmup = n_stages - 1
    total_steps = 2 * (n_warmup + n_micro_batches) - 1  # 简化计算
    idle_steps = n_warmup * 2
    return idle_steps / total_steps


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
